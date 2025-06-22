"""Streamlit dashboard for exploring TokenSlasher near-duplicate clusters.

Run with:
    streamlit run tokenSlasher/dashboard_app.py

It expects a *report.json* file produced by the dedup pipeline.  The minimal
schema:

{
  "summary": {
      "total_documents": int,
      "duplicate_clusters": int,
      "tokens_removed": int,
      "redundancy_pct": float,
      "runtime_sec": float
  },
  "clusters": [
      {
         "similarity": 0.91,          # representative score
         "documents": [
              {
                  "id": "doc_123",
                  "text": "full document text …",
                  "tokens": 250,
                  "source": "Wikipedia"
              },
              …
         ]
      },
      …
  ]
}

If *report.json* is missing the user can upload their own file.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

DEFAULT_REPORT = Path("results/report.json")

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _load_report(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _summarise(report: Dict[str, Any]) -> None:
    s = report.get("summary", {})
    st.metric("Total documents", f"{s.get('total_documents', 0):,}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters", f"{s.get('duplicate_clusters', 0):,}")
    col2.metric("Tokens removed", f"{s.get('tokens_removed', 0):,}")
    col3.metric("Redundancy %", f"{s.get('redundancy_pct', 0):.2f} %")
    col4.metric("Runtime", f"{s.get('runtime_sec', 0):.1f} s")


def _filter_clusters(clusters: List[Dict[str, Any]], similarity_min: float, size_min: int, sources: List[str]) -> List[Dict[str, Any]]:
    out = []
    for c in clusters:
        if c.get("similarity", 0) < similarity_min:
            continue
        docs = c.get("documents", [])
        if len(docs) < size_min:
            continue
        if sources:
            if not any(d.get("source") in sources for d in docs):
                continue
        out.append(c)
    return out


def _render_diff(doc_a: str, doc_b: str) -> None:
    diff_html = difflib.HtmlDiff().make_table(doc_a.split(), doc_b.split(), context=True, numlines=5)
    st.markdown("""<style>table.diff {font-size: 0.8em;}</style>""", unsafe_allow_html=True)
    st.components.v1.html(diff_html, height=300, scrolling=True)


# -----------------------------------------------------------
# UI definition
# -----------------------------------------------------------

def main() -> None:  # pragma: no cover
    st.set_page_config(page_title="TokenSlasher Dashboard", layout="wide")
    st.title("🔪 TokenSlasher Duplicate-Detection Report")

    # Sidebar – file selection and filters
    with st.sidebar:
        st.header("Input report")
        if DEFAULT_REPORT.exists():
            default_choice = "Default (results/report.json)"
        else:
            default_choice = "Upload"
        report_source = st.radio("Report file", [default_choice, "Upload"])

        if report_source == default_choice and DEFAULT_REPORT.exists():
            report_path = DEFAULT_REPORT
            st.success(f"Using {report_path}")
            report = _load_report(report_path)
        else:
            uploaded = st.file_uploader("Upload report.json", type=["json"])
            if uploaded is None:
                st.warning("Please upload a report.json file.")
                st.stop()
            report = json.load(uploaded)

        # Filters
        st.markdown("---")
        st.subheader("Filters")
        similarity_min = st.slider("Min similarity", 0.0, 1.0, 0.85, 0.01)
        size_min = st.number_input("Min cluster size", 2, 100, 3)

        # Gather sources list
        all_sources = sorted({d.get("source", "?") for c in report["clusters"] for d in c.get("documents", [])})
        source_filter = st.multiselect("Sources", options=all_sources, default=[])

        token_min, token_max = st.slider("Token count range", 0, 5000, (0, 5000))
        search_kw = st.text_input("Keyword search")

    # Tabs
    overview_tab, clusters_tab, stats_tab = st.tabs(["Overview", "Clusters", "Stats"])

    with overview_tab:
        _summarise(report)

    with clusters_tab:
        clusters = _filter_clusters(report["clusters"], similarity_min, size_min, source_filter)
        st.write(f"Showing {len(clusters):,} clusters after filters.")

        for idx, cluster in enumerate(clusters, 1):
            sim = cluster.get("similarity", 0.0)
            docs = cluster.get("documents", [])
            # Additional doc-level filters
            docs = [d for d in docs if token_min <= d.get("tokens", 0) <= token_max]
            if search_kw:
                docs = [d for d in docs if search_kw.lower() in d.get("text", "").lower()]
            if not docs:
                continue

            with st.expander(f"Cluster {idx}  –  sim≈{sim:.2f}  –  {len(docs)} docs", expanded=False):
                doc_cols = st.columns([1, 10, 1])
                sel_box_key = f"sel_{idx}"
                selections: List[str] = st.session_state.get(sel_box_key, [])
                for doc in docs:
                    id_ = doc.get("id")
                    text = doc.get("text", "")[:300] + ("…" if len(doc.get("text", "")) > 300 else "")
                    with doc_cols[1]:
                        st.markdown(f"**{id_}**  (tokens={doc.get('tokens')})")
                        st.markdown(f"`{doc.get('source','?')}`")
                        st.markdown(text)
                    with doc_cols[2]:
                        st.button("Copy ID", key=f"copy_{idx}_{id_}", on_click=st.write, args=(id_,))
                        if st.checkbox("Select", key=f"sel_{idx}_{id_}"):
                            selections.append(id_)
                st.session_state[sel_box_key] = selections
                if len(selections) == 2:
                    st.markdown("---")
                    st.subheader("Diff of selected docs")
                    texts = {d["id"]: d["text"] for d in docs if d["id"] in selections}
                    _render_diff(texts[selections[0]], texts[selections[1]])

    with stats_tab:
        st.json(report.get("summary", {}), expanded=False)
        st.write("Raw clusters JSON snippet:")
        st.json(report["clusters"][:3])


if __name__ == "__main__":  # pragma: no cover
    main() 