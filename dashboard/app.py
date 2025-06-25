import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TokenSlasher Dashboard", layout="wide")

results_dir = st.sidebar.text_input("Results directory", "results")
err_path = Path(results_dir) / "errors.jsonl"
metrics_path = Path(results_dir) / "questions_metrics.txt"

st.title("📊 TokenSlasher Evaluation Dashboard")

# Intro markdown panel
st.markdown(
    """
    **Welcome!**  This dashboard helps you explore TokenSlasher's duplicate-detection
    results.

    • **Summary** – high-level metrics (ROC-AUC, PR-AUC, throughput).  
    • **Inspect Duplicates** – interactive table of the top false-positives / false-negatives; use the score slider to focus on edge-cases.  
    • **Deep Dive** – raw JSON record viewer so you can copy-paste into notebooks for further analysis.
    """
)

# Step-by-step tabs
tab_summary, tab_inspect, tab_deep = st.tabs([
    "1. Summary",
    "2. Inspect Duplicates",
    "3. Deep Dive",
])

with tab_summary:
    st.caption("High-level view of how well the classifier performed and how fast it ran.")
    if metrics_path.exists():
        st.text(metrics_path.read_text())
    else:
        st.info("Run eval_questions.py first to generate metrics.txt.")

    speed_file = Path(results_dir) / "speed_eval.txt"
    if speed_file.exists():
        st.subheader("Speed / Memory")
        st.text(speed_file.read_text())


with tab_inspect:
    st.caption("False-positive / false-negative table with quick filters.")
    if err_path.exists():
        data = [json.loads(l) for l in err_path.read_text().splitlines()]
        df = pd.DataFrame(data)

        min_score, max_score = float(df["blend"].min()), float(df["blend"].max())
        thresh = st.slider("Blend score >=", min_value=min_score, max_value=max_score, value=min_score, step=0.05)
        show_df = df[df["blend"] >= thresh]
        st.dataframe(show_df, height=400)
    else:
        st.info("errors.jsonl not found – run evaluation first.")


with tab_deep:
    st.caption("Raw record viewer for custom analysis.")
    if err_path.exists():
        raw = err_path.read_text()
        st.download_button("Download errors.jsonl", raw, file_name="errors.jsonl")
        st.text_area("Preview (first 2k chars)", raw[:2000], height=300)
    else:
        st.info("No error file to show.") 