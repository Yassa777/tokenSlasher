"""TokenSlasher unified command-line interface.

Usage
-----
$ tokenslasher run config.yml
$ tokenslasher diff old_corpus_dir new_corpus_dir

The *run* command executes the duplicate-detection pipeline based on a YAML
configuration file, then emits an interactive Plotly HTML dashboard
(*results/dashboard.html* by default).

The *diff* command compares two corpora and prints a short report on token
bloat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml  # type: ignore

try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except ImportError:  # pragma: no cover
    px = None  # type: ignore
    go = None  # type: ignore

from .detector.dedup import process_corpus
from .detector.ingest import slab_generator, tokenize, SENTINEL

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _tokens_in_dir(path: Path) -> int:
    """Return total token count (quick approximation) for files under *path*."""
    total = 0
    for fp in path.glob("**/*"):
        if not fp.is_file():
            continue
        for slab in slab_generator(fp):
            total += len(slab)
    return total


def _build_dashboard(results_dir: Path) -> None:  # pragma: no cover – viz only
    """Create an interactive HTML dashboard inside *results_dir*."""
    if px is None:
        print("Plotly not installed – skipping dashboard generation.", file=sys.stderr)
        return

    speed_path = results_dir / "speed.txt"
    dup_pairs_path = results_dir / "samples.jsonl"

    if not speed_path.exists():
        print(f"[dashboard] Missing {speed_path}; skipping dashboard.", file=sys.stderr)
        return

    # Timing breakdown – currently only throughput, can be expanded later.
    with speed_path.open() as f:
        header = next(f).strip().split("\t")
        values = next(f).strip().split("\t")
        timing = {k: float(v) if k != "tokens_processed" else int(v) for k, v in zip(header, values)}

    fig_timing = px.bar(
        x=["seconds", "throughput_tokens_per_s"],
        y=[timing["seconds"], timing["throughput_tokens_per_s"],],
        labels={"x": "Metric", "y": "Value"},
        title="Timing breakdown",
    )

    # Heat-map placeholder – uses duplicate sample pairs counts per source file.
    sources: Dict[str, int] = {}
    if dup_pairs_path.exists():
        with dup_pairs_path.open() as f:
            for line in f:
                rec = json.loads(line)
                for side in ("a", "b"):
                    key = rec[side].split("_")[0]  # doc_123 → doc
                    sources[key] = sources.get(key, 0) + 1
    if sources:
        heatmap_fig = px.density_heatmap(
            x=list(sources.keys()),
            y=["token_saves"] * len(sources),
            z=list(sources.values()),
            title="Token-saving heat-map per source",
            labels={"x": "Source", "z": "Dup pairs"},
        )
    else:
        heatmap_fig = go.Figure()
        heatmap_fig.update_layout(title="Token-saving heat-map per source (no data)")

    # ROC curve placeholder – requires ground-truth; here we emit an empty fig.
    roc_fig = go.Figure()
    roc_fig.update_layout(title="ROC curve (ground-truth not available)")

    # Compose dashboard
    from plotly.subplots import make_subplots  # type: ignore

    dashboard = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Timing", "Token savings", "ROC curve"),
        vertical_spacing=0.15,
    )
    for trace in fig_timing.data:
        dashboard.add_trace(trace, row=1, col=1)
    for trace in heatmap_fig.data:
        dashboard.add_trace(trace, row=2, col=1)
    for trace in roc_fig.data:
        dashboard.add_trace(trace, row=3, col=1)

    dashboard.update_layout(height=1200, showlegend=False, title_text="TokenSlasher report")

    out_path = results_dir / "dashboard.html"
    dashboard.write_html(str(out_path))
    print(f"[dashboard] Written to {out_path.relative_to(Path.cwd())}")


# -----------------------------------------------------------
# Commands
# -----------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> None:
    cfg_path: Path = args.config.resolve()
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    # Required keys with defaults
    data_dir = Path(cfg["data_dir"]).expanduser().resolve()
    out_dir = Path(cfg.get("out", "results")).expanduser().resolve()

    process_corpus(
        data_dir=data_dir,
        ngram=cfg.get("ngram", "auto"),
        threshold=float(cfg.get("threshold", 0.8)),
        topk=int(cfg.get("topk", 5000)),
        out_dir=out_dir,
        processes=int(cfg.get("processes", 8)),
        metric=str(cfg.get("metric", "jaccard")),
    )

    # Build dashboard automatically.
    _build_dashboard(out_dir)


def _cmd_diff(args: argparse.Namespace) -> None:
    old_dir = Path(args.old).expanduser().resolve()
    new_dir = Path(args.new).expanduser().resolve()

    print("Computing token statistics – this may take a while…")
    old_tokens = _tokens_in_dir(old_dir)
    new_tokens = _tokens_in_dir(new_dir)

    delta = new_tokens - old_tokens
    growth_pct = (delta / max(old_tokens, 1)) * 100

    print("\n======= TokenSlasher diff =======")
    print(f"Old corpus tokens : {old_tokens:,}")
    print(f"New corpus tokens : {new_tokens:,}")
    print(f"Δ tokens         : {delta:+,} ({growth_pct:+.2f}%)")


# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------


def main(argv: List[str] | None = None) -> None:  # noqa: D401 – simple
    parser = argparse.ArgumentParser(prog="tokenslasher", description="TokenSlasher toolkit")
    sub = parser.add_subparsers(required=True, dest="cmd")

    # run
    p_run = sub.add_parser("run", help="Run duplicate-detection pipeline via YAML config")
    p_run.add_argument("config", type=Path, help="Path to YAML configuration file")
    p_run.set_defaults(func=_cmd_run)

    # diff
    p_diff = sub.add_parser("diff", help="Compare two corpora for incremental bloat")
    p_diff.add_argument("old", type=Path, help="Old corpus directory")
    p_diff.add_argument("new", type=Path, help="New corpus directory")
    p_diff.set_defaults(func=_cmd_diff)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main() 