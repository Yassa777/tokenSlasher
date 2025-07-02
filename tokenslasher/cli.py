"""TokenSlasher unified command-line interface.

Usage
-----
$ tokenslasher clean input_data/ --output cleaned_data.jsonl
$ tokenslasher preview input_data/
$ tokenslasher run config.yml  # Legacy V1.0 command
$ tokenslasher diff old_corpus_dir new_corpus_dir

The *clean* command (V1.5) executes the complete data cleaning, filtering, and 
deduplication pipeline with modern features.

The *preview* command shows a sample of input data without processing.

The *run* command (Legacy V1.0) executes the duplicate-detection pipeline based on a YAML
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
from .detector.pipeline import run_pipeline, preview_input, estimate_processing_time
from .detector.filters import create_default_pipeline

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

def _cmd_clean(args: argparse.Namespace) -> None:
    """V1.5 clean command - comprehensive data processing pipeline."""
    print(f"🧹 TokenSlasher V1.5 - Cleaning data from {args.input}")
    
    # Build filter pipeline
    filter_kwargs = {}
    if args.languages:
        filter_kwargs['allowed_languages'] = set(args.languages)
    if args.min_length:
        filter_kwargs['min_length'] = args.min_length
    if args.max_perplexity:
        filter_kwargs['max_perplexity'] = args.max_perplexity
    if args.pii_mode:
        filter_kwargs['pii_mode'] = args.pii_mode
    if args.use_toxicity_model:
        filter_kwargs['use_toxicity_model'] = True
    
    filter_pipeline = create_default_pipeline(**filter_kwargs)
    
    # Run pipeline
    stats = run_pipeline(
        input_paths=args.input,
        output_path=args.output,
        filter_pipeline=filter_pipeline,
        enable_dedup=not args.no_dedup,
        dedup_threshold=args.dedup_threshold,
        output_format=args.format,
        shard_size=args.shard_size,
        verbose=not args.quiet
    )
    
    if args.save_stats:
        stats_path = Path(args.output).parent / f"{Path(args.output).stem}_full_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"💾 Detailed stats saved to {stats_path}")


def _cmd_preview(args: argparse.Namespace) -> None:
    """Preview input data."""
    preview_input(args.input, max_samples=args.samples)
    
    if args.estimate_time:
        estimates = estimate_processing_time(args.input, docs_per_second=args.speed)
        print(f"\n⏱️ Processing Time Estimates:")
        print(f"   - Documents: ~{estimates['estimated_documents']:,}")
        print(f"   - Time: ~{estimates['estimated_minutes']:.1f} minutes")
        if estimates['estimated_hours'] > 1:
            print(f"           (~{estimates['estimated_hours']:.1f} hours)")


def _cmd_run(args: argparse.Namespace) -> None:
    """Legacy V1.0 run command."""
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
    """Compare two corpora."""
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
    parser = argparse.ArgumentParser(
        prog="tokenslasher", 
        description="TokenSlasher V1.5 - Complete text data processing toolkit"
    )
    sub = parser.add_subparsers(required=True, dest="cmd")

    # clean (V1.5 main command)
    p_clean = sub.add_parser(
        "clean", 
        help="Clean, filter, and deduplicate text data (V1.5)"
    )
    p_clean.add_argument("input", nargs="+", help="Input files, directories, or glob patterns")
    p_clean.add_argument("-o", "--output", required=True, help="Output file path")
    p_clean.add_argument("--format", default="auto", 
                        choices=["auto", "jsonl", "txt", "parquet"],
                        help="Output format (default: auto-detect from extension)")
    p_clean.add_argument("--shard-size", type=int,
                        help="Records per shard for large outputs")
    p_clean.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication")
    p_clean.add_argument("--dedup-threshold", type=float, default=0.8,
                        help="Deduplication similarity threshold (default: 0.8)")
    p_clean.add_argument("--languages", nargs="+", default=["en"],
                        help="Allowed languages (default: en)")
    p_clean.add_argument("--min-length", type=int, default=50,
                        help="Minimum text length in characters (default: 50)")
    p_clean.add_argument("--max-perplexity", type=float, default=1000,
                        help="Maximum perplexity threshold (default: 1000)")
    p_clean.add_argument("--pii-mode", choices=["filter", "anonymize"], default="anonymize",
                        help="PII handling mode (default: anonymize)")
    p_clean.add_argument("--use-toxicity-model", action="store_true",
                        help="Use ML model for toxicity detection")
    p_clean.add_argument("--save-stats", action="store_true",
                        help="Save detailed statistics to JSON")
    p_clean.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")
    p_clean.set_defaults(func=_cmd_clean)

    # preview
    p_preview = sub.add_parser("preview", help="Preview input data")
    p_preview.add_argument("input", nargs="+", help="Input files, directories, or glob patterns")
    p_preview.add_argument("--samples", type=int, default=5,
                          help="Number of sample documents to show (default: 5)")
    p_preview.add_argument("--estimate-time", action="store_true",
                          help="Estimate processing time")
    p_preview.add_argument("--speed", type=float, default=100,
                          help="Estimated processing speed (docs/sec, default: 100)")
    p_preview.set_defaults(func=_cmd_preview)

    # run (legacy V1.0)
    p_run = sub.add_parser("run", help="Run duplicate-detection pipeline via YAML config (Legacy V1.0)")
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