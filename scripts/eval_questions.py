#!/usr/bin/env python
"""Evaluate TokenSlasher similarity against Quora-style questions.csv.

Expected CSV columns:
    qid1, qid2, question1, question2, is_duplicate

Outputs
-------
1. `results/questions_metrics.txt` – ROC-AUC, PR-AUC, optimal F1 threshold.
2. `results/roc_curve.png` and `results/pr_curve.png` – plots.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path when executed as a script *before* local imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Local imports (after sys.path tweak)
from detector.ingest import ngrams, tokenize
from detector.minhash import compute_minhash

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def build_minhashes(questions: pd.Series, n: int) -> Dict[int, "MinHash"]:  # noqa: F821
    """Return dict mapping row index → MinHash of *question* text."""
    cache: Dict[str, "MinHash"] = {}
    mh_dict: Dict[int, "MinHash"] = {}
    for idx, text in questions.items():
        if text in cache:
            mh_dict[idx] = cache[text]
            continue
        toks = tokenize(text)
        mh = compute_minhash(ngrams(toks, n=n))
        cache[text] = mh
        mh_dict[idx] = mh
    return mh_dict


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------


def eval_questions(csv_path: Path, ngram: int, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    if not {"question1", "question2", "is_duplicate"}.issubset(df.columns):
        raise ValueError("CSV missing required columns")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build MinHash for each unique question only once.
    mh_cache = build_minhashes(pd.concat([df["question1"], df["question2"]]), ngram)

    scores: List[float] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring pairs"):
        mh1 = mh_cache[idx] if idx in mh_cache else compute_minhash(ngrams(tokenize(row["question1"]), n=ngram))
        mh2 = mh_cache[idx + len(df)] if (idx + len(df)) in mh_cache else compute_minhash(ngrams(tokenize(row["question2"]), n=ngram))
        scores.append(mh1.jaccard(mh2))

    y_true = df["is_duplicate"].astype(int).values
    y_score = np.array(scores)

    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)

    precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    # Optimal threshold = max F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_scores))
    best_thresh = pr_thresh[best_idx] if best_idx < len(pr_thresh) else 0.5
    best_f1 = f1_scores[best_idx]

    metrics_txt = (
        f"ROC_AUC: {roc_auc:.4f}\n"
        f"PR_AUC:  {pr_auc:.4f}\n"
        f"Best_F1: {best_f1:.4f} at threshold {best_thresh:.4f}\n"
    )
    (out_dir / "questions_metrics.txt").write_text(metrics_txt)
    print(metrics_txt)

    # Plots
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Questions Dedup")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)

    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where="post", label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve – Questions Dedup")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150)


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate questions.csv duplicate labels")
    p.add_argument("--csv", type=Path, default=Path("data/questions.csv"))
    p.add_argument("--ngram", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    eval_questions(args.csv, args.ngram, args.out)


if __name__ == "__main__":
    main() 