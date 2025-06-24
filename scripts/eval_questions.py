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
import numpy as np

# Ensure repository root is on sys.path when executed as a script *before* local imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Local imports (after sys.path tweak)
from detector.ingest import tokenize, char_ngrams, token_skipgrams
from detector.minhash import batch_xxhash64
from detector.semantic import embed_text, embed_texts, cosine_similarity

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _make_shingle_set(text: str) -> set[int]:
    """Return hashed shingle set for *text*."""
    toks = tokenize(text)
    shingles = set(char_ngrams(text, 5)).union(token_skipgrams(toks, skip=1))
    return set(batch_xxhash64(list(shingles)))


def build_caches(questions: pd.Series) -> Tuple[Dict[int, set[int]], Dict[int, "np.ndarray"]]:  # noqa: F821
    # Build shingle sets
    unique_texts = questions.unique().tolist()
    unique_texts = ["" if isinstance(t, float) else t for t in unique_texts]

    shingle_cache: Dict[str, set[int]] = {t: _make_shingle_set(t) for t in unique_texts}

    # Batch embeddings once
    embs = embed_texts(unique_texts, batch_size=128)
    embed_cache: Dict[str, "np.ndarray"] = {t: e for t, e in zip(unique_texts, embs)}

    sh_map: Dict[int, set[int]] = {}
    emb_map: Dict[int, "np.ndarray"] = {}
    for idx, text in questions.items():
        sh_map[idx] = shingle_cache[text]
        emb_map[idx] = embed_cache[text]

    return sh_map, emb_map


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------


def eval_questions(csv_path: Path, ngram: int, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    if not {"question1", "question2", "is_duplicate"}.issubset(df.columns):
        raise ValueError("CSV missing required columns")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate questions and reset index so second half is offset by len(df)
    combined_q = pd.concat([df["question1"], df["question2"]], ignore_index=True)
    shingles_map, emb_map = build_caches(combined_q)

    lexical_scores: List[float] = []
    sem_scores: List[float] = []
    blend_scores: List[float] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring pairs"):
        s1 = shingles_map[idx]
        s2 = shingles_map[idx + len(df)]
        lex = len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0

        emb1 = emb_map[idx]
        emb2 = emb_map[idx + len(df)]
        sem = cosine_similarity(emb1, emb2)

        blend = 0.6 * lex + 0.4 * sem

        lexical_scores.append(lex)
        sem_scores.append(sem)
        blend_scores.append(blend)

    y_true = df["is_duplicate"].astype(int).values
    y_score = np.array(blend_scores)

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