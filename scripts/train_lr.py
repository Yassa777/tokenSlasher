#!/usr/bin/env python
"""Train logistic-regression duplicate classifier using questions.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

from detector.ingest import tokenize, char_ngrams, token_skipgrams
from detector.minhash import batch_xxhash64
from detector.semantic import embed_texts, cosine_similarity
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/questions.csv"))
    p.add_argument("--out", type=Path, default=Path("results/lr_model.joblib"))
    p.add_argument("--sample", type=int, default=100_000, help="limit rows for speed")
    args = p.parse_args()

    df = pd.read_csv(args.csv).sample(frac=1, random_state=42)  # shuffle
    if args.sample:
        df = df.head(args.sample)

    # -----------------------
    # Build caches (shingles & embeddings) once
    # -----------------------
    uniq_texts = pd.Series(pd.concat([df["question1"], df["question2"]]).unique())

    def _make_shingle(text: str) -> set[int]:
        toks = tokenize(text)
        shingles = set(char_ngrams(text, 5)).union(token_skipgrams(toks, skip=1))
        return set(batch_xxhash64(list(shingles)))

    sh_cache = {t: _make_shingle(str(t)) for t in tqdm(uniq_texts, desc="Shingles")}

    # Batch-embed with progress bar
    embeddings = []
    BATCH = 64
    for i in tqdm(range(0, len(uniq_texts), BATCH), desc="Embeddings"):
        chunk = uniq_texts.iloc[i : i + BATCH].tolist()
        emb_chunk = embed_texts(chunk, batch_size=BATCH)
        embeddings.extend(list(emb_chunk))

    embed_cache = {t: e for t, e in zip(uniq_texts, embeddings)}

    # -----------------------
    # Feature matrix
    # -----------------------
    features: List[np.ndarray] = []
    labels: List[int] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Feature rows"):
        q1 = str(row["question1"])
        q2 = str(row["question2"])

        s1 = sh_cache[q1]
        s2 = sh_cache[q2]

        len1 = len(q1)
        len2 = len(q2)
        tlen1 = len(tokenize(q1))
        tlen2 = len(tokenize(q2))

        lex_jacc = len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0
        len_ratio = min(len1, len2) / max(len1, len2 or 1)
        tok_ratio = min(tlen1, tlen2) / max(tlen1, tlen2 or 1)
        toks1 = set(tokenize(q1))
        toks2 = set(tokenize(q2))
        tok_overlap = len(toks1 & toks2) / len(toks1 | toks2) if toks1 and toks2 else 0.0

        sem_cos = cosine_similarity(embed_cache[q1], embed_cache[q2])

        feats = np.array([
            lex_jacc,
            sem_cos,
            len_ratio,
            tok_ratio,
            tok_overlap,
            abs(len1 - len2),
            abs(tlen1 - tlen2),
        ], dtype="float32")

        features.append(feats)
        labels.append(int(row["is_duplicate"]))

    X = np.vstack(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr = auc(rec, prec)

    print(f"ROC_AUC={roc:.3f}  PR_AUC={pr:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out)
    print("Saved model to", args.out)


if __name__ == "__main__":
    main() 