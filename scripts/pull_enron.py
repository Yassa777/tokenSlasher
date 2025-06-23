#!/usr/bin/env python
"""Download a small slice of Enron Emails from the Pile (uncopyrighted).

Example
-------
python scripts/pull_enron.py --out data/enron.txt --max_docs 2500
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset


def stream_enron(max_docs: int) -> list[str]:
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    docs: list[str] = []
    for example in ds:
        if example["meta"]["pile_set_name"] == "Enron Emails":
            docs.append(example["text"].strip())
            if len(docs) >= max_docs:
                break
    return docs


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Enron Emails slice from HF")
    p.add_argument("--out", type=Path, default=Path("data/enron.txt"), help="Output text file path")
    p.add_argument("--max_docs", type=int, default=2500, help="Number of documents to fetch")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    docs = stream_enron(args.max_docs)
    with args.out.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.replace("\n", " ").strip() + "\n")
    print(f"Wrote {len(docs)} docs to {args.out}")


if __name__ == "__main__":
    main() 