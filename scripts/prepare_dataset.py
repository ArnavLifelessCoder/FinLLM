"""Build train.bin/val.bin from a text corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from finllm.data import build_binary_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare binary token datasets for FinLLM.")
    parser.add_argument("--corpus", default="financial_training_corpus_clean.txt")
    parser.add_argument("--tokenizer", default="finance_tokenizer.model")
    parser.add_argument("--output-dir", default="data/finance")
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-chars", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_binary_dataset(
        corpus_path=args.corpus,
        tokenizer_model=args.tokenizer,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_chars=args.min_chars,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

