"""Train a SentencePiece tokenizer for the financial corpus."""

from __future__ import annotations

import argparse

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument("--input", default="financial_training_corpus_clean.txt")
    parser.add_argument("--model-prefix", default="finance_tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram", "char", "word"])
    parser.add_argument("--character-coverage", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )


if __name__ == "__main__":
    main()

