"""Evaluate validation loss and perplexity for a checkpoint."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext

import torch

from finllm.checkpoint import load_model_checkpoint
from finllm.config import TrainConfig
from finllm.data import load_splits
from finllm.train import estimate_loss
from finllm.utils import resolve_device, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FinLLM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/finance")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype("auto", device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    model = load_model_checkpoint(args.checkpoint, map_location=device).to(device)
    train_data, val_data = load_splits(args.data_dir, model.config.block_size)
    train_config = TrainConfig(batch_size=args.batch_size, eval_iters=args.eval_iters)
    losses = estimate_loss(
        model=model,
        train_data=train_data,
        val_data=val_data,
        train_config=train_config,
        device=device,
        ctx=ctx,
    )
    print(f"train_loss={losses['train']:.4f}")
    print(f"val_loss={losses['val']:.4f}")
    print(f"val_perplexity={math.exp(losses['val']):.2f}")


if __name__ == "__main__":
    main()

