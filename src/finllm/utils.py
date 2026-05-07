"""Runtime utilities shared by CLIs."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "float32" or device == "cpu":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def cosine_lr(
    iteration: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    if iteration < warmup_iters:
        return learning_rate * iteration / max(1, warmup_iters)
    if iteration > lr_decay_iters:
        return min_lr
    decay_ratio = (iteration - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def worker_count(default: int = 0) -> int:
    return int(os.environ.get("FINLLM_WORKERS", default))

