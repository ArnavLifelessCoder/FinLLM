"""Checkpoint save/load helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from finllm.config import ModelConfig, TrainConfig
from finllm.model import FinanceGPT


def save_checkpoint(
    path: str | Path,
    *,
    model: FinanceGPT,
    optimizer: torch.optim.Optimizer | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    iteration: int,
    best_val_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "iter_num": iteration,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


def load_model_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> FinanceGPT:
    checkpoint = torch.load(path, map_location=map_location)
    config_payload = dict(checkpoint["model_config"])
    if "architecture" not in config_payload:
        config_payload["architecture"] = "legacy"
    model_config = ModelConfig(**config_payload)
    model = FinanceGPT(model_config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    return model
