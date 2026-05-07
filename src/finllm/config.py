"""Configuration helpers for model and training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "modern"
    vocab_size: int = 32_000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    mlp_type: str = "swiglu"
    mlp_hidden_mult: float = 4.0
    use_rope: bool = True
    rope_base: int = 10_000
    num_kv_heads: int | None = None


@dataclass(slots=True)
class TrainConfig:
    data_dir: str = "data/finance"
    tokenizer_model: str = "finance_tokenizer.model"
    out_dir: str = "runs/finance-small"
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_iters: int = 2_000
    eval_interval: int = 100
    eval_iters: int = 50
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    lr_decay_iters: int = 2_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    resume_from: str | None = None
    init_from: str | None = None
    cpu_threads: int | None = None
    top_p: float = 0.95
    repetition_penalty: float = 1.05


T = TypeVar("T")


def _filtered_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf8"))
    return _filtered_dataclass(ModelConfig, raw.get("model", {})), _filtered_dataclass(
        TrainConfig, raw.get("train", {})
    )


def dump_config(path: str | Path, model: ModelConfig, train: TrainConfig) -> None:
    payload = {"model": asdict(model), "train": asdict(train)}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8")
