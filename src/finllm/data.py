"""Streaming corpus preparation and efficient binary token datasets."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from finllm.tokenizer import SentencePieceTokenizer


@dataclass(slots=True)
class DatasetManifest:
    source_path: str
    tokenizer_model: str
    vocab_size: int
    dtype: str
    train_tokens: int
    val_tokens: int
    val_fraction: float
    seed: int
    source_sha256: str


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_binary_dataset(
    *,
    corpus_path: str | Path,
    tokenizer_model: str | Path,
    output_dir: str | Path,
    val_fraction: float = 0.005,
    seed: int = 1337,
    min_chars: int = 1,
) -> DatasetManifest:
    """Tokenize a text corpus into train.bin and val.bin without loading it all at once."""

    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    tokenizer = SentencePieceTokenizer(tokenizer_model)
    dtype = token_dtype(tokenizer.vocab_size)
    rng = random.Random(seed)
    train_tokens = 0
    val_tokens = 0

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    with corpus_path.open("r", encoding="utf8", errors="ignore") as source, train_path.open(
        "wb"
    ) as train_file, val_path.open("wb") as val_file:
        for line in tqdm(source, desc="tokenizing corpus", unit="lines"):
            text = line.strip()
            if len(text) < min_chars:
                continue
            ids = tokenizer.encode(text, add_eos=True)
            if not ids:
                continue
            arr = np.asarray(ids, dtype=dtype)
            if rng.random() < val_fraction:
                arr.tofile(val_file)
                val_tokens += int(arr.size)
            else:
                arr.tofile(train_file)
                train_tokens += int(arr.size)

    manifest = DatasetManifest(
        source_path=str(corpus_path),
        tokenizer_model=str(tokenizer_model),
        vocab_size=tokenizer.vocab_size,
        dtype=dtype.name,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        val_fraction=val_fraction,
        seed=seed,
        source_sha256=sha256_file(corpus_path),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True),
        encoding="utf8",
    )
    return manifest


class BinaryTokenDataset:
    """Memory-mapped token stream for random next-token batches."""

    def __init__(self, path: str | Path, *, block_size: int, dtype: str | np.dtype):
        self.path = Path(path)
        self.block_size = block_size
        self.dtype = np.dtype(dtype)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.path}")
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(
                f"{self.path} has {len(self.data)} tokens, need more than {block_size + 1}"
            )

    def __len__(self) -> int:
        return int(len(self.data))

    def close(self) -> None:
        mmap = getattr(self.data, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self) -> "BinaryTokenDataset":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.random.randint(0, len(self.data) - self.block_size - 1, size=batch_size)
        offsets = np.arange(self.block_size)
        positions = starts[:, None] + offsets[None, :]
        x = torch.from_numpy(np.asarray(self.data[positions], dtype=np.int64))
        y = torch.from_numpy(np.asarray(self.data[positions + 1], dtype=np.int64))
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


def load_manifest(data_dir: str | Path) -> dict:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run scripts/prepare_dataset.py before training."
        )
    return json.loads(manifest_path.read_text(encoding="utf8"))


def load_splits(data_dir: str | Path, block_size: int) -> tuple[BinaryTokenDataset, BinaryTokenDataset]:
    manifest = load_manifest(data_dir)
    dtype = manifest["dtype"]
    data_dir = Path(data_dir)
    return (
        BinaryTokenDataset(data_dir / "train.bin", block_size=block_size, dtype=dtype),
        BinaryTokenDataset(data_dir / "val.bin", block_size=block_size, dtype=dtype),
    )
