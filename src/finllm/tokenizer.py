"""SentencePiece tokenizer wrapper used by training and generation."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class SentencePieceTokenizer:
    """Small typed wrapper around a SentencePiece model."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {self.model_path}")

        self._sp = spm.SentencePieceProcessor()
        loaded = self._sp.load(str(self.model_path))
        if not loaded:
            raise RuntimeError(f"SentencePiece failed to load {self.model_path}")

    @property
    def vocab_size(self) -> int:
        return int(self._sp.get_piece_size())

    @property
    def bos_id(self) -> int:
        return int(self._sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self._sp.eos_id())

    @property
    def pad_id(self) -> int:
        return int(self._sp.pad_id())

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self._sp.encode(text, out_type=int))
        if add_bos and self.bos_id >= 0:
            ids.insert(0, self.bos_id)
        if add_eos and self.eos_id >= 0:
            ids.append(self.eos_id)
        return ids

    def pieces(self, text: str) -> list[str]:
        return list(self._sp.encode(text, out_type=str))

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return str(self._sp.decode(list(ids)))
