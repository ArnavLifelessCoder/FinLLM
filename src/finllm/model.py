"""Decoder-only Transformer language models for FinLLM.

This module keeps a legacy GPT-style path for old checkpoints and adds a more
modern compact LLM path with RMSNorm, RoPE, GQA, and SwiGLU.
"""

from __future__ import annotations

import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F

from finllm.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def build_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type.lower() == "layernorm":
        return nn.LayerNorm(config.n_embd, eps=config.norm_eps, bias=config.bias)
    if config.norm_type.lower() == "rmsnorm":
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    raise ValueError(f"Unsupported norm_type: {config.norm_type}")


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, kv_heads, time, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, kv_heads, n_rep, time, head_dim)
        .reshape(batch, kv_heads * n_rep, time, head_dim)
    )


class LegacyCausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")

        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head

        q = q.view(batch, time, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch, time, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch, time, self.n_head, head_size).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :time, :time] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(y))


class ModernCausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.num_kv_heads = config.num_kv_heads or config.n_head
        if self.num_kv_heads <= 0 or self.n_head % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must be a positive divisor of n_head")

        self.dropout = config.dropout
        self.kv_repeat = self.n_head // self.num_kv_heads
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(
            config.n_embd, self.num_kv_heads * self.head_dim, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.num_kv_heads * self.head_dim, bias=config.bias
        )
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base) if config.use_rope else None

        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.size()
        q = self.q_proj(x).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope.cos_sin(time, x.device, x.dtype)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :time, :time] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.out_proj(y))


class LegacyMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class ModernMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = max(4, int(config.mlp_hidden_mult * config.n_embd))
        self.mlp_type = config.mlp_type.lower()
        if self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        elif self.mlp_type == "gelu":
            self.fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported mlp_type: {config.mlp_type}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "swiglu":
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            x = self.act(self.fc(x))
        return self.dropout(self.down_proj(x))


class LegacyBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = LegacyCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = LegacyMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ModernBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_1 = build_norm(config)
        self.attn = ModernCausalSelfAttention(config)
        self.norm_2 = build_norm(config)
        self.mlp = ModernMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class FinanceGPT(nn.Module):
    """Decoder-only LM with legacy GPT and modern compact LLM modes."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.architecture = config.architecture.lower()
        if self.architecture not in {"legacy", "modern"}:
            raise ValueError("architecture must be 'legacy' or 'modern'")

        if self.architecture == "legacy":
            self.transformer = nn.ModuleDict(
                {
                    "wte": nn.Embedding(config.vocab_size, config.n_embd),
                    "wpe": nn.Embedding(config.block_size, config.n_embd),
                    "drop": nn.Dropout(config.dropout),
                    "h": nn.ModuleList([LegacyBlock(config) for _ in range(config.n_layer)]),
                    "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
                }
            )
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight
        else:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
            self.pos_embeddings = (
                None
                if config.use_rope
                else nn.Embedding(config.block_size, config.n_embd)
            )
            self.drop = nn.Dropout(config.dropout)
            self.layers = nn.ModuleList([ModernBlock(config) for _ in range(config.n_layer)])
            self.norm = build_norm(config)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.tok_embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith(("c_proj.weight", "out_proj.weight", "down_proj.weight")):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_legacy(
        self, idx: torch.Tensor, targets: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = idx.size()
        positions = torch.arange(0, time, dtype=torch.long, device=idx.device)
        token_embeddings = self.transformer["wte"](idx)
        position_embeddings = self.transformer["wpe"](positions)
        x = self.transformer["drop"](token_embeddings + position_embeddings)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _forward_modern(
        self, idx: torch.Tensor, targets: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = idx.size()
        x = self.tok_embeddings(idx)
        if self.pos_embeddings is not None:
            positions = torch.arange(0, time, dtype=torch.long, device=idx.device)
            x = x + self.pos_embeddings(positions)
        x = self.drop(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = idx.size()
        if time > self.config.block_size:
            raise ValueError(
                f"Sequence length {time} exceeds block size {self.config.block_size}"
            )
        if self.architecture == "legacy":
            return self._forward_legacy(idx, targets)
        return self._forward_modern(idx, targets)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if repetition_penalty > 1.0:
                for batch_idx in range(logits.size(0)):
                    seen = torch.unique(idx[batch_idx])
                    logits[batch_idx, seen] = logits[batch_idx, seen] / repetition_penalty

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(1, sorted_indices, sorted_mask)
                logits = logits.masked_fill(mask, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and torch.all(idx_next == eos_token_id):
                break
        return idx

    def crop_block_size(self, block_size: int) -> None:
        if block_size > self.config.block_size:
            raise ValueError("Cannot increase block size after initialization")
        self.config.block_size = block_size
        if self.architecture == "legacy":
            self.transformer["wpe"].weight = nn.Parameter(
                self.transformer["wpe"].weight[:block_size]
            )
            for block in self.transformer["h"]:
                if hasattr(block.attn, "bias"):
                    block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
        else:
            if self.pos_embeddings is not None:
                self.pos_embeddings.weight = nn.Parameter(self.pos_embeddings.weight[:block_size])
            for block in self.layers:
                if hasattr(block.attn, "bias"):
                    block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(
        self,
        *,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        decay_params = [param for param in params.values() if param.dim() >= 2]
        nodecay_params = [param for param in params.values() if param.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        signature = inspect.signature(torch.optim.AdamW).parameters
        fused_available = "fused" in signature
        foreach_available = "foreach" in signature
        kwargs = {}
        if fused_available and device_type == "cuda":
            kwargs["fused"] = True
        if foreach_available and device_type == "cpu":
            kwargs["foreach"] = True
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **kwargs,
        )
