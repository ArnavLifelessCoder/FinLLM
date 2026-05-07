"""Training entrypoint for FinLLM."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext

import torch

from finllm.checkpoint import save_checkpoint
from finllm.config import TrainConfig, dump_config, load_config
from finllm.data import load_splits
from finllm.model import FinanceGPT
from finllm.tokenizer import SentencePieceTokenizer
from finllm.utils import (
    cosine_lr,
    count_parameters,
    ensure_dir,
    format_count,
    resolve_device,
    resolve_dtype,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch financial GPT model.")
    parser.add_argument("--config", default="configs/finance_small.json")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--init-from")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    *,
    model: FinanceGPT,
    train_data,
    val_data,
    train_config: TrainConfig,
    device: str,
    ctx,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, dataset in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(train_config.eval_iters)
        for index in range(train_config.eval_iters):
            x, y = dataset.get_batch(train_config.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[index] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def main() -> None:
    args = parse_args()
    model_config, train_config = load_config(args.config)
    if args.max_iters is not None:
        train_config.max_iters = args.max_iters
        train_config.lr_decay_iters = min(train_config.lr_decay_iters, args.max_iters)
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.device is not None:
        train_config.device = args.device
    if args.compile:
        train_config.compile = True
    if args.resume_from is not None:
        train_config.resume_from = args.resume_from
    if args.init_from is not None:
        train_config.init_from = args.init_from

    set_seed(train_config.seed)
    if train_config.cpu_threads is not None and train_config.cpu_threads > 0:
        torch.set_num_threads(train_config.cpu_threads)
    device = resolve_device(train_config.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = resolve_dtype(train_config.dtype, device)
    ctx = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = SentencePieceTokenizer(train_config.tokenizer_model)
    model_config.vocab_size = tokenizer.vocab_size
    train_data, val_data = load_splits(train_config.data_dir, model_config.block_size)
    out_dir = ensure_dir(train_config.out_dir)
    dump_config(out_dir / "resolved_config.json", model_config, train_config)

    model = FinanceGPT(model_config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        device_type=device_type,
    )
    iter_num = 0
    best_val_loss = float("inf")

    if train_config.init_from:
        checkpoint = torch.load(train_config.init_from, map_location=device)
        init_model_config = checkpoint.get("model_config", {})
        if init_model_config:
            loaded_arch = init_model_config.get("architecture", "legacy")
            if loaded_arch != model_config.architecture:
                raise ValueError(
                    "init_from checkpoint architecture does not match current config: "
                    f"{loaded_arch} vs {model_config.architecture}"
                )
        model.load_state_dict(checkpoint["model"], strict=True)

    if train_config.resume_from:
        checkpoint = torch.load(train_config.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        iter_num = int(checkpoint.get("iter_num", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    if train_config.compile:
        model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler(enabled=device_type == "cuda" and dtype == torch.float16)

    print(
        json.dumps(
            {
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "architecture": model_config.architecture,
                "parameters": format_count(count_parameters(model)),
                "train_tokens": len(train_data),
                "val_tokens": len(val_data),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )

    x, y = train_data.get_batch(train_config.batch_size, device)
    t0 = time.time()
    while iter_num <= train_config.max_iters:
        lr = cosine_lr(
            iter_num,
            learning_rate=train_config.learning_rate,
            min_lr=train_config.min_lr,
            warmup_iters=train_config.warmup_iters,
            lr_decay_iters=train_config.lr_decay_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % train_config.eval_interval == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                train_config=train_config,
                device=device,
                ctx=ctx,
            )
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_checkpoint(
                    out_dir / "best.pt",
                    model=raw_model,
                    optimizer=optimizer,
                    model_config=model_config,
                    train_config=train_config,
                    iteration=iter_num,
                    best_val_loss=best_val_loss,
                )

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(train_config.gradient_accumulation_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / train_config.gradient_accumulation_steps
            x, y = train_data.get_batch(train_config.batch_size, device)
            scaler.scale(loss).backward()
            total_loss += float(loss.item())

        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if iter_num % train_config.log_interval == 0:
            dt = time.time() - t0
            print(
                f"step {iter_num}: loss {total_loss:.4f}, lr {lr:.2e}, "
                f"{dt * 1000:.0f} ms"
            )
            t0 = time.time()
        iter_num += 1

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_checkpoint(
        out_dir / "last.pt",
        model=raw_model,
        optimizer=optimizer,
        model_config=model_config,
        train_config=train_config,
        iteration=iter_num,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()
