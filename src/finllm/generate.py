"""Text generation CLI for a trained FinLLM checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch

from finllm.checkpoint import load_model_checkpoint
from finllm.tokenizer import SentencePieceTokenizer
from finllm.utils import resolve_device, resolve_dtype, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained FinLLM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="finance_tokenizer.model")
    parser.add_argument("--prompt", default="Revenue increased because")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qa-mode", action="store_true", help="Use stricter decoding for Q&A (lower temp, higher penalties)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype("auto", device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # Apply stricter decoding for QA mode
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty
    
    if args.qa_mode:
        temperature = 0.5  # More deterministic
        top_k = 30  # Narrower sampling
        top_p = 0.85  # More focused
        repetition_penalty = 1.15  # Stronger penalty
        print("Using QA mode: stricter decoding for factual answers")

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    model = load_model_checkpoint(args.checkpoint, map_location=device).to(device)
    model.eval()

    ids = tokenizer.encode(args.prompt, add_bos=True)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad(), ctx:
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_id if tokenizer.eos_id >= 0 else None,
        )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
