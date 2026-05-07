# Model Card: FinLLM

## Overview

FinLLM is a GPT-style causal language model trained from scratch on a finance-heavy
text corpus. The corpus in this workspace includes financial phrase data, financial
news, and SEC 10-K style filings. The tokenizer is a 32k SentencePiece BPE model
stored at `finance_tokenizer.model`.

## Intended use

- Educational and portfolio demonstration of LLM systems engineering.
- Financial-domain language modeling experiments.
- Text continuation and qualitative analysis of domain-specific generation.

## Out of scope

- Financial advice.
- Production investment workflows.
- Factual question answering without retrieval and verification.
- Use as a safety-critical decision system.

## Architecture

- Decoder-only Transformer.
- Legacy GPT-style path for older checkpoints.
- Modern compact LLM path with RMSNorm, RoPE, grouped-query attention, and SwiGLU.
- Token embeddings and optional learned positional embeddings.
- Tied input embedding and output projection weights.
- Cross-entropy next-token objective.

Default `finance_small` config:

- Architecture: modern compact LLM.
- Context length: 256 tokens.
- Layers: 6.
- Attention heads: 6.
- KV heads: 2.
- Embedding width: 384.
- Dropout: 0.1.
- Vocabulary: loaded from `finance_tokenizer.model`.

## Data pipeline

The corpus is streamed line by line through SentencePiece and written into compact
binary token arrays:

- `train.bin`
- `val.bin`
- `manifest.json`

This design avoids loading the full 1.5 GB text corpus into memory during training.

## Evaluation

Primary metrics:

- validation cross-entropy loss
- validation perplexity
- qualitative generated samples
- retrieval hit quality for finance questions
- citation coverage for grounded answers

Recommended reporting:

```text
model_config:
hardware:
train_tokens:
val_tokens:
iterations:
effective_batch_tokens:
best_val_loss:
best_val_perplexity:
sample_prompt:
sample_output:
known_failure_cases:
```

## Risks and limitations

- The model can hallucinate companies, numbers, dates, and causal claims.
- SEC filings and news can contain historical or stale information.
- Training data may contain boilerplate, duplicate text, and formatting artifacts.
- The compact context length limits long-form reasoning.
- Grounded chat reduces hallucination by retrieving evidence, but it is still not
  a substitute for primary-source verification.

## Responsible use

Generated text should be clearly labeled as synthetic. Any factual or financial
claim should be checked against primary sources before use.
