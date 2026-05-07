"""Small full-stack web server for FinLLM Studio.

The server intentionally uses the Python standard library so the demo can run
without adding a web framework dependency.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from finllm.assistant import HybridFinanceAssistant
from finllm.checkpoint import load_model_checkpoint
from finllm.config import load_config
from finllm.retrieval import index_stats, search
from finllm.tokenizer import SentencePieceTokenizer
from finllm.utils import resolve_device, set_seed

# Try to import Ollama backend
try:
    from finllm.ollama_backend import OllamaBackend, check_ollama_available
    OLLAMA_AVAILABLE = check_ollama_available()
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaBackend = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_TOKENIZER = PROJECT_ROOT / "finance_tokenizer.model"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "finance_modern_cpu.json"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "retrieval" / "finance_fts.sqlite"
DEFAULT_CHECKPOINTS = [
    PROJECT_ROOT / "runs" / "finance-modern-sft-cpu" / "best.pt",
    PROJECT_ROOT / "runs" / "finance-modern-sft-cpu" / "last.pt",
    PROJECT_ROOT / "runs" / "finance-modern-cpu" / "best.pt",
    PROJECT_ROOT / "runs" / "finance-modern-cpu" / "last.pt",
    PROJECT_ROOT / "runs" / "finance-small" / "best.pt",
    PROJECT_ROOT / "runs" / "finance-small" / "last.pt",
]

_TOKENIZER: SentencePieceTokenizer | None = None
_MODEL_CACHE: dict[str, object] = {}
_ASSISTANT: HybridFinanceAssistant | None = None
_OLLAMA_BACKEND: OllamaBackend | None = None
_USE_OLLAMA: bool = True  # Default to Ollama (better quality, works out of the box)
_CONVERSATION_SESSIONS: dict[str, dict] = {}  # Store conversation sessions

TAG_RE = re.compile(r"<[^>]{1,240}>")
STYLE_RE = re.compile(r"\b(?:style|class|font|span|div|table|tbody|tr|td|vertical-align|font-size)\b", re.I)
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_RE = re.compile(r"\n{3,}")


def file_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 2)


def load_tokenizer() -> SentencePieceTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = SentencePieceTokenizer(DEFAULT_TOKENIZER)
    return _TOKENIZER


def load_assistant() -> HybridFinanceAssistant:
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = HybridFinanceAssistant(DEFAULT_INDEX)
    return _ASSISTANT


def load_ollama_backend() -> OllamaBackend | None:
    """Load Ollama backend if available."""
    global _OLLAMA_BACKEND
    if not OLLAMA_AVAILABLE or OllamaBackend is None:
        return None
    if _OLLAMA_BACKEND is None:
        try:
            _OLLAMA_BACKEND = OllamaBackend(model="llama3.2:3b")
        except Exception as e:
            print(f"Could not initialize Ollama: {e}")
            return None
    return _OLLAMA_BACKEND


def set_model_mode(use_ollama: bool) -> None:
    """Set whether to use Ollama or custom model."""
    global _USE_OLLAMA
    _USE_OLLAMA = use_ollama and OLLAMA_AVAILABLE


def choose_checkpoint(path: str | None = None) -> Path | None:
    if path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate if candidate.exists() else None
    for candidate in DEFAULT_CHECKPOINTS:
        if candidate.exists():
            return candidate
    return None


def choose_assistant_checkpoint() -> Path | None:
    for candidate in DEFAULT_CHECKPOINTS:
        if candidate.exists() and "modern-sft" in str(candidate).lower():
            return candidate
    return None


def load_cached_model(checkpoint: Path, device: str):
    cache_key = str(checkpoint.resolve())
    mtime = checkpoint.stat().st_mtime
    cached = _MODEL_CACHE.get(cache_key)
    if not cached or cached["mtime"] != mtime or cached["device"] != device:
        model = load_model_checkpoint(checkpoint, map_location=device).to(device)
        model.eval()
        _MODEL_CACHE[cache_key] = {"model": model, "mtime": mtime, "device": device}
    return _MODEL_CACHE[cache_key]["model"]


def sample_new_text(
    *,
    checkpoint: Path,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    tokenizer = load_tokenizer()
    model = load_cached_model(checkpoint, device)
    set_seed(1337)
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_id if tokenizer.eos_id >= 0 else None,
        )
    new_tokens = y[0].tolist()[len(ids) :]
    return tokenizer.decode(new_tokens).strip()


def project_snapshot() -> dict:
    tokenizer = load_tokenizer()
    model_config, train_config = load_config(DEFAULT_CONFIG)
    model_config.vocab_size = tokenizer.vocab_size
    checkpoints = [
        {
            "name": checkpoint.name,
            "path": str(checkpoint.relative_to(PROJECT_ROOT)),
            "exists": checkpoint.exists(),
            "size_mb": file_mb(checkpoint),
        }
        for checkpoint in DEFAULT_CHECKPOINTS
    ]
    corpus = PROJECT_ROOT / "financial_training_corpus_clean.txt"
    raw_corpus = PROJECT_ROOT / "financial_training_corpus.txt"
    data_dir = PROJECT_ROOT / train_config.data_dir
    manifest = data_dir / "manifest.json"
    retrieval = index_stats(DEFAULT_INDEX)
    retrieval["path"] = str(DEFAULT_INDEX.relative_to(PROJECT_ROOT))
    
    # Check Ollama status
    ollama_status = {
        "available": OLLAMA_AVAILABLE,
        "enabled": _USE_OLLAMA,
        "models": []
    }
    if OLLAMA_AVAILABLE:
        try:
            from finllm.ollama_backend import get_available_models
            ollama_status["models"] = get_available_models()
        except:
            pass
    
    return {
        "name": "FinLLM Studio",
        "cost": {
            "paid_apis": False,
            "external_model_apis": _USE_OLLAMA,
            "runs_locally": True,
            "notes": [
                "No OpenAI, Mistral, or Hugging Face hosted inference is called.",
                "The web app, retrieval index, tokenizer, and PyTorch model run from local files.",
                "Ollama support available for production-quality generation." if OLLAMA_AVAILABLE else "Install Ollama for production-quality generation.",
            ],
        },
        "ollama": ollama_status,
        "tokenizer": {
            "path": str(DEFAULT_TOKENIZER.relative_to(PROJECT_ROOT)),
            "vocab_size": tokenizer.vocab_size,
            "size_mb": file_mb(DEFAULT_TOKENIZER),
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "pad_id": tokenizer.pad_id,
        },
        "corpus": {
            "clean_path": str(corpus.relative_to(PROJECT_ROOT)),
            "clean_size_mb": file_mb(corpus),
            "raw_path": str(raw_corpus.relative_to(PROJECT_ROOT)),
            "raw_size_mb": file_mb(raw_corpus),
        },
        "model": {
            "architecture": model_config.architecture,
            "block_size": model_config.block_size,
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "n_embd": model_config.n_embd,
            "dropout": model_config.dropout,
            "norm_type": model_config.norm_type,
            "mlp_type": model_config.mlp_type,
            "use_rope": model_config.use_rope,
            "num_kv_heads": model_config.num_kv_heads or model_config.n_head,
        },
        "training": {
            "data_dir": train_config.data_dir,
            "out_dir": train_config.out_dir,
            "batch_size": train_config.batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "max_iters": train_config.max_iters,
            "manifest_exists": manifest.exists(),
        },
        "assistant": {
            "sft_checkpoint_ready": choose_assistant_checkpoint() is not None,
            "hybrid_modes": ["local_knowledge", "grounded_retrieval", "model_refinement"],
        },
        "retrieval": retrieval,
        "checkpoints": checkpoints,
        "backend": "ollama" if _USE_OLLAMA else "custom",
        "commands": {
            "prepare": (
                "python scripts/prepare_dataset.py --corpus financial_training_corpus_clean.txt "
                "--tokenizer finance_tokenizer.model --output-dir data/finance --val-fraction 0.005"
            ),
            "train": "python -m finllm.train --config configs/finance_modern_cpu.json",
            "train_cpu": "python -m finllm.train --config configs/finance_modern_cpu.json",
            "generate": (
                "python -m finllm.generate --checkpoint runs/finance-modern-cpu/best.pt "
                "--tokenizer finance_tokenizer.model --prompt \"Apple revenue increased because\""
            ),
            "index": (
                "python scripts/build_retrieval_index.py --corpus financial_training_corpus_clean.txt "
                "--index data/retrieval/finance_fts.sqlite"
            ),
            "sft_data": (
                "python scripts/build_instruction_corpus.py --corpus financial_training_corpus_clean.txt "
                "--output-dir data/instruction --max-examples 50000"
            ),
            "sft_train": "python -m finllm.train --config configs/finance_modern_sft_cpu.json",
        },
    }


def safe_static_path(request_path: str) -> Path | None:
    parsed_path = urlparse(request_path).path
    if parsed_path in {"", "/"}:
        relative = "index.html"
    else:
        relative = parsed_path.removeprefix("/")
    candidate = (STATIC_ROOT / relative).resolve()
    try:
        candidate.relative_to(STATIC_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def clean_generated_text(text: str) -> tuple[str, bool]:
    """Remove common SEC/XBRL/HTML fragments from generated samples."""

    original = text
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\\n", "\n")
    text = text.replace(">\n", "\n")
    lines: list[str] = []
    for line in text.splitlines():
        line = SPACE_RE.sub(" ", line).strip()
        if not line:
            continue
        if STYLE_RE.search(line) and any(symbol in line for symbol in ["=", ";", "</", "<"]):
            continue
        if line.count(";") >= 4 and "=" in line:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = BLANK_RE.sub("\n\n", cleaned).strip()
    if not cleaned:
        cleaned = "The sample contained mostly filing markup, so it was filtered out. Try a shorter prompt or lower top-k."
    return cleaned, cleaned != original.strip()


def refine_assistant_answer(question: str, evidence: list[dict], draft_answer: str) -> str | None:
    checkpoint = choose_assistant_checkpoint()
    if checkpoint is None:
        return None

    notes = [f"[{item['rank']}] {item['text']}" for item in evidence[:4]]
    if not notes:
        notes = [draft_answer]

    prompt = (
        "<|user|>\n"
        "Answer the finance question in 3 to 6 detailed sentences. "
        "Provide comprehensive explanation with context and implications. "
        "Use only the notes. If the notes are insufficient, say Insufficient evidence.\n\n"
        f"Question: {question}\n"
        "Notes:\n"
        + "\n".join(f"- {note}" for note in notes)
        + "\n<|assistant|>\nAnswer:"
    )
    text = sample_new_text(
        checkpoint=checkpoint,
        prompt=prompt,
        device="cpu",
        max_new_tokens=200,  # Increased for longer responses
        temperature=0.3,  # Lower for more focused answers
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.15,
    )
    text = clean_generated_text(text)[0]
    if text.lower().startswith("answer:"):
        text = text.split(":", 1)[1].strip()
    return text or None


def generate_text(payload: dict) -> dict:
    # Check if we should use Ollama
    if _USE_OLLAMA:
        ollama = load_ollama_backend()
        if ollama:
            prompt = str(payload.get("prompt", "")).strip() or "Revenue increased because"
            max_new_tokens = min(max(int(payload.get("max_new_tokens", 96)), 1), 256)
            temperature = min(max(float(payload.get("temperature", 0.7)), 0.05), 2.0)
            top_p = min(max(float(payload.get("top_p", 0.9)), 0.1), 1.0)
            
            started = time.perf_counter()
            generated = ollama.generate(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            
            return {
                "prompt": prompt,
                "text": prompt + generated,
                "generated_text": generated,
                "elapsed_ms": elapsed_ms,
                "backend": "ollama",
                "model": ollama.model
            }
    
    # Use custom model
    checkpoint = choose_checkpoint(payload.get("checkpoint"))
    if checkpoint is None:
        return {
            "error": "checkpoint_missing",
            "message": "Train a checkpoint first, then generation will run here.",
            "next_command": "python -m finllm.train --config configs/finance_modern_cpu.json",
        }

    prompt = str(payload.get("prompt", "")).strip() or "Revenue increased because"
    max_new_tokens = min(max(int(payload.get("max_new_tokens", 96)), 1), 256)
    temperature = min(max(float(payload.get("temperature", 0.8)), 0.05), 2.0)
    top_k = min(max(int(payload.get("top_k", 50)), 1), 500)
    top_p = min(max(float(payload.get("top_p", 0.95)), 0.1), 1.0)
    repetition_penalty = min(max(float(payload.get("repetition_penalty", 1.05)), 1.0), 2.0)
    device = resolve_device(str(payload.get("device", "auto")))
    seed = int(payload.get("seed", 1337))

    cache_key = str(checkpoint.resolve())
    mtime = checkpoint.stat().st_mtime
    cached = _MODEL_CACHE.get(cache_key)
    if not cached or cached["mtime"] != mtime or cached["device"] != device:
        model = load_model_checkpoint(checkpoint, map_location=device).to(device)
        model.eval()
        _MODEL_CACHE[cache_key] = {"model": model, "mtime": mtime, "device": device}
    model = _MODEL_CACHE[cache_key]["model"]

    tokenizer = load_tokenizer()
    set_seed(seed)
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    started = time.perf_counter()
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_id if tokenizer.eos_id >= 0 else None,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    raw_text = tokenizer.decode(y[0].tolist())
    cleaned_text, was_cleaned = clean_generated_text(raw_text)
    return {
        "prompt": prompt,
        "text": cleaned_text,
        "raw_text": raw_text,
        "backend": "custom",
        "cleaned": was_cleaned,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "tokens_generated": max_new_tokens,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "elapsed_ms": elapsed_ms,
        "device": device,
    }


class FinLLMHandler(BaseHTTPRequestHandler):
    server_version = "FinLLMStudio/0.1"

    def log_message(self, format: str, *args) -> None:
        try:
            print(f"{self.address_string()} - {format % args}")
        except OSError:
            pass

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        try:
            body = json.dumps(payload, indent=2).encode("utf8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            # Client disconnected, ignore
            pass

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf8")
        return json.loads(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json(HTTPStatus.OK, {"ok": True, "service": "finllm-studio"})
            return
        if parsed.path == "/api/project":
            self.send_json(HTTPStatus.OK, project_snapshot())
            return

        static_path = safe_static_path(self.path)
        if static_path is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        body = static_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/tokenize":
                text = str(payload.get("text", ""))
                tokenizer = load_tokenizer()
                ids = tokenizer.encode(text)
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "text": text,
                        "ids": ids,
                        "pieces": tokenizer.pieces(text),
                        "count": len(ids),
                    },
                )
                return
            if parsed.path == "/api/retrieve":
                query = str(payload.get("query", ""))
                limit = min(max(int(payload.get("limit", 6)), 1), 20)
                results = search(DEFAULT_INDEX, query, limit=limit)
                self.send_json(
                    HTTPStatus.OK,
                    {"query": query, "results": [asdict(result) for result in results]},
                )
                return
            if parsed.path == "/api/chat":
                question = str(payload.get("question", ""))
                top_k = min(max(int(payload.get("top_k", 6)), 1), 12)
                session_id = str(payload.get("session_id", "default"))
                clear_history = bool(payload.get("clear_history", False))
                
                # Clear history if requested
                if clear_history:
                    if _USE_OLLAMA:
                        ollama = load_ollama_backend()
                        if ollama:
                            ollama.clear_memory()
                    else:
                        assistant = load_assistant()
                        assistant.clear_memory()
                    
                    self.send_json(HTTPStatus.OK, {
                        "message": "Conversation history cleared",
                        "session_id": session_id
                    })
                    return
                
                # If Ollama is enabled, use it directly for better answers
                if _USE_OLLAMA:
                    try:
                        ollama = load_ollama_backend()
                        if ollama:
                            # Get retrieval evidence
                            results = search(DEFAULT_INDEX, question, limit=top_k)
                            evidence = [asdict(result) for result in results]
                            
                            # Build richer context from evidence
                            context_parts = []
                            for i, r in enumerate(results[:4]):  # Use top 4 results
                                context_parts.append(f"[Source {i+1}] {r.text[:600]}")
                            context = "\n\n".join(context_parts) if context_parts else None
                            
                            # Use Ollama to answer with context and memory
                            answer_text = ollama.generate_financial_qa(
                                question=question,
                                context=context,
                                max_tokens=400,  # Increased for longer responses
                                use_memory=True
                            )
                            
                            # Get conversation history
                            history = ollama.get_conversation_history()
                            
                            self.send_json(HTTPStatus.OK, {
                                "mode": "ollama_grounded",
                                "answer": answer_text,
                                "confidence": "high" if results else "medium",
                                "evidence": evidence,
                                "backend": "ollama",
                                "conversation_history": history,
                                "session_id": session_id
                            })
                            return
                    except Exception as e:
                        # Ollama failed, fall back to custom assistant
                        print(f"Ollama error: {e}, falling back to custom assistant")
                        pass
                
                # Fall back to custom assistant with memory
                try:
                    assistant = load_assistant()
                    answer = assistant.answer(question, top_k=top_k, use_memory=True)
                    answer["session_id"] = session_id
                    self.send_json(HTTPStatus.OK, answer)
                except Exception as e:
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                        "error": "assistant_error",
                        "message": f"Assistant error: {str(e)}",
                        "backend": "custom"
                    })
                return
            if parsed.path == "/api/generate":
                result = generate_text(payload)
                status = (
                    HTTPStatus.CONFLICT if result.get("error") == "checkpoint_missing" else HTTPStatus.OK
                )
                self.send_json(status, result)
                return
            if parsed.path == "/api/set-backend":
                backend = str(payload.get("backend", "custom"))
                use_ollama = backend == "ollama"
                set_model_mode(use_ollama)
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "backend": "ollama" if _USE_OLLAMA else "custom",
                        "ollama_available": OLLAMA_AVAILABLE,
                        "message": f"Switched to {backend} backend" if OLLAMA_AVAILABLE or not use_ollama else "Ollama not available"
                    }
                )
                return
            if parsed.path == "/api/clear-memory":
                # Clear conversation memory
                if _USE_OLLAMA:
                    ollama = load_ollama_backend()
                    if ollama:
                        ollama.clear_memory()
                else:
                    assistant = load_assistant()
                    assistant.clear_memory()
                
                self.send_json(HTTPStatus.OK, {"message": "Conversation memory cleared"})
                return
            if parsed.path == "/api/conversation-history":
                # Get conversation history
                history = []
                if _USE_OLLAMA:
                    ollama = load_ollama_backend()
                    if ollama:
                        history = ollama.get_conversation_history()
                else:
                    assistant = load_assistant()
                    history = assistant.get_conversation_history()
                
                self.send_json(HTTPStatus.OK, {"history": history})
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": exc.__class__.__name__, "message": str(exc)},
            )


class FinLLMHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False
    allow_reuse_port = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FinLLM Studio full-stack app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        server = FinLLMHTTPServer((args.host, args.port), FinLLMHandler)
    except OSError as exc:
        raise SystemExit(
            f"FinLLM Studio could not start on http://{args.host}:{args.port} ({exc}). "
            "Another server may already be running on that port."
        )
    try:
        print(f"FinLLM Studio running at http://{args.host}:{args.port}")
    except OSError:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FinLLM Studio")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
