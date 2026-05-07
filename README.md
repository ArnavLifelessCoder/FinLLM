# FinLLM Chatbot Studio 

A from-scratch financial-domain decoder-only Transformer language model and Retrieval-Augmented Generation (RAG) chatbot.

FinLLM is designed as an educational and portfolio demonstration of LLM systems engineering, including model training, custom tokenization, full-text retrieval, and a modern web UI.

---

##  Key Features

*   **Custom Financial Domain Model:** A GPT-style causal language model trained from scratch on financial phrase data, news, and SEC 10-K filings.
*   **Dual Backend Support:** Switch seamlessly between the local custom PyTorch model and an Ollama-powered backend (e.g., `llama3.2:3b`).
*   **Retrieval-Augmented Generation (RAG):** Grounded Q&A backed by a local SQLite FTS5 search index to reduce hallucinations.
*   **Conversation Memory:** Maintains up to 10 turns of conversation context for a natural chat experience.
*   **Modern Web UI:** A redesigned interface with gradient design, real-time confidence indicators, interactive evidence viewer, and responsive parameter controls.
*   **Production Ready:** Includes one-command deployment scripts and Docker containerization support.

---

##  Architecture

The system is built with a component-based architecture:

*   **Client Layer:** Vanilla JavaScript and HTML/CSS web UI with a conversation container and interactive backend toggles.
*   **Server Layer:** A lightweight Python HTTP server (`ThreadingHTTPServer`) with RESTful API endpoints for chat, memory management, and backend switching.
*   **Backend Layer:**
    *   **Custom Assistant:** A custom-trained `FinLLM` PyTorch checkpoint combined with a retrieval system.
    *   **Ollama Backend:** Integration with an external Ollama instance for larger parameter models.
*   **Retrieval Layer:** SQLite FTS5 for fast full-text search with BM25 relevance ranking.

*(See [3_ARCHITECTURE_AND_FLOW.md](3_ARCHITECTURE_AND_FLOW.md) for detailed architecture diagrams and flow descriptions.)*

---

##  Getting Started

Follow these instructions to set up, train, and run the FinLLM Chatbot Studio locally.

### 1. Installation

Clone the repository and install the project in editable mode:

```bash
python -m pip install -e .
```
*(If you encounter module not found errors for `finllm`, run: `export PYTHONPATH="src"` on Linux/Mac or `$env:PYTHONPATH = "src"` on Windows PowerShell).*

### 2. Prepare Base Training Data

Prepare the dataset using the provided text corpus and tokenizer:

```bash
python scripts/prepare_dataset.py --corpus financial_training_corpus_clean.txt --tokenizer finance_tokenizer.model --output-dir data/finance --val-fraction 0.005
```

### 3. Train the Base Model

Train a base model on the prepared dataset. For a CPU-friendly modern model:

```bash
python -m finllm.train --config configs/finance_modern_cpu.json
```
*(To use an NVIDIA GPU, add `--device cuda --compile` to the command).*

### 4. Build the Retrieval Index

Build the SQLite FTS5 index to power the grounded Q&A and reduce hallucinations:

```bash
python scripts/build_retrieval_index.py --corpus financial_training_corpus_clean.txt --index data/retrieval/finance_fts.sqlite --max-chunks 20000
```

### 5. Build and Train on Instruction Data (Recommended)

Create a stronger QA dataset with reasoning examples and fine-tune:

```bash
# Build instruction data
python scripts/build_instruction_corpus.py --corpus financial_training_corpus_clean.txt --output-dir data/instruction --max-examples 50000

# Tokenize instruction data
python scripts/prepare_dataset.py --corpus data/instruction/finance_sft.txt --tokenizer finance_tokenizer.model --output-dir data/finance_sft --val-fraction 0.02

# Fine-tune the model
python -m finllm.train --config configs/finance_modern_sft_cpu.json
```

### 6. Run the Web Interface

Start the local server to interact with your trained model:

```bash
python scripts/run_webapp.py --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

*(See [run.md](run.md) for full detailed instructions including deployment options).*

---

## CLI Usage

You can also run generation directly from the command line:

**For factual Q&A with stricter decoding:**
```bash
python -m finllm.generate --checkpoint runs/finance-modern-sft-cpu/best.pt --tokenizer finance_tokenizer.model --prompt "What is operating cash flow?" --max-new-tokens 120 --qa-mode
```

**For creative continuation:**
```bash
python -m finllm.generate --checkpoint runs/finance-modern-sft-cpu/best.pt --tokenizer finance_tokenizer.model --prompt "Apple reported stronger operating cash flow because" --max-new-tokens 120
```

---

## Documentation

For deeper dives into the project's methodology, architecture, and model specifics, refer to the following documentation:

*   [**QUICK_REFERENCE.md**](QUICK_REFERENCE.md) - Quick start and feature overview.
*   [**MODEL_CARD.md**](MODEL_CARD.md) - Model architecture, evaluation, risks, and intended use.
*   [**2_METHODOLOGY_AND_FLOW.md**](2_METHODOLOGY_AND_FLOW.md) - Iterative development approach and lessons learned.
*   [**3_ARCHITECTURE_AND_FLOW.md**](3_ARCHITECTURE_AND_FLOW.md) - Detailed system architecture, component breakdowns, and data flow.

---

## Risks and Limitations

*   **Not Financial Advice:** This model is for educational and demonstration purposes only. Do not use for production investment workflows or as a safety-critical decision system.
*   **Hallucinations:** The model can hallucinate companies, numbers, dates, and causal claims. Always check factual or financial claims against primary sources.
*   **Stale Information:** Training data (SEC filings and news) may contain historical or stale information.
*   Generated text should be clearly labeled as synthetic.
