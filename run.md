Run it in this order from PowerShell.

**1. Go to project**
```powershell
cd "C:\Users\Arnav Gawade(pro)\Downloads\llm"
```

**2. Optional: stop current website server**
If the website is already running:

```powershell
Stop-Process -Id 37708
```

If that PID is old, find the current one:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_webapp.py*' } | Select-Object ProcessId,CommandLine
```

Then:

```powershell
Stop-Process -Id PROCESS_ID
```

**3. Install project**
Do once:

```powershell
python -m pip install -e .
```

If `python -m finllm.train` ever says module not found, run:

```powershell
$env:PYTHONPATH = "src"
```

**4. Prepare base LLM training data**
You already seem to have done this, but this is the command:

```powershell
python scripts/prepare_dataset.py --corpus financial_training_corpus_clean.txt --tokenizer finance_tokenizer.model --output-dir data/finance --val-fraction 0.005
```

**5. Train base model**

For CPU-friendly modern model (RECOMMENDED):

```powershell
python -m finllm.train --config configs/finance_modern_cpu.json
```

For tiny quick test:

```powershell
python -m finllm.train --config configs/finance_tiny_cpu.json
```

For main better model, but slow on CPU:

```powershell
python -m finllm.train --config configs/finance_small.json
```

If you have NVIDIA GPU:

```powershell
python -m finllm.train --config configs/finance_small.json --device cuda --compile
```

This creates checkpoints like:

```text
runs/finance-modern-cpu/best.pt
runs/finance-modern-cpu/last.pt
```

**6. Build retrieval index**
This powers the grounded Q&A/chat and reduces hallucination.

Fast starter index:

```powershell
python scripts/build_retrieval_index.py --corpus financial_training_corpus_clean.txt --index data/retrieval/finance_fts.sqlite --max-chunks 20000
```

Full index, better but takes longer:

```powershell
python scripts/build_retrieval_index.py --corpus financial_training_corpus_clean.txt --index data/retrieval/finance_fts.sqlite
```

**7. Build improved instruction data (NEW)**
This creates a stronger QA dataset with reasoning examples:

```powershell
python scripts/build_instruction_corpus.py --corpus financial_training_corpus_clean.txt --output-dir data/instruction --max-examples 50000
```

Tokenize instruction data:

```powershell
python scripts/prepare_dataset.py --corpus data/instruction/finance_sft.txt --tokenizer finance_tokenizer.model --output-dir data/finance_sft --val-fraction 0.02
```

**8. Fine-tune on instructions (RECOMMENDED)**
Train the instruction-following model:

```powershell
python -m finllm.train --config configs/finance_modern_sft_cpu.json
```

This creates:
```text
runs/finance-modern-sft-cpu/best.pt
runs/finance-modern-sft-cpu/last.pt
```

**9. Run the modern web interface (NEW)**
```powershell
python scripts/run_webapp.py --host 127.0.0.1 --port 8000
```

Open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

**10. Use the redesigned website**
The new interface features:

- **Grounded Chat**: Evidence-backed Q&A with confidence scores
- **Generate**: Base model sampling with interactive controls
- **Operations**: System metrics and training commands

**11. Test QA mode generation (NEW)**
For factual Q&A with stricter decoding:

```powershell
python -m finllm.generate --checkpoint runs/finance-modern-sft-cpu/best.pt --tokenizer finance_tokenizer.model --prompt "What is operating cash flow?" --max-new-tokens 120 --qa-mode
```

For creative continuation:

```powershell
python -m finllm.generate --checkpoint runs/finance-modern-sft-cpu/best.pt --tokenizer finance_tokenizer.model --prompt "Apple reported stronger operating cash flow because" --max-new-tokens 120
```

**12. Deploy as production app (NEW)**
Package for deployment:

```powershell
python scripts/deploy_app.py --output-dir dist
```

Run from deployment bundle:

```powershell
cd dist
./run.sh  # Unix/Linux/Mac
run.bat   # Windows
```

Or deploy with Docker:

```powershell
cd dist
docker build -t finllm-studio .
docker run -p 8000:8000 finllm-studio
```

**Most practical CPU path (UPDATED)**
Since your CPU training is slow, do this complete workflow:

```powershell
cd "C:\Users\Arnav Gawade(pro)\Downloads\llm"
python -m pip install -e .

# Train base model
python -m finllm.train --config configs/finance_modern_cpu.json

# Build retrieval index
python scripts/build_retrieval_index.py --corpus financial_training_corpus_clean.txt --index data/retrieval/finance_fts.sqlite --max-chunks 20000

# Build improved instruction data
python scripts/build_instruction_corpus.py --corpus financial_training_corpus_clean.txt --output-dir data/instruction --max-examples 50000

# Tokenize instruction data
python scripts/prepare_dataset.py --corpus data/instruction/finance_sft.txt --tokenizer finance_tokenizer.model --output-dir data/finance_sft --val-fraction 0.02

# Fine-tune on instructions
python -m finllm.train --config configs/finance_modern_sft_cpu.json

# Run modern web interface
python scripts/run_webapp.py --host 127.0.0.1 --port 8000
```

Then open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

## What's New

### Enhanced Reasoning
- 15 curated QA examples with detailed explanations
- Improved instruction dataset biased toward definitions
- Stricter decoding for factual answers (--qa-mode flag)
- Better confidence scoring (high/medium/low)

### Modern UI
- Completely redesigned interface with gradient design
- Real-time confidence indicators
- Interactive evidence viewer with ranked results
- Responsive parameter controls
- Professional metrics dashboard

### Production Ready
- One-command deployment script
- Docker containerization support
- Complete deployment documentation
- Security best practices included