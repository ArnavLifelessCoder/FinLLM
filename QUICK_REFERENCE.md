# ⚡ Quick Reference - FinLLM Chatbot

## Start Server
```bash
python scripts/run_webapp.py
```

## Open Browser
```
http://127.0.0.1:8000
```

## Documentation

| File | Purpose | Size |
|------|---------|------|
| **1_INTERVIEW_GUIDE.md** | What, Why, Results | 14 KB |
| **2_METHODOLOGY_AND_FLOW.md** | How We Built It | 17 KB |
| **3_ARCHITECTURE_AND_FLOW.md** | System Architecture | 30 KB |
| **README_CHATBOT_DOCS.md** | Documentation Guide | 3 KB |

## Key Features

✅ Conversation Memory (10 turns)  
✅ Longer Responses (200-400 tokens)  
✅ Chat UI (message bubbles)  
✅ Dual Backend (Ollama/Custom)  
✅ Retrieval-Augmented Generation  

## Switch Backend

**In UI**: Click toggle in sidebar  
**In Code**: Edit `webapp/server.py` line 58  
**Via API**: `POST /api/set-backend {"backend": "ollama"}`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama timeout | Increased to 120s, auto-fallback |
| Connection errors | Handled gracefully, ignore |
| Short responses | Use Ollama backend (toggle in UI) |
| No memory | Don't refresh page, memory is per-session |

## Files Modified

1. `src/finllm/ollama_backend.py` - Memory + timeout
2. `src/finllm/assistant.py` - Memory tracking
3. `webapp/server.py` - Endpoints + fallbacks
4. `webapp/static/app.js` - Chat UI logic
5. `webapp/static/index.html` - Conversation display
6. `webapp/static/styles.css` - Message styles

## Improvements

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Response (Ollama) | 150 | 400 | +167% |
| Response (Custom) | 96 | 200 | +108% |
| Memory | 0 | 10 turns | NEW |
| Evidence | 3 | 4 | +33% |

## For Interviews

**Prep Time**: 1 hour  
**Read**: 1_INTERVIEW_GUIDE.md  
**Skim**: 2_METHODOLOGY_AND_FLOW.md  
**Review**: 3_ARCHITECTURE_AND_FLOW.md diagrams  

## Status

✅ Production Ready  
✅ All Bugs Fixed  
✅ Fully Documented  
✅ Ready to Demo  
