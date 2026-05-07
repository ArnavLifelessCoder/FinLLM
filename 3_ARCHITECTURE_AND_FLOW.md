# 🏗️ Architecture and Flow - FinLLM Chatbot

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Browser (HTML + CSS + JavaScript)                        │  │
│  │  - Message input                                          │  │
│  │  - Conversation display (message bubbles)                 │  │
│  │  - Backend toggle (Ollama/Custom)                         │  │
│  │  - Clear history button                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/JSON (RESTful API)
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                      SERVER LAYER                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Python HTTP Server (ThreadingHTTPServer)                 │  │
│  │  - Request routing                                        │  │
│  │  - Session management                                     │  │
│  │  - Backend selection (_USE_OLLAMA flag)                   │  │
│  │  - Error handling & fallbacks                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────┴──────────┐      ┌──────────┴─────────┐
│   OLLAMA BACKEND   │      │  CUSTOM ASSISTANT  │
│                    │      │                    │
│ ┌────────────────┐ │      │ ┌────────────────┐ │
│ │ Memory Manager │ │      │ │ Memory Manager │ │
│ │ - Store msgs   │ │      │ │ - Store msgs   │ │
│ │ - Trim to 20   │ │      │ │ - Trim to 20   │ │
│ │ - Get history  │ │      │ │ - Get history  │ │
│ └────────────────┘ │      │ └────────────────┘ │
│                    │      │                    │
│ ┌────────────────┐ │      │ ┌────────────────┐ │
│ │   Generator    │ │      │ │   Generator    │ │
│ │ - 400 tokens   │ │      │ │ - 200 tokens   │ │
│ │ - temp: 0.6    │ │      │ │ - temp: 0.3    │ │
│ │ - top_p: 0.9   │ │      │ │ - top_p: 0.85  │ │
│ └────────────────┘ │      │ └────────────────┘ │
│                    │      │                    │
│ ┌────────────────┐ │      │ ┌────────────────┐ │
│ │ Ollama API     │ │      │ │ PyTorch Model  │ │
│ │ - llama3.2:3b  │ │      │ │ - Custom FinLLM│ │
│ │ - localhost    │ │      │ │ - Checkpoint   │ │
│ └────────────────┘ │      │ └────────────────┘ │
└────────────────────┘      └──────────┬─────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                ┌────────┴────────┐      ┌──────────┴─────────┐
                │ RETRIEVAL LAYER │      │  KNOWLEDGE LAYER   │
                │                 │      │                    │
                │ ┌─────────────┐ │      │ ┌────────────────┐ │
                │ │ SQLite FTS5 │ │      │ │ Local Defs     │ │
                │ │ - Full-text │ │      │ │ - EBITDA       │ │
                │ │ - Ranking   │ │      │ │ - Revenue      │ │
                │ │ - Top-K     │ │      │ │ - etc.         │ │
                │ └─────────────┘ │      │ └────────────────┘ │
                │                 │      │                    │
                │ ┌─────────────┐ │      └────────────────────┘
                │ │ Evidence    │ │
                │ │ - 4 sources │ │
                │ │ - 600 chars │ │
                │ └─────────────┘ │
                └─────────────────┘
```

---

## Component Details

### 1. Client Layer

#### HTML Structure
```html
<div class="app-container">
    <!-- Sidebar -->
    <aside class="sidebar">
        <div class="logo">FinLLM Studio</div>
        <div class="metrics">Status, Ollama, etc.</div>
        <div class="model-toggle">
            <button data-backend="custom">Custom Model</button>
            <button data-backend="ollama">Ollama</button>
        </div>
    </aside>
    
    <!-- Main Content -->
    <main class="main-content">
        <!-- Conversation Display -->
        <div class="conversation-display">
            <div class="conversation-header">
                <h3>Conversation</h3>
                <button id="clear-conversation">Clear History</button>
            </div>
            <div class="conversation-container">
                <!-- Messages appear here dynamically -->
            </div>
        </div>
        
        <!-- Input -->
        <textarea id="chat-question"></textarea>
        <button id="ask">Ask FinLLM</button>
        
        <!-- Latest Answer -->
        <div id="chat-output"></div>
        
        <!-- Evidence -->
        <div id="evidence-section"></div>
    </main>
</div>
```

#### JavaScript Architecture
```javascript
// State Management
let conversationHistory = [];

// API Communication
async function api(path, options) {
    const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
    });
    return await response.json();
}

// Core Functions
async function askQuestion() {
    // 1. Get question
    const question = chatQuestionEl.value.trim();
    
    // 2. Display user message
    addMessageToConversation("user", question);
    
    // 3. Call API
    const response = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ question })
    });
    
    // 4. Display assistant message
    addMessageToConversation("assistant", response.answer);
    
    // 5. Update history
    conversationHistory = response.conversation_history;
}

// UI Updates
function addMessageToConversation(role, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `conversation-message ${role}-message`;
    messageDiv.innerHTML = `
        <div class="message-role">${role === "user" ? "You" : "Assistant"}</div>
        <div class="message-content">${content}</div>
    `;
    conversationContainer.appendChild(messageDiv);
    conversationContainer.scrollTop = conversationContainer.scrollHeight;
}
```

**Why This Design**:
- ✅ Vanilla JS = No build step, fast loading
- ✅ Component-based = Easy to maintain
- ✅ State management = Tracks conversation
- ✅ Async/await = Clean async code

---

### 2. Server Layer

#### Request Flow
```python
class FinLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. Parse request
        parsed = urlparse(self.path)
        payload = self.read_json()
        
        # 2. Route to handler
        if parsed.path == "/api/chat":
            self.handle_chat(payload)
        elif parsed.path == "/api/clear-memory":
            self.handle_clear_memory()
        elif parsed.path == "/api/set-backend":
            self.handle_set_backend(payload)
    
    def handle_chat(self, payload):
        question = payload.get("question", "")
        
        # 3. Try Ollama first
        if _USE_OLLAMA:
            try:
                answer = self.generate_with_ollama(question)
                return self.send_json(HTTPStatus.OK, answer)
            except Exception as e:
                print(f"Ollama failed: {e}, falling back")
        
        # 4. Fallback to custom
        try:
            answer = self.generate_with_custom(question)
            return self.send_json(HTTPStatus.OK, answer)
        except Exception as e:
            return self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": str(e)
            })
```

**Why This Design**:
- ✅ Try-catch with fallback = Robust
- ✅ Clear routing = Easy to extend
- ✅ Error handling = Graceful degradation
- ✅ Standard library = No dependencies

---

### 3. Ollama Backend

#### Architecture
```python
class OllamaBackend:
    def __init__(self, model="llama3.2:3b", timeout=120):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.timeout = timeout
        self.conversation_history = []  # Memory
    
    def generate_financial_qa(
        self,
        question: str,
        context: str | None = None,
        max_tokens: int = 400,
        use_memory: bool = True
    ) -> str:
        # 1. Build system prompt
        system_prompt = """
        You are an expert financial analysis assistant.
        Provide comprehensive, detailed answers (3-6 sentences minimum).
        Use provided context when available.
        Reference previous conversation when relevant.
        """
        
        # 2. Build messages with history
        messages = [{"role": "system", "content": system_prompt}]
        
        if use_memory and self.conversation_history:
            # Include last 6 messages (3 turns)
            messages.extend(self.conversation_history[-6:])
        
        # 3. Add current question with context
        user_message = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        messages.append({"role": "user", "content": user_message})
        
        # 4. Generate response
        response = self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9
        )
        
        # 5. Store in memory
        if use_memory:
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Trim to last 20 messages
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def chat(self, messages, max_tokens, temperature, top_p):
        # Call Ollama API
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        
        return response.json()["message"]["content"]
```

**Why This Design**:
- ✅ Stateful = Maintains conversation
- ✅ Configurable = Easy to tune
- ✅ Memory management = Automatic trimming
- ✅ Context-aware = Uses history

---

### 4. Custom Assistant

#### Architecture
```python
class HybridFinanceAssistant:
    def __init__(self, index_path):
        self.index_path = Path(index_path)
        self.conversation_history = []  # Memory
    
    def answer(
        self,
        question: str,
        top_k: int = 6,
        refiner: Callable | None = None,
        use_memory: bool = True
    ) -> dict:
        # 1. Store question in memory
        if use_memory:
            self.conversation_history.append({
                "role": "user",
                "content": question,
                "timestamp": None
            })
        
        # 2. Check local knowledge first
        knowledge = lookup_finance_definition(question)
        if knowledge:
            answer = knowledge["answer"]
            # ... return with evidence
        
        # 3. Retrieve from corpus
        results = self.retrieve(question, top_k=top_k)
        if not results:
            # ... return "no evidence" message
        
        # 4. Extract evidence
        terms = query_terms(question, limit=10)
        answer, confidence = extractive_summary(results, terms, question)
        evidence = evidence_to_dict(results)
        
        # 5. Refine with model (if available)
        if refiner and confidence != "low":
            refined = refiner(question, evidence, answer)
            if refined and looks_like_usable_answer(refined):
                answer = refined
        
        # 6. Store answer in memory
        if use_memory:
            self.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "timestamp": None
            })
            
            # Trim to last 20 messages
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
        
        # 7. Return with history
        return {
            "mode": "grounded_retrieval",
            "answer": answer,
            "confidence": confidence,
            "evidence": evidence,
            "conversation_history": self.conversation_history
        }
```

**Why This Design**:
- ✅ Hybrid = Local knowledge + retrieval
- ✅ Grounded = Evidence-based answers
- ✅ Memory = Conversation tracking
- ✅ Refinement = Optional model enhancement

---

### 5. Retrieval Layer

#### Architecture
```python
# SQLite FTS5 Full-Text Search
def search(index_path: Path, query: str, limit: int = 6) -> list[SearchResult]:
    # 1. Connect to SQLite
    conn = sqlite3.connect(index_path)
    cursor = conn.cursor()
    
    # 2. Full-text search with ranking
    cursor.execute("""
        SELECT rowid, source, chunk_index, text, 
               bm25(corpus_fts) as score
        FROM corpus_fts
        WHERE corpus_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """, (query, limit))
    
    # 3. Build results
    results = []
    for rank, row in enumerate(cursor.fetchall(), 1):
        results.append(SearchResult(
            rank=rank,
            rowid=row[0],
            source=row[1],
            chunk_index=row[2],
            text=row[3],
            score=row[4]
        ))
    
    return results
```

**Why This Design**:
- ✅ FTS5 = Fast full-text search
- ✅ BM25 = Relevance ranking
- ✅ SQLite = No separate server
- ✅ Portable = Single file

---

## Data Flow Diagrams

### Flow 1: User Asks Question

```
┌──────┐
│ User │ Types: "What is EBITDA?"
└───┬──┘
    │
    ↓
┌───────────────┐
│   Frontend    │ 1. Capture input
│               │ 2. Display user message (blue bubble)
│               │ 3. POST /api/chat {"question": "..."}
└───────┬───────┘
        │
        ↓
┌───────────────┐
│    Server     │ 4. Receive request
│               │ 5. Check _USE_OLLAMA flag
│               │ 6. Route to Ollama backend
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ Ollama Backend│ 7. Get conversation history (last 6 msgs)
│               │ 8. Retrieve evidence (4 sources, 600 chars)
│               │ 9. Build context:
│               │    - System prompt
│               │    - History (last 6)
│               │    - Evidence
│               │    - Question
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ Retrieval     │ 10. FTS5 search on "EBITDA"
│               │ 11. Rank by BM25 score
│               │ 12. Return top 4 results
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ Ollama API    │ 13. Generate response (400 tokens)
│               │ 14. Temperature: 0.6
│               │ 15. Top-p: 0.9
│               │ 16. Takes ~5-10 seconds
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ Ollama Backend│ 17. Store question in history
│               │ 18. Store answer in history
│               │ 19. Trim to last 20 messages
│               │ 20. Return answer + history
└───────┬───────┘
        │
        ↓
┌───────────────┐
│    Server     │ 21. Receive response
│               │ 22. Add session_id
│               │ 23. Return JSON
└───────┬───────┘
        │
        ↓
┌───────────────┐
│   Frontend    │ 24. Receive response
│               │ 25. Display assistant message (green bubble)
│               │ 26. Update conversation history
│               │ 27. Auto-scroll to bottom
└───────┬───────┘
        │
        ↓
┌──────┐
│ User │ Sees detailed answer with context
└──────┘
```

**Timing**:
- Steps 1-6: <100ms (frontend + routing)
- Steps 7-12: ~500ms (retrieval)
- Steps 13-16: ~5-10s (generation)
- Steps 17-27: <200ms (storage + display)
- **Total**: ~6-11 seconds

---

### Flow 2: Backend Switching

```
┌──────┐
│ User │ Clicks "Ollama" button
└───┬──┘
    │
    ↓
┌───────────────┐
│   Frontend    │ 1. Capture click
│               │ 2. POST /api/set-backend {"backend": "ollama"}
└───────┬───────┘
        │
        ↓
┌───────────────┐
│    Server     │ 3. Receive request
│               │ 4. Validate backend name
│               │ 5. Set _USE_OLLAMA = True
│               │ 6. Return {"backend": "ollama", "message": "Switched"}
└───────┬───────┘
        │
        ↓
┌───────────────┐
│   Frontend    │ 7. Receive response
│               │ 8. Update UI (highlight button)
│               │ 9. Show notification
│               │ 10. Reload project info
└───────┬───────┘
        │
        ↓
┌──────┐
│ User │ Sees "Switched to ollama backend"
└──────┘
```

**Timing**: <100ms (instant)

---

### Flow 3: Clear Conversation

```
┌──────┐
│ User │ Clicks "Clear History" button
└───┬──┘
    │
    ↓
┌───────────────┐
│   Frontend    │ 1. Capture click
│               │ 2. POST /api/clear-memory {}
└───────┬───────┘
        │
        ↓
┌───────────────┐
│    Server     │ 3. Receive request
│               │ 4. Check _USE_OLLAMA flag
│               │ 5. Call backend.clear_memory()
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ Ollama Backend│ 6. Clear conversation_history = []
│               │ 7. Return success
└───────┬───────┘
        │
        ↓
┌───────────────┐
│    Server     │ 8. Return {"message": "Cleared"}
└───────┬───────┘
        │
        ↓
┌───────────────┐
│   Frontend    │ 9. Receive response
│               │ 10. Clear conversationHistory = []
│               │ 11. Clear conversation-container HTML
│               │ 12. Show success message
└───────┬───────┘
        │
        ↓
┌──────┐
│ User │ Sees empty conversation, fresh start
└──────┘
```

**Timing**: <50ms (instant)

---

## Memory Management Architecture

### Memory Structure
```python
conversation_history = [
    {"role": "user", "content": "What is EBITDA?"},
    {"role": "assistant", "content": "EBITDA stands for..."},
    {"role": "user", "content": "How is it calculated?"},
    {"role": "assistant", "content": "Building on what we discussed..."},
    # ... up to 20 messages total
]
```

### Memory Lifecycle
```
New Message
    ↓
Append to conversation_history[]
    ↓
Check: len(conversation_history) > 20?
    ↓ Yes
Trim: conversation_history = conversation_history[-20:]
    ↓ No
Continue
    ↓
When generating:
    ↓
Extract last 6 messages
    ↓
Include in context window
    ↓
Generate response
    ↓
Append response to history
    ↓
Repeat
```

### Memory Limits
- **Storage**: Last 20 messages (10 turns)
- **Context**: Last 6 messages (3 turns)
- **Per Message**: ~100-500 chars
- **Total Memory**: ~10-50 KB per session

**Why These Limits**:
- ✅ 20 messages = Enough for meaningful conversation
- ✅ 6 in context = Fits in context window
- ✅ Auto-trim = Prevents memory bloat
- ✅ Per-session = Isolated conversations

---

## Error Handling Architecture

### Error Hierarchy
```
┌─────────────────────────────────────┐
│         Connection Errors           │
│  (ConnectionAbortedError, etc.)     │
│  → Catch and ignore (client gone)   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│         Ollama Errors               │
│  (Timeout, connection refused)      │
│  → Fallback to custom assistant     │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      Custom Assistant Errors        │
│  (Missing checkpoint, etc.)         │
│  → Return error message to user     │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│         Retrieval Errors            │
│  (Missing index, etc.)              │
│  → Return "no evidence" message     │
└─────────────────────────────────────┘
```

### Fallback Chain
```
User Question
    ↓
Try: Ollama Backend
    ↓ Fails
Try: Custom Assistant
    ↓ Fails
Return: Error Message
    ↓
User sees: "Error: Could not generate answer. Please try again."
```

**Why This Design**:
- ✅ Multiple fallbacks = High availability
- ✅ Graceful degradation = Always responds
- ✅ Clear errors = Easy to debug
- ✅ User-friendly = No technical jargon

---

## Performance Characteristics

### Response Times
| Operation | Time | Notes |
|-----------|------|-------|
| Frontend render | <50ms | Instant |
| API routing | <10ms | Very fast |
| Retrieval search | 200-500ms | SQLite FTS5 |
| Ollama generation | 5-10s | 400 tokens |
| Custom generation | 2-3s | 200 tokens |
| Memory operations | <1ms | In-memory |
| **Total (Ollama)** | **6-11s** | Acceptable |
| **Total (Custom)** | **3-4s** | Fast |

### Memory Usage
| Component | Memory | Notes |
|-----------|--------|-------|
| Per session | 10-50 KB | 20 messages |
| 10 sessions | 100-500 KB | Acceptable |
| 100 sessions | 1-5 MB | Still fine |
| Ollama model | ~2 GB | External process |
| Custom model | ~500 MB | Loaded on demand |

### Scalability
| Users | Performance | Bottleneck |
|-------|-------------|------------|
| 1-5 | Excellent | None |
| 5-10 | Good | Ollama generation |
| 10-20 | Acceptable | CPU/Memory |
| 20+ | Needs optimization | Threading |

**Scaling Strategies**:
1. Add Redis for session storage
2. Use FastAPI with async
3. Load balance across servers
4. Cache common responses
5. Use GPU for generation

---

## Security Considerations

### Current Security
- ✅ Runs locally (no external API calls)
- ✅ No user authentication (single-user)
- ✅ No data persistence (memory only)
- ✅ Input validation (basic)
- ❌ No rate limiting
- ❌ No CSRF protection
- ❌ No XSS protection

### Production Security Needs
1. **Authentication**: User login system
2. **Authorization**: Role-based access
3. **Rate Limiting**: Prevent abuse
4. **Input Sanitization**: Prevent injection
5. **HTTPS**: Encrypt traffic
6. **CSRF Tokens**: Prevent cross-site attacks
7. **XSS Protection**: Sanitize output

---

## Deployment Architecture

### Current (Development)
```
Single Server
    ↓
Python HTTP Server (port 8000)
    ↓
Ollama (port 11434)
    ↓
SQLite (file)
```

### Production (Recommended)
```
Load Balancer (NGINX)
    ↓
┌─────────┬─────────┬─────────┐
│ Server 1│ Server 2│ Server 3│
└─────────┴─────────┴─────────┘
    ↓           ↓           ↓
┌─────────────────────────────┐
│      Redis (Sessions)        │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   PostgreSQL (Persistence)   │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   Ollama Cluster (GPU)       │
└─────────────────────────────┘
```

---

## Conclusion

This architecture provides:
- ✅ **Modularity**: Clear separation of concerns
- ✅ **Scalability**: Can scale to 100+ users with modifications
- ✅ **Reliability**: Multiple fallbacks and error handling
- ✅ **Performance**: 3-11s response time acceptable for chatbot
- ✅ **Maintainability**: Well-documented and organized
- ✅ **Extensibility**: Easy to add new features

**Production-Ready**: With minor modifications (FastAPI, Redis, PostgreSQL), this architecture can handle production workloads.
