// DOM Elements
const statusEl = document.querySelector("#service-status");
const statusTextEl = statusEl?.querySelector(".status-text");
const metricsEl = document.querySelector("#metrics");
const outputEl = document.querySelector("#output");
const chatOutputEl = document.querySelector("#chat-output");
const confidenceBadgeEl = document.querySelector("#confidence-badge");
const evidenceSectionEl = document.querySelector("#evidence-section");
const evidenceListEl = document.querySelector("#evidence-list");
const modelConfigEl = document.querySelector("#model-config");
const commandsEl = document.querySelector("#commands");
const checkpointsEl = document.querySelector("#checkpoints");
const retrievalEl = document.querySelector("#retrieval");
const costEl = document.querySelector("#cost");
const promptEl = document.querySelector("#prompt");
const chatQuestionEl = document.querySelector("#chat-question");

// Conversation history
let conversationHistory = [];

// Slider value displays
const maxTokensSlider = document.querySelector("#max-new-tokens");
const temperatureSlider = document.querySelector("#temperature");
const topKSlider = document.querySelector("#top-k");
const topPSlider = document.querySelector("#top-p");

const maxTokensValue = document.querySelector("#max-tokens-value");
const temperatureValue = document.querySelector("#temperature-value");
const topKValue = document.querySelector("#top-k-value");
const topPValue = document.querySelector("#top-p-value");

// Update slider displays
if (maxTokensSlider) {
  maxTokensSlider.addEventListener("input", (e) => {
    maxTokensValue.textContent = e.target.value;
  });
}

if (temperatureSlider) {
  temperatureSlider.addEventListener("input", (e) => {
    temperatureValue.textContent = parseFloat(e.target.value).toFixed(2);
  });
}

if (topKSlider) {
  topKSlider.addEventListener("input", (e) => {
    topKValue.textContent = e.target.value;
  });
}

if (topPSlider) {
  topPSlider.addEventListener("input", (e) => {
    topPValue.textContent = parseFloat(e.target.value).toFixed(2);
  });
}

// Status management
function setStatus(text, state = "neutral") {
  if (statusTextEl) statusTextEl.textContent = text;
  if (statusEl) statusEl.dataset.state = state;
}

// Utility functions
function formatMb(value) {
  if (value === null || value === undefined) return "Missing";
  if (value >= 1024) return `${(value / 1024).toFixed(2)} GB`;
  return `${value.toFixed(2)} MB`;
}

function createMetricCard(label, value, type = "neutral") {
  const card = document.createElement("div");
  card.className = `metric-card ${type}`;
  card.innerHTML = `
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
  `;
  return card;
}

function renderProject(project) {
  const hasCheckpoint = project.checkpoints.some((checkpoint) => checkpoint.exists);
  
  if (metricsEl) {
    metricsEl.innerHTML = "";
    metricsEl.appendChild(
      createMetricCard("Tokenizer", `${project.tokenizer.vocab_size.toLocaleString()} vocab`, "success")
    );
    metricsEl.appendChild(
      createMetricCard("Corpus", formatMb(project.corpus.clean_size_mb), "neutral")
    );
    metricsEl.appendChild(
      createMetricCard(
        "Retrieval",
        project.retrieval.exists ? `${project.retrieval.chunks.toLocaleString()} chunks` : "Not indexed",
        project.retrieval.exists ? "success" : "danger"
      )
    );
    metricsEl.appendChild(
      createMetricCard("Checkpoint", hasCheckpoint ? "Ready" : "Missing", hasCheckpoint ? "success" : "warning")
    );
    metricsEl.appendChild(
      createMetricCard("Cost", project.cost.paid_apis ? "Paid API" : "Local/Free", project.cost.paid_apis ? "danger" : "success")
    );
    
    // Add Ollama status if available
    if (project.ollama && project.ollama.available) {
      metricsEl.appendChild(
        createMetricCard(
          "Ollama",
          project.ollama.enabled ? "Enabled" : "Available",
          project.ollama.enabled ? "success" : "neutral"
        )
      );
    }
  }

  // Show/hide model toggle based on Ollama availability
  const toggleSection = document.querySelector("#model-toggle-section");
  if (toggleSection && project.ollama) {
    if (project.ollama.available) {
      toggleSection.style.display = "block";
      // Update active button
      document.querySelectorAll(".toggle-btn").forEach(btn => {
        const backend = btn.dataset.backend;
        if (backend === project.backend) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });
    } else {
      toggleSection.style.display = "none";
    }
  }

  if (modelConfigEl) modelConfigEl.textContent = JSON.stringify(project.model, null, 2);
  if (commandsEl) {
    commandsEl.textContent = [
      "# Prepare dataset",
      project.commands.prepare,
      "",
      "# Build retrieval index",
      project.commands.index,
      "",
      "# Train base model (CPU)",
      project.commands.train_cpu,
      "",
      "# Build instruction data",
      project.commands.sft_data,
      "",
      "# Fine-tune on instructions",
      project.commands.sft_train,
      "",
      "# Generate samples",
      project.commands.generate
    ].join("\n");
  }
  if (retrievalEl) retrievalEl.textContent = JSON.stringify(project.retrieval, null, 2);
  if (costEl) costEl.textContent = JSON.stringify(project.cost, null, 2);
  if (checkpointsEl) checkpointsEl.textContent = JSON.stringify(project.checkpoints, null, 2);
  
  setStatus(hasCheckpoint ? "Ready" : "Setup", hasCheckpoint ? "ok" : "warn");
}

// API helper
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const payload = await response.json();
  if (!response.ok && !payload.error) {
    throw new Error(`Request failed with ${response.status}`);
  }
  return { response, payload };
}

// Load project info
async function loadProject() {
  try {
    setStatus("Loading", "neutral");
    const { payload } = await api("/api/project");
    renderProject(payload);
  } catch (error) {
    setStatus("Offline", "bad");
    if (chatOutputEl) chatOutputEl.textContent = error.message;
  }
}

// Tokenize prompt
async function tokenizePrompt() {
  if (!outputEl || !promptEl) return;
  outputEl.textContent = "Tokenizing...";
  try {
    const { payload } = await api("/api/tokenize", {
      method: "POST",
      body: JSON.stringify({ text: promptEl.value })
    });
    outputEl.textContent = JSON.stringify(
      {
        token_count: payload.count,
        pieces: payload.pieces,
        ids: payload.ids
      },
      null,
      2
    );
  } catch (error) {
    outputEl.textContent = `Error: ${error.message}`;
  }
}

// Generate text
async function generateText() {
  if (!outputEl || !promptEl) return;
  outputEl.textContent = "Generating...";
  outputEl.parentElement.classList.add("loading");
  
  try {
    const { payload } = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        prompt: promptEl.value,
        max_new_tokens: parseInt(maxTokensSlider?.value || 96),
        temperature: parseFloat(temperatureSlider?.value || 0.7),
        top_k: parseInt(topKSlider?.value || 40),
        top_p: parseFloat(topPSlider?.value || 0.9)
      })
    });
    
    if (payload.error) {
      outputEl.textContent = `Error: ${payload.error}`;
    } else {
      outputEl.textContent = payload.text || payload.generated_text || "No output";
    }
  } catch (error) {
    outputEl.textContent = `Error: ${error.message}`;
  } finally {
    outputEl.parentElement.classList.remove("loading");
  }
}

// Ask question
async function askQuestion() {
  if (!chatOutputEl || !chatQuestionEl) return;
  
  const question = chatQuestionEl.value.trim();
  if (!question) return;
  
  // Add user message to conversation display
  addMessageToConversation("user", question);
  
  chatOutputEl.textContent = "Thinking...";
  chatOutputEl.parentElement.classList.add("loading");
  if (confidenceBadgeEl) confidenceBadgeEl.textContent = "Processing";
  if (evidenceSectionEl) evidenceSectionEl.style.display = "none";
  
  // Clear input
  chatQuestionEl.value = "";
  
  try {
    const { payload } = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ question: question })
    });
    
    if (payload.error) {
      chatOutputEl.textContent = `Error: ${payload.error}`;
      if (confidenceBadgeEl) {
        confidenceBadgeEl.textContent = "Error";
        confidenceBadgeEl.className = "confidence-badge";
      }
    } else {
      const answer = payload.answer || "No answer available";
      chatOutputEl.textContent = answer;
      
      // Add assistant message to conversation display
      addMessageToConversation("assistant", answer);
      
      // Update confidence badge
      if (confidenceBadgeEl) {
        const confidence = payload.confidence || "none";
        confidenceBadgeEl.textContent = confidence.toUpperCase();
        confidenceBadgeEl.className = `confidence-badge ${confidence}`;
      }
      
      // Show evidence if available
      if (payload.evidence && payload.evidence.length > 0 && evidenceListEl) {
        renderEvidence(payload.evidence);
        if (evidenceSectionEl) evidenceSectionEl.style.display = "block";
      }
      
      // Update conversation history
      if (payload.conversation_history) {
        conversationHistory = payload.conversation_history;
      }
    }
  } catch (error) {
    chatOutputEl.textContent = `Error: ${error.message}`;
    if (confidenceBadgeEl) {
      confidenceBadgeEl.textContent = "Error";
      confidenceBadgeEl.className = "confidence-badge";
    }
  } finally {
    chatOutputEl.parentElement.classList.remove("loading");
  }
}

// Add message to conversation display
function addMessageToConversation(role, content) {
  const conversationContainer = document.querySelector("#conversation-container");
  if (!conversationContainer) return;
  
  const messageDiv = document.createElement("div");
  messageDiv.className = `conversation-message ${role}-message fade-in`;
  
  const roleLabel = document.createElement("div");
  roleLabel.className = "message-role";
  roleLabel.textContent = role === "user" ? "You" : "Assistant";
  
  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = content;
  
  messageDiv.appendChild(roleLabel);
  messageDiv.appendChild(contentDiv);
  conversationContainer.appendChild(messageDiv);
  
  // Scroll to bottom
  conversationContainer.scrollTop = conversationContainer.scrollHeight;
}

// Clear conversation
async function clearConversation() {
  try {
    await api("/api/clear-memory", {
      method: "POST",
      body: JSON.stringify({})
    });
    
    conversationHistory = [];
    const conversationContainer = document.querySelector("#conversation-container");
    if (conversationContainer) {
      conversationContainer.innerHTML = "";
    }
    
    if (chatOutputEl) {
      chatOutputEl.textContent = "Conversation cleared. Ask me anything about finance!";
    }
  } catch (error) {
    if (chatOutputEl) {
      chatOutputEl.textContent = `Error clearing conversation: ${error.message}`;
    }
  }
}

// Retrieve evidence
async function retrieveEvidence() {
  if (!chatQuestionEl || !evidenceListEl) return;
  if (evidenceSectionEl) evidenceSectionEl.style.display = "none";
  
  try {
    const { payload } = await api("/api/retrieve", {
      method: "POST",
      body: JSON.stringify({ question: chatQuestionEl.value })
    });
    
    if (payload.error) {
      if (chatOutputEl) chatOutputEl.textContent = `Error: ${payload.error}`;
    } else if (payload.results && payload.results.length > 0) {
      renderEvidence(payload.results);
      if (evidenceSectionEl) evidenceSectionEl.style.display = "block";
      if (chatOutputEl) chatOutputEl.textContent = `Retrieved ${payload.results.length} evidence chunks.`;
    } else {
      if (chatOutputEl) chatOutputEl.textContent = "No evidence found for this query.";
    }
  } catch (error) {
    if (chatOutputEl) chatOutputEl.textContent = `Error: ${error.message}`;
  }
}

// Render evidence
function renderEvidence(evidence) {
  if (!evidenceListEl) return;
  evidenceListEl.innerHTML = "";
  
  evidence.forEach((item) => {
    const evidenceItem = document.createElement("div");
    evidenceItem.className = "evidence-item fade-in";
    evidenceItem.innerHTML = `
      <div class="evidence-header">
        <span class="evidence-rank">Rank ${item.rank}</span>
        <span class="evidence-score">Score: ${item.score?.toFixed(3) || "N/A"}</span>
      </div>
      <div class="evidence-text">${escapeHtml(item.text || "")}</div>
      <div class="evidence-source">${escapeHtml(item.source || "")}</div>
    `;
    evidenceListEl.appendChild(evidenceItem);
  });
}

// Escape HTML
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Tab switching
document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    const targetTab = button.dataset.tab;
    
    // Update nav items
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.classList.remove("active");
    });
    button.classList.add("active");
    
    // Update panels
    document.querySelectorAll(".panel").forEach((panel) => {
      panel.classList.remove("active");
    });
    const targetPanel = document.querySelector(`#${targetTab}`);
    if (targetPanel) targetPanel.classList.add("active");
  });
});

// Event listeners
document.querySelector("#refresh-project")?.addEventListener("click", loadProject);
document.querySelector("#tokenize")?.addEventListener("click", tokenizePrompt);
document.querySelector("#generate")?.addEventListener("click", generateText);
document.querySelector("#ask")?.addEventListener("click", askQuestion);
document.querySelector("#retrieve")?.addEventListener("click", retrieveEvidence);
document.querySelector("#clear-conversation")?.addEventListener("click", clearConversation);

// Model toggle
document.querySelectorAll(".toggle-btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    const backend = btn.dataset.backend;
    try {
      const { payload } = await api("/api/set-backend", {
        method: "POST",
        body: JSON.stringify({ backend })
      });
      
      // Update UI
      document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      
      // Show notification
      if (outputEl) {
        outputEl.textContent = `Switched to ${backend} backend. ${payload.message || ""}`;
      }
      
      // Reload project info
      loadProject();
    } catch (error) {
      if (outputEl) outputEl.textContent = `Error switching backend: ${error.message}`;
    }
  });
});

// Enter key support
promptEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    generateText();
  }
});

chatQuestionEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    askQuestion();
  }
});

// Initialize
loadProject();
