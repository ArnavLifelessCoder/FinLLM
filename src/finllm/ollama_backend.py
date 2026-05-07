"""Ollama backend for production-quality generation.

This module provides an interface to Ollama models while maintaining
compatibility with the custom FinLLM interface.
"""

from __future__ import annotations

import json
from typing import Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class OllamaBackend:
    """Ollama model backend for production use with conversation memory."""
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """Initialize Ollama backend.
        
        Args:
            model: Ollama model name (e.g., "llama3.2:3b", "mistral")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library required for Ollama backend. "
                "Install with: pip install requests"
            )
        
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.conversation_history: list[dict[str, str]] = []
        self._check_connection()
    
    def _check_connection(self) -> None:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if self.model not in model_names:
                print(f"Warning: Model '{self.model}' not found in Ollama.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"Run: ollama pull {self.model}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Ollama at {self.base_url}")
            print(f"Error: {e}")
            print("Make sure Ollama is running: https://ollama.com")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        system_prompt: str | None = None
    ) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop: Stop sequences
            system_prompt: Optional system prompt for instruction
            
        Returns:
            Generated text
        """
        # Build the prompt with system message if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            return f"Error: Could not generate with Ollama. {e}"
    
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Chat completion using Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            return f"Error: Could not chat with Ollama. {e}"
    
    def generate_financial_qa(
        self,
        question: str,
        context: str | None = None,
        max_tokens: int = 400,
        use_memory: bool = True
    ) -> str:
        """Generate financial Q&A response with conversation memory.
        
        Args:
            question: Financial question
            context: Optional context/evidence from retrieval
            max_tokens: Maximum tokens to generate
            use_memory: Whether to use conversation history
            
        Returns:
            Answer to the question
        """
        system_prompt = (
            "You are an expert financial analysis assistant with deep knowledge of finance, "
            "accounting, and business metrics. Provide comprehensive, detailed answers that:\n"
            "- Explain concepts thoroughly with examples when helpful\n"
            "- Use the provided context/evidence when available\n"
            "- Reference previous conversation context when relevant\n"
            "- Give practical insights and implications\n"
            "- Use clear structure with multiple paragraphs for complex topics\n"
            "- Aim for 3-5 sentences minimum, more for complex questions\n\n"
            "Be professional, accurate, and educational in your responses."
        )
        
        # Build the user message
        if context:
            user_message = (
                f"Based on the following financial documents and data:\n\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                f"Please provide a detailed, comprehensive answer."
            )
        else:
            user_message = f"Question: {question}\n\nPlease provide a detailed, comprehensive answer."
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        if use_memory and self.conversation_history:
            # Include recent conversation history (last 6 messages = 3 turns)
            messages.extend(self.conversation_history[-6:])
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.6,  # Balanced for detailed but focused answers
            top_p=0.9
        )
        
        # Store in conversation history
        if use_memory:
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep only last 20 messages (10 turns) to manage memory
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def clear_memory(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get current conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()
    
    def set_conversation_history(self, history: list[dict[str, str]]) -> None:
        """Set conversation history.
        
        Args:
            history: List of message dictionaries
        """
        self.conversation_history = history.copy()


def check_ollama_available() -> bool:
    """Check if Ollama is available and running.
    
    Returns:
        True if Ollama is available, False otherwise
    """
    if not REQUESTS_AVAILABLE:
        return False
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> list[str]:
    """Get list of available Ollama models.
    
    Returns:
        List of model names
    """
    if not REQUESTS_AVAILABLE:
        return []
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except:
        return []
