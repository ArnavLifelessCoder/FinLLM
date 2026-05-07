#!/usr/bin/env python3
"""Quick test script for the chatbot functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ollama():
    """Test Ollama backend."""
    print("=" * 60)
    print("Testing Ollama Backend")
    print("=" * 60)
    
    try:
        from finllm.ollama_backend import check_ollama_available, OllamaBackend
        
        available = check_ollama_available()
        print(f"✓ Ollama available: {available}")
        
        if available:
            backend = OllamaBackend()
            print(f"✓ Ollama backend initialized")
            print(f"✓ Model: {backend.model}")
            
            # Test generation
            print("\nTesting generation...")
            result = backend.generate("What is 2+2?", max_tokens=50)
            print(f"✓ Generation works: {result[:100]}...")
            
            # Test memory
            print("\nTesting conversation memory...")
            backend.generate_financial_qa("What is EBITDA?", max_tokens=100)
            history = backend.get_conversation_history()
            print(f"✓ Memory works: {len(history)} messages stored")
            
            backend.clear_memory()
            history = backend.get_conversation_history()
            print(f"✓ Clear memory works: {len(history)} messages after clear")
            
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_assistant():
    """Test custom assistant."""
    print("\n" + "=" * 60)
    print("Testing Custom Assistant")
    print("=" * 60)
    
    try:
        from finllm.assistant import HybridFinanceAssistant
        
        index_path = Path("data/retrieval/finance_fts.sqlite")
        if not index_path.exists():
            print(f"✗ Index not found: {index_path}")
            return False
        
        print(f"✓ Index found: {index_path}")
        
        assistant = HybridFinanceAssistant(index_path)
        print(f"✓ Assistant initialized")
        
        # Test answer
        print("\nTesting answer...")
        result = assistant.answer("What is revenue?", use_memory=True)
        print(f"✓ Answer works: {result['mode']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Answer length: {len(result['answer'])} chars")
        
        # Test memory
        history = assistant.get_conversation_history()
        print(f"✓ Memory works: {len(history)} messages stored")
        
        assistant.clear_memory()
        history = assistant.get_conversation_history()
        print(f"✓ Clear memory works: {len(history)} messages after clear")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval():
    """Test retrieval system."""
    print("\n" + "=" * 60)
    print("Testing Retrieval System")
    print("=" * 60)
    
    try:
        from finllm.retrieval import search, index_stats
        
        index_path = Path("data/retrieval/finance_fts.sqlite")
        if not index_path.exists():
            print(f"✗ Index not found: {index_path}")
            return False
        
        print(f"✓ Index found: {index_path}")
        
        # Test stats
        stats = index_stats(index_path)
        print(f"✓ Index stats:")
        print(f"  Exists: {stats['exists']}")
        print(f"  Chunks: {stats['chunks']}")
        
        # Test search
        results = search(index_path, "revenue growth", limit=3)
        print(f"✓ Search works: {len(results)} results")
        if results:
            print(f"  First result: {results[0].text[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FINLLM CHATBOT TEST SUITE")
    print("=" * 60 + "\n")
    
    results = {
        "Ollama Backend": test_ollama(),
        "Retrieval System": test_retrieval(),
        "Custom Assistant": test_assistant(),
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYour chatbot is ready to use!")
        print("Run: python scripts/run_webapp.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nCheck the errors above and fix them.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
