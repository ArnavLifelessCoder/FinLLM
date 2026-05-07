"""Hybrid local financial assistant.

The assistant combines:
- a local finance knowledge pack for definition-style questions
- retrieval over the corpus for grounded evidence
- optional model refinement with strict validation
- improved reasoning and explanation capabilities
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from finllm.knowledge import lookup_finance_definition
from finllm.retrieval import SearchResult, query_terms, search


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b")
QUESTION_WORDS = {"How", "What", "When", "Where", "Which", "Why", "Does", "Did", "Can", "Could"}
GARBAGE_RE = re.compile(r"(?:font-size|vertical-align|</|<td|<div|b['\"])", re.I)


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]


def sentence_score(sentence: str, terms: list[str]) -> int:
    lowered = sentence.lower()
    return sum(1 for term in terms if term in lowered)


def evidence_to_dict(results: list[SearchResult]) -> list[dict]:
    return [asdict(result) for result in results]


def named_entities(question: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_RE.finditer(question):
        entity = match.group(0)
        if entity in QUESTION_WORDS or entity in entities:
            continue
        entities.append(entity)
    return entities


def looks_like_usable_answer(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 25:
        return False
    if GARBAGE_RE.search(stripped):
        return False
    alpha = sum(char.isalpha() for char in stripped)
    weird = sum(char in "<>{}|" for char in stripped)
    return alpha >= 20 and weird <= 2


def extractive_summary(results: list[SearchResult], terms: list[str], question: str) -> tuple[str, str]:
    entities = named_entities(question)
    candidate_results = results
    if entities:
        entity_hits = [
            result
            for result in results
            if all(entity.lower() in result.text.lower() for entity in entities)
        ]
        if not entity_hits:
            return (
                "I found related finance snippets, but not enough evidence that directly mentions "
                f"{', '.join(entities)}. I will not infer an answer from unrelated context.",
                "low",
            )
        candidate_results = entity_hits

    selected: list[tuple[int, str, SearchResult]] = []
    for result in candidate_results:
        for sentence in split_sentences(result.text):
            score = sentence_score(sentence, terms)
            if score > 0:
                selected.append((score, sentence, result))
    selected.sort(key=lambda item: item[0], reverse=True)

    required_score = min(2, len(terms)) if terms else 1
    selected = [item for item in selected if item[0] >= required_score]
    if not selected:
        return (
            "I retrieved some nearby text, but the evidence is too weak to answer reliably. "
            "Try a more specific question or rebuild the retrieval index over the full corpus.",
            "low",
        )

    facts: list[str] = []
    used_sources: set[int] = set()
    for _, sentence, result in selected:
        if len(facts) >= 5:
            break
        cleaned = sentence.strip()
        if len(cleaned) > 420:
            cleaned = cleaned[:417].rstrip() + "..."
        facts.append(f"{cleaned} [{result.rank}]")
        used_sources.add(result.rank)

    confidence = "high" if len(used_sources) >= 3 else "medium" if len(used_sources) >= 2 else "low"
    
    # Build a more structured answer
    answer_parts = [
        "Based on the retrieved financial corpus:\n"
    ]
    
    # Add reasoning if we have strong evidence
    if confidence in ["high", "medium"]:
        answer_parts.append("The evidence shows:\n")
    
    answer_parts.append("\n".join(f"- {fact}" for fact in facts))
    answer_parts.append(
        "\n\nNote: These are extracted facts from the indexed corpus. "
        "Verify numeric claims and dates against the original filing or source."
    )
    
    answer = "".join(answer_parts)
    return answer, confidence


class HybridFinanceAssistant:
    """Local assistant with knowledge lookup, retrieval, and optional LM refinement."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.conversation_history: list[dict] = []

    def retrieve(self, question: str, *, top_k: int = 6) -> list[SearchResult]:
        return search(self.index_path, question, limit=top_k)

    def answer(
        self,
        question: str,
        *,
        top_k: int = 6,
        refiner: Callable[[str, list[dict], str], str | None] | None = None,
        use_memory: bool = True,
    ) -> dict:
        question = question.strip()
        if not question:
            return {
                "mode": "assistant",
                "answer": "Ask a finance question and I will answer from local knowledge or retrieved evidence.",
                "confidence": "none",
                "evidence": [],
                "conversation_history": self.conversation_history if use_memory else [],
            }
        
        # Store question in history
        if use_memory:
            self.conversation_history.append({
                "role": "user",
                "content": question,
                "timestamp": None
            })

        knowledge = lookup_finance_definition(question)
        if knowledge is not None:
            answer = knowledge["answer"]
            evidence = [
                {
                    "rank": 1,
                    "rowid": -1,
                    "source": f"local_knowledge:{knowledge['term']}",
                    "chunk_index": 0,
                    "score": 0.0,
                    "text": answer,
                }
            ]
            if refiner is not None:
                refined = refiner(question, evidence, answer)
                if refined and looks_like_usable_answer(refined):
                    answer = refined
            
            # Store answer in history
            if use_memory:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": None
                })
                # Keep last 20 messages
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
            
            return {
                "mode": "local_knowledge",
                "answer": answer,
                "confidence": "high",
                "evidence": evidence,
                "conversation_history": self.conversation_history if use_memory else [],
            }

        results = self.retrieve(question, top_k=top_k)
        if not results:
            answer = (
                "I do not have enough indexed evidence to answer that reliably. "
                "Build the retrieval index or ask about a topic covered by the corpus."
            )
            
            if use_memory:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": None
                })
            
            return {
                "mode": "grounded_retrieval",
                "answer": answer,
                "confidence": "low",
                "evidence": [],
                "conversation_history": self.conversation_history if use_memory else [],
            }

        terms = query_terms(question, limit=10)
        answer, confidence = extractive_summary(results, terms, question)
        evidence = evidence_to_dict(results)

        if refiner is not None and confidence != "low":
            refined = refiner(question, evidence, answer)
            if refined and looks_like_usable_answer(refined):
                answer = refined
                confidence = "high" if confidence == "medium" else confidence
        
        # Store answer in history
        if use_memory:
            self.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "timestamp": None
            })
            # Keep last 20 messages
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

        return {
            "mode": "grounded_retrieval",
            "answer": answer,
            "confidence": confidence,
            "evidence": evidence,
            "conversation_history": self.conversation_history if use_memory else [],
        }
    
    def clear_memory(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_conversation_history(self) -> list[dict]:
        """Get current conversation history."""
        return self.conversation_history.copy()
