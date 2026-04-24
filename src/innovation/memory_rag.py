"""
Part G: Innovation — Conversation Memory RAG (Free Version)
-----------------------------------------------------------
Maintains recent conversation turns and resolves short follow-up questions
without using a paid API. Resolution is rule-based, transparent and sufficient
for examples like "And Volta?" after a question about NDC votes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("logs/memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class ConversationMemory:
    def __init__(self, session_id: str, max_turns: int = 5):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: list[dict] = []
        self._load_from_disk()

    def add_turn(self, question: str, answer: str, chunks_used: list[dict] | None = None) -> None:
        turn = {
            "turn_number": len(self.turns) + 1,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat(),
            "chunks_used": [c.get("text", "")[:100] for c in (chunks_used or [])],
        }
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        self._save_to_disk()

    def get_recent_turns(self, n: int | None = None) -> list[dict]:
        return self.turns[-(n or self.max_turns):]

    def format_history_for_prompt(self) -> str:
        if not self.turns:
            return ""
        lines = ["CONVERSATION HISTORY (for context only):"]
        for t in self.get_recent_turns():
            answer_preview = t["answer"][:200] + "..." if len(t["answer"]) > 200 else t["answer"]
            lines.append(f"Q{t['turn_number']}: {t['question']}")
            lines.append(f"A{t['turn_number']}: {answer_preview}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns = []
        self._save_to_disk()

    def resolve_query(self, new_query: str) -> str:
        """Rule-based follow-up expansion using the most recent question."""
        if not self.turns:
            return new_query

        words = new_query.split()
        q_lower = new_query.lower().strip()
        followup_triggers = ("and ", "what about", "how about", "same for", "there", "that")
        is_short_followup = len(words) <= 4 or q_lower.startswith(followup_triggers)
        if not is_short_followup:
            return new_query

        previous_question = self.turns[-1]["question"]
        cleaned = q_lower.replace("what about", "").replace("how about", "").replace("and", "").strip(" ?")
        if not cleaned:
            return new_query

        resolved = f"{previous_question} Also answer specifically for {cleaned}."
        logger.info("Memory resolved query: %r -> %r", new_query, resolved)
        return resolved

    def _session_file(self) -> Path:
        safe_id = "".join(c for c in self.session_id if c.isalnum() or c in "-_")
        return MEMORY_DIR / f"{safe_id}.json"

    def _save_to_disk(self) -> None:
        with open(self._session_file(), "w", encoding="utf-8") as f:
            json.dump({"session_id": self.session_id, "turns": self.turns}, f, indent=2)

    def _load_from_disk(self) -> None:
        path = self._session_file()
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    self.turns = json.load(f).get("turns", [])
            except Exception as exc:
                logger.warning("Could not load memory: %s", exc)


class MemoryRAGPipeline:
    def __init__(self, base_pipeline, session_id: str = "default"):
        self.base = base_pipeline
        self.memory = ConversationMemory(session_id=session_id)

    def query(self, user_question: str) -> dict:
        resolved_query = self.memory.resolve_query(user_question)
        result = self.base.query(resolved_query)

        if resolved_query != user_question:
            result["query_resolved_to"] = resolved_query
            result["answer"] = f"[Interpreted as: {resolved_query}]\n\n" + result["answer"]

        self.memory.add_turn(
            question=resolved_query,
            answer=result["answer"],
            chunks_used=result.get("retrieved_chunks", []),
        )
        result["conversation_turn"] = len(self.memory.turns)
        result["memory_context"] = self.memory.format_history_for_prompt()
        return result

    def clear_memory(self) -> None:
        self.memory.clear()
