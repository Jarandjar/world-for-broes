"""Shared GPT-5.1 client helpers for the Swarm codebase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import os
from pathlib import Path

from openai import OpenAI
from openai.types.responses import Response

try:  # optional dependencies
    from swarm_vector_db import SwarmVectorDB
except Exception:  # pragma: no cover - optional runtime feature
    SwarmVectorDB = None  # type: ignore

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover - optional runtime feature
    duckdb = None  # type: ignore

CodexRole = str  # system | user | assistant

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = Path(os.getenv("CONTEXT_SUMMARY_DIR", REPO_ROOT / "summaries"))
DEFAULT_SUMMARY_DB = os.getenv("CONTEXT_SUMMARY_DUCKDB_PATH") or os.getenv("EVIDENCE_DUCKDB_PATH", "evidence.duckdb")


class ContextRetriever:
    """Hybrid retriever combining Qdrant vectors, DuckDB, and on-disk summaries."""

    def __init__(self, summary_dir: Path = DEFAULT_SUMMARY_DIR, db_path: str = DEFAULT_SUMMARY_DB):
        self.summary_dir = Path(summary_dir)
        self.db_path = str(db_path) if db_path else ""
        self._vdb: Optional[SwarmVectorDB] = None

    def retrieve(self, query: str, *, top_k: int = 3, max_chars: int = 4000) -> Optional[str]:
        query = (query or "").strip()
        if not query:
            return None

        sections: List[str] = []
        remaining = max_chars

        # 1) Qdrant semantic search
        for text in self._vector_results(query, limit=top_k):
            clipped = self._clip(text, remaining)
            if not clipped:
                continue
            sections.append(clipped)
            remaining -= len(clipped)
            if remaining <= 0:
                break

        # 2) DuckDB fallback
        if remaining > 200 and duckdb is not None:
            for text in self._duckdb_results(query, limit=max(1, top_k - len(sections))):
                clipped = self._clip(text, remaining)
                if not clipped:
                    continue
                sections.append(clipped)
                remaining -= len(clipped)
                if remaining <= 0:
                    break

        # 3) File keyword fallback
        if remaining > 200:
            for text in self._file_results(query, limit=max(1, top_k - len(sections))):
                clipped = self._clip(text, remaining)
                if not clipped:
                    continue
                sections.append(clipped)
                remaining -= len(clipped)
                if remaining <= 0:
                    break

        if not sections:
            return None

        return "\n\n".join(sections).strip()

    def _vector_results(self, query: str, limit: int) -> List[str]:
        if SwarmVectorDB is None:
            return []
        try:
            if self._vdb is None:
                self._vdb = SwarmVectorDB()
        except Exception:
            self._vdb = None
            return []
        if not getattr(self._vdb, "client", None):
            return []
        try:
            results = self._vdb.search(query, collection="context_summaries", limit=limit)
        except Exception:
            return []
        formatted: List[str] = []
        for res in results:
            payload = res.get("payload") or {}
            subsystem = payload.get("subsystem") or payload.get("realm") or "unknown"
            summary_text = payload.get("content") or self._load_summary(subsystem)
            if not summary_text:
                continue
            formatted.append(
                f"### Retrieved: {subsystem} (score {res.get('score', 0):.3f})\n{summary_text.strip()}"
            )
        return formatted

    def _duckdb_results(self, query: str, limit: int) -> List[str]:
        if duckdb is None or not self.db_path:
            return []
        like_pattern = f"%{query.lower()}%"
        try:
            with duckdb.connect(self.db_path) as con:
                rows = con.execute(
                    "SELECT subsystem, summary_md FROM context_summaries WHERE lower(summary_md) LIKE ? LIMIT ?",
                    [like_pattern, limit],
                ).fetchall()
        except Exception:
            return []
        return [f"### Retrieved (DuckDB): {row[0]}\n{row[1].strip()}" for row in rows if row and row[1]]

    def _file_results(self, query: str, limit: int) -> List[str]:
        if not self.summary_dir.exists():
            return []
        terms = [t.lower() for t in query.split() if len(t) >= 3]
        results: List[tuple[int, str, str]] = []
        for path in self.summary_dir.glob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            haystack = text.lower()
            score = sum(haystack.count(term) for term in terms) if terms else haystack.count(query.lower())
            if score > 0:
                results.append((score, path.stem, text))
        results.sort(key=lambda tup: tup[0], reverse=True)
        formatted: List[str] = []
        for _, name, text in results[:limit]:
            formatted.append(f"### Retrieved (files): {name}\n{text.strip()}")
        return formatted

    def _load_summary(self, subsystem: str) -> Optional[str]:
        if not subsystem:
            return None
        file_path = self.summary_dir / f"{subsystem}.md"
        if file_path.exists():
            try:
                return file_path.read_text(encoding="utf-8")
            except Exception:
                pass
        if duckdb is None or not self.db_path:
            return None
        row = None
        try:
            with duckdb.connect(self.db_path) as con:
                row = con.execute(
                    "SELECT summary_md FROM context_summaries WHERE subsystem = ?",
                    [subsystem],
                ).fetchone()
        except Exception:
            return None
        if row:
            return row[0]
        return None

    @staticmethod
    def _clip(text: str, remaining_chars: int) -> Optional[str]:
        snippet = text.strip()
        if not snippet:
            return None
        if len(snippet) <= remaining_chars:
            return snippet
        return snippet[:remaining_chars].rstrip() + "\n...[retrieval truncated]"


@dataclass
class CodexMessage:
    role: CodexRole
    content: str


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class CodexChatResult:
    text: str
    reasoning: Optional[str]
    usage: Optional[TokenUsage]
    raw: Response


class CodexClient:
    """Thin wrapper around the OpenAI Responses API.

    The class centralizes configuration that every agent/service needs:
    - API key / base URL discovery via environment variables
    - Default model (GPT-5.1) and reasoning effort
    - Helper methods to normalize request/response payloads
    - Optional retrieval augmentation that injects context_summaries matches
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        default_context: Optional[str] = None,
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
        max_output_tokens: int = 1200,
        metadata: Optional[Dict[str, str]] = None,
        enable_retrieval: bool = True,
        retrieval_top_k: int = 3,
        retrieval_max_chars: int = 4000,
        retriever: Optional[ContextRetriever] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Export it or pass api_key explicitly.")

        self._client = OpenAI(api_key=key, base_url=base_url or os.getenv("OPENAI_BASE_URL"))
        self._model = model or os.getenv("CODEX_MODEL") or "gpt-5.1"
        self._system_prompt = system_prompt
        self._default_context = default_context
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._metadata = metadata or {}
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_max_chars = retrieval_max_chars
        self._retriever = retriever if retriever is not None else (ContextRetriever() if enable_retrieval else None)

    def chat(
        self,
        *,
        messages: Sequence[CodexMessage],
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        context_query: Optional[str] = None,
        context_top_k: Optional[int] = None,
        context_max_chars: Optional[int] = None,
    ) -> CodexChatResult:
        if not messages:
            raise ValueError("Codex chat requires at least one message")

        base_context = context if context is not None else self._default_context

        retrieved_context: Optional[str] = None
        if context_query and self._retriever:
            retrieved_context = self._retriever.retrieve(
                context_query,
                top_k=context_top_k or self._retrieval_top_k,
                max_chars=context_max_chars or self._retrieval_max_chars,
            )

        enriched_context = base_context
        if retrieved_context:
            context_parts: List[str] = []
            if base_context:
                context_parts.append(base_context)
            context_parts.append(f"Retrieved Context:\n{retrieved_context}")
            enriched_context = "\n\n".join(context_parts)

        payload = self._build_input(messages, system_prompt, enriched_context)
        merged_metadata = {**self._metadata, **(metadata or {})}

        response = self._client.responses.create(
            model=model or self._model,
            input=payload,
            temperature=temperature if temperature is not None else self._temperature,
            max_output_tokens=max_output_tokens if max_output_tokens is not None else self._max_output_tokens,
            reasoning={"effort": reasoning_effort or self._reasoning_effort},
            metadata=merged_metadata or None,
            tools=tools,
            response_format=response_format,
        )

        return CodexChatResult(
            text=self._extract_text(response),
            reasoning=self._extract_reasoning(response),
            usage=self._extract_usage(response),
            raw=response,
        )

    def _build_input(
        self,
        messages: Sequence[CodexMessage],
        system_prompt: Optional[str],
        context: Optional[str],
    ) -> List[Dict[str, Any]]:
        compiled: List[Dict[str, Any]] = []
        system_parts: List[str] = []

        base_prompt = system_prompt or self._system_prompt
        base_context = context or self._default_context

        if base_prompt:
            system_parts.append(base_prompt)
        if base_context:
            system_parts.append(f"Context:\n{base_context}")

        if system_parts:
            compiled.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": part} for part in system_parts],
                }
            )

        for message in messages:
            compiled.append(
                {
                    "role": message.role,
                    "content": [{"type": "text", "text": message.content}],
                }
            )

        return compiled

    @staticmethod
    def _extract_text(response: Response) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text.strip()

        chunks = CodexClient._collect_content(response, {"output_text", "text"})
        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_reasoning(response: Response) -> Optional[str]:
        chunks = CodexClient._collect_content(response, {"reasoning", "reasoning_output"})
        if not chunks:
            return None
        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_usage(response: Response) -> Optional[TokenUsage]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None

        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens
        return TokenUsage(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(total_tokens),
        )

    @staticmethod
    def _collect_content(response: Response, target_types: set[str]) -> List[str]:
        messages = getattr(response, "output", []) or []
        collected: List[str] = []

        for message in messages:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type not in target_types:
                    continue
                text_value = CodexClient._unwrap(getattr(block, "text", None))
                if block_type.startswith("reason"):
                    reasoning_val = CodexClient._unwrap(getattr(block, "reasoning", None))
                    if reasoning_val:
                        collected.append(reasoning_val)
                        continue
                if text_value:
                    collected.append(text_value)
        return collected

    @staticmethod
    def _unwrap(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("value"), str):
            return value["value"]
        return None


__all__ = [
    "CodexClient",
    "CodexMessage",
    "CodexChatResult",
    "TokenUsage",
    "ContextRetriever",
]
