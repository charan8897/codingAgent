import time
from dataclasses import dataclass, field
from typing import Optional
from src.core.llm_client import LLMClient


@dataclass
class AttemptRecord:
    attempt: int
    command: str
    stdout: str
    stderr: str
    returncode: int
    status: str
    eval_feedback: str = ""
    eval_reasoning: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_compact(self) -> str:
        return f"[{self.attempt}] {self.command} → {self.status} (conf:{self.confidence:.0%})"

    def to_full(self) -> str:
        lines = [
            f"=== Attempt {self.attempt} ===",
            f"Command: {self.command}",
            f"Status: {self.status} | Return code: {self.returncode}",
        ]
        if self.eval_feedback:
            lines.append(f"Feedback: {self.eval_feedback}")
            lines.append(f"Reasoning: {self.eval_reasoning}")
        if self.stdout:
            lines.append(f"Stdout (first 500): {self.stdout[:500]}")
        if self.stderr:
            lines.append(f"Stderr: {self.stderr[:200]}")
        return "\n".join(lines)


class HistoryManager:
    def __init__(self, llm: LLMClient, config: dict):
        self.llm = llm
        self.config = config
        self.recent_window = config.get("recent_window", 3)
        self.summary_trigger = config.get("summary_trigger", 4)
        self.max_stdout = config.get("max_stdout_per_entry", 3000)
        self.max_stderr = config.get("max_stderr_per_entry", 500)
        self._records: list[AttemptRecord] = []
        self._cached_summary: Optional[str] = None

    def add(self, record: AttemptRecord):
        record.stdout = record.stdout[:self.max_stdout]
        record.stderr = record.stderr[:self.max_stderr]
        self._records.append(record)
        self._cached_summary = None

    def build_context(self) -> str:
        if not self._records:
            return "No investigation history yet."

        if len(self._records) <= self.recent_window:
            return self._build_full_context(self._records)

        old_records = self._records[:-self.recent_window]
        recent_records = self._records[-self.recent_window:]

        sections = []

        if old_records:
            summary = self._get_or_build_summary(old_records)
            sections.append(f"HISTORY SUMMARY ({len(old_records)} earlier attempts):\n{summary}\n")

        sections.append("RECENT DETAIL:")
        sections.append(self._build_full_context(recent_records))

        return "\n".join(sections)

    def _build_full_context(self, records: list[AttemptRecord]) -> str:
        return "\n".join(r.to_full() for r in records)

    def _get_or_build_summary(self, old_records: list[AttemptRecord]) -> str:
        if self._cached_summary:
            return self._cached_summary

        if len(old_records) < self.summary_trigger:
            return "Not enough history to summarize."

        attempts_text = "\n".join(r.to_compact() for r in old_records)

        try:
            response = self.llm.call(
                prompt_name="history_summarizer",
                variables={
                    "attempt_count": str(len(old_records)),
                    "attempts_text": attempts_text
                }
            )
            summary = response.content.strip()
            self._cached_summary = summary
            return summary
        except Exception as e:
            return f"Summary unavailable: {e}"

    def get_all_commands(self) -> list[str]:
        return [r.command for r in self._records]

    def get_record_count(self) -> int:
        return len(self._records)

    def get_last_record(self) -> Optional[AttemptRecord]:
        return self._records[-1] if self._records else None