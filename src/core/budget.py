import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BudgetConfig:
    max_llm_calls: int = 12
    max_cost_usd: float = 0.50
    max_total_tokens: int = 80000
    max_wall_seconds: int = 120
    max_cli_attempts: int = 8
    min_confidence: float = 0.80

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetConfig":
        b = d.get("budget", {})
        return cls(
            max_llm_calls=b.get("max_llm_calls", 12),
            max_cost_usd=b.get("max_cost_usd", 0.50),
            max_total_tokens=b.get("max_total_tokens", 80000),
            max_wall_seconds=b.get("max_wall_seconds", 120),
            max_cli_attempts=b.get("max_cli_attempts", 8),
            min_confidence=b.get("min_confidence", 0.80)
        )


@dataclass
class BudgetState:
    config: BudgetConfig
    start_time: float = field(default_factory=time.time)
    llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cli_attempts: int = 0
    total_cost_usd: float = 0.0

    COST_PER_1K_INPUT = 0.0
    COST_PER_1K_OUTPUT = 0.0

    def record_llm_call(self, input_tokens: int, output_tokens: int):
        self.llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT
        output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT
        self.total_cost_usd += input_cost + output_cost

    def record_cli_attempt(self):
        self.cli_attempts += 1

    @property
    def is_exhausted(self) -> tuple[bool, Optional[str]]:
        if self.llm_calls >= self.config.max_llm_calls:
            return True, f"LLM calls exhausted: {self.llm_calls}/{self.config.max_llm_calls}"

        if self.total_cost_usd >= self.config.max_cost_usd:
            return True, f"Cost limit reached: ${self.total_cost_usd:.4f}/${self.config.max_cost_usd:.2f}"

        total_tokens = self.total_input_tokens + self.total_output_tokens
        if total_tokens >= self.config.max_total_tokens:
            return True, f"Token limit reached: {total_tokens}/{self.config.max_total_tokens}"

        elapsed = time.time() - self.start_time
        if elapsed >= self.config.max_wall_seconds:
            return True, f"Time limit reached: {elapsed:.0f}s/{self.config.max_wall_seconds}s"

        if self.cli_attempts >= self.config.max_cli_attempts:
            return True, f"CLI attempts exhausted: {self.cli_attempts}/{self.config.max_cli_attempts}"

        return False, None

    def summary(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.config.max_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "max_total_tokens": self.config.max_total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "max_cost_usd": self.config.max_cost_usd,
            "cli_attempts": self.cli_attempts,
            "max_cli_attempts": self.config.max_cli_attempts,
            "elapsed_seconds": elapsed,
            "max_wall_seconds": self.config.max_wall_seconds
        }