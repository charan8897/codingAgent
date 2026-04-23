from dataclasses import dataclass
from typing import Optional
from src.core.llm_client import LLMClient
from src.core.budget import BudgetState


@dataclass
class ValidationResult:
    decision: str
    confidence: float
    reasoning: str
    specific_concerns: list[str]
    safer_alternative: Optional[str]


class CommandValidator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def validate(
        self,
        command: str,
        user_query: str,
        command_purpose: str,
        execution_environment: str,
        prior_commands: list[str],
        budget: BudgetState
    ) -> ValidationResult:

        prior_cmd_text = "\n".join(
            f"  {i+1}. {cmd}" 
            for i, cmd in enumerate(prior_commands)
        ) or "  None"

        response = self.llm.call(
            prompt_name="command_validator",
            variables={
                "command": command,
                "user_query": user_query,
                "command_purpose": command_purpose,
                "execution_environment": execution_environment,
                "prior_commands": prior_cmd_text
            }
        )

        data = response.parse_json()

        return ValidationResult(
            decision=data.get("decision", "BLOCK"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", "Validation failed to parse"),
            specific_concerns=data.get("specific_concerns", []),
            safer_alternative=data.get("safer_alternative")
        )