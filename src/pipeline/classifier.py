from dataclasses import dataclass
from typing import Optional
from src.core.llm_client import LLMClient
from src.core.budget import BudgetState


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    reasoning: str
    ambiguity_resolution: str = ""
    prompt_version: str = ""


class Classifier:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def classify(
        self,
        user_query: str,
        budget: BudgetState,
        context: dict = None
    ) -> ClassificationResult:
        ctx = context or {}

        response = self.llm.call(
            prompt_name="intent_classifier",
            variables={
                "user_query": user_query,
                "os_hint": ctx.get("os_hint", "unknown"),
                "session_context": ctx.get("session_context", "none"),
                "expertise_level": ctx.get("expertise_level", "intermediate")
            }
        )

        data = response.parse_json()
        template = self.llm.registry.get("intent_classifier")

        budget.record_llm_call(response.input_tokens, response.output_tokens)

        return ClassificationResult(
            intent=data.get("intent", "CONVERSATIONAL"),
            confidence=float(data.get("confidence", 0.7)),
            reasoning=data.get("reasoning", ""),
            ambiguity_resolution=data.get("ambiguity_resolution", ""),
            prompt_version=template.version
        )