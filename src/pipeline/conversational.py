from typing import Generator
from src.core.llm_client import LLMClient
from src.core.budget import BudgetState


class ConversationalBranch:
    def __init__(self, llm: LLMClient, config: dict):
        self.llm = llm
        self.config = config

    def handle(
        self,
        user_query: str,
        context: dict,
        budget: BudgetState
    ) -> Generator[dict, None, None]:
        ctx = context or {}

        response = self.llm.call(
            "conversational_responder",
            {
                "user_query": user_query,
                "expertise_level": ctx.get("expertise_level", "intermediate"),
                "session_context": ctx.get("session_context", "none"),
                "platform": ctx.get("platform", "unknown")
            }
        )

        content = response.content

        import re
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        response_match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)

        thinking = thinking_match.group(1).strip() if thinking_match else ""
        response_text = response_match.group(1).strip() if response_match else content

        budget.record_llm_call(response.input_tokens, response.output_tokens)

        for word in response_text.split():
            yield {"event": "token", "text": word + " "}

        yield {
            "event": "complete",
            "response": response_text.strip(),
            "thinking": thinking.strip()
        }