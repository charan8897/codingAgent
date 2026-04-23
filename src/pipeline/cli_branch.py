from typing import Generator
from src.core.llm_client import LLMClient
from src.core.budget import BudgetState
from src.pipeline.evaluator import EvaluationLoop


class CLIBranch:
    def __init__(self, llm: LLMClient, eval_loop: EvaluationLoop, config: dict):
        self.llm = llm
        self.eval_loop = eval_loop
        self.config = config

    def handle(
        self,
        user_query: str,
        context: dict,
        budget: BudgetState,
        session_id: str
    ) -> Generator[dict, None, None]:

        session_context = context.get("session_context", "none")
        
        response = self.llm.call(
            prompt_name="command_generator",
            variables={
                "user_query": user_query,
                "platform": context.get("platform", "linux"),
                "shell": context.get("shell", "bash"),
                "available_tools": context.get("available_tools", "docker, kubectl, git"),
                "domain_context": context.get("domain_context", ""),
                "session_context": session_context
            }
        )

        data = response.parse_json()
        first_command = data.get("command", "")
        purpose = data.get("purpose", "")
        thinking = data.get("thinking", "")

        budget.record_llm_call(response.input_tokens, response.output_tokens)

        yield {
            "event": "first_command",
            "command": first_command,
            "purpose": purpose,
            "thinking": thinking,
            "risk_assessment": data.get("risk_assessment", "medium")
        }

        if not first_command:
            yield {"event": "error", "message": "Failed to generate command"}
            return

        for event in self.eval_loop.run(
            user_query=user_query,
            initial_command=first_command,
            initial_purpose=purpose,
            platform_context=context,
            budget=budget,
            session_id=session_id
        ):
            yield event