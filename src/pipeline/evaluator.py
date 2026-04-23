from dataclasses import dataclass
from typing import Generator, Optional
from src.core.llm_client import LLMClient
from src.core.budget import BudgetState, BudgetConfig
from src.core.history import HistoryManager, AttemptRecord
from src.executor.sandbox import BaseSandbox
from src.pipeline.validator import CommandValidator


@dataclass
class EvaluationResult:
    status: str
    final_answer: str
    budget_summary: dict
    total_attempts: int
    session_id: str
    stop_reason: str


class EvaluationLoop:
    def __init__(
        self,
        llm: LLMClient,
        sandbox: BaseSandbox,
        validator: CommandValidator,
        config: dict
    ):
        self.llm = llm
        self.sandbox = sandbox
        self.validator = validator
        self.config = config

    def run(
        self,
        user_query: str,
        initial_command: str,
        initial_purpose: str,
        platform_context: dict,
        budget: BudgetState,
        session_id: str
    ) -> Generator[dict, None, EvaluationResult]:

        history = HistoryManager(self.llm, self.config.get("history", {}))

        command = initial_command
        purpose = initial_purpose
        attempt_num = 0

        while True:
            is_exhausted, exhaust_reason = budget.is_exhausted
            if is_exhausted:
                yield {"event": "budget_hit", "reason": exhaust_reason}
                final_answer = self._synthesize(user_query, exhaust_reason, history, budget)
                yield {"event": "complete", "final_answer": final_answer}
                return self._result(
                    "BUDGET_EXHAUSTED", final_answer,
                    budget, history, session_id, exhaust_reason
                )

            attempt_num += 1
            budget.record_cli_attempt()
            yield {"event": "executing", "command": command, "attempt": attempt_num}

            exec_result = self.sandbox.execute(command, timeout=30)
            yield {
                "event": "executed",
                "status": exec_result.status,
                "stdout_preview": exec_result.stdout[:200]
            }

            response = self.llm.call(
                prompt_name="evaluator",
                variables={
                    "user_query": user_query,
                    "command": command,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "returncode": str(exec_result.returncode),
                    "status": exec_result.status,
                    "history_context": history.build_context(),
                    "attempts_used": str(attempt_num),
                    "max_attempts": str(budget.config.max_cli_attempts),
                    "cost_so_far": f"{budget.total_cost_usd:.4f}",
                    "min_confidence": str(budget.config.min_confidence)
                }
            )

            eval_data = response.parse_json()
            confidence = float(eval_data.get("confidence", 0.5))

            budget.record_llm_call(response.input_tokens, response.output_tokens)

            yield {
                "event": "evaluated",
                "confidence": confidence,
                "feedback": eval_data.get("feedback", ""),
                "gaps": eval_data.get("gaps", [])
            }

            record = AttemptRecord(
                attempt=attempt_num,
                command=command,
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                returncode=exec_result.returncode,
                status=exec_result.status,
                eval_feedback=eval_data.get("feedback", ""),
                eval_reasoning=eval_data.get("reasoning", ""),
                confidence=confidence
            )
            history.add(record)

            min_conf = self.config.get("budget", {}).get("min_confidence", 0.8)
            
            is_success = exec_result.status == "success" and exec_result.returncode == 0
            
            if eval_data.get("is_sufficient") and confidence >= min_conf and is_success:
                final_answer = eval_data.get("final_answer", "")
                if not final_answer:
                    final_answer = self._synthesize(user_query, "Sufficient answer from evaluation", history, budget)
                yield {"event": "complete", "final_answer": final_answer}
                return self._result(
                    "COMPLETE", final_answer,
                    budget, history, session_id, "Sufficient answer found"
                )
            
            if not is_success:
                yield {"event": "command_failed", "status": exec_result.status, "stderr": exec_result.stderr[:200]}

            executed_cmds = "\n".join(history.get_all_commands())

            next_response = self.llm.call(
                prompt_name="next_command_generator",
                variables={
                    "user_query": user_query,
                    "feedback": eval_data.get("feedback", ""),
                    "gaps": str(eval_data.get("gaps", [])),
                    "next_direction": eval_data.get("next_direction", ""),
                    "platform": platform_context.get("platform", "unknown"),
                    "shell": platform_context.get("shell", "unknown"),
                    "executed_commands": executed_cmds
                }
            )

            next_data = next_response.parse_json()
            command = next_data.get("command", "")
            purpose = next_data.get("purpose", "")

            budget.record_llm_call(next_response.input_tokens, next_response.output_tokens)

            if not command:
                final = self._synthesize(user_query, "No further commands possible", history, budget)
                yield {"event": "complete", "final_answer": final}
                return self._result("EXHAUSTED", final, budget, history, session_id, "LLM could not generate next command")

            yield {
                "event": "next_command",
                "command": command,
                "purpose": purpose,
                "thinking": next_data.get("thinking", "")
            }

    def _synthesize(self, user_query: str, stop_reason: str, history: HistoryManager, budget: BudgetState) -> str:
        try:
            response = self.llm.call(
                prompt_name="answer_synthesizer",
                variables={
                    "user_query": user_query,
                    "stop_reason": stop_reason,
                    "history_context": history.build_context()
                }
            )
            budget.record_llm_call(response.input_tokens, response.output_tokens)
            return response.content.strip()
        except Exception:
            return f"Investigation ended: {stop_reason}. Check session history for findings."

    def _result(self, status: str, final_answer: str, budget: BudgetState, history: HistoryManager, session_id: str, stop_reason: str) -> EvaluationResult:
        return EvaluationResult(
            status=status,
            final_answer=final_answer,
            budget_summary=budget.summary(),
            total_attempts=history.get_record_count(),
            session_id=session_id,
            stop_reason=stop_reason
        )