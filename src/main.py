#!/usr/bin/env python3
import sys
import os
import yaml
import uuid
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.prompt_engine import PromptRegistry
from src.core.llm_client import LLMClient
from src.core.budget import BudgetConfig, BudgetState
from src.core.history import HistoryManager, AttemptRecord
from src.executor.sandbox import BaseSandbox, RestrictedSubprocess, DockerSandbox
from src.pipeline.classifier import Classifier
from src.pipeline.validator import CommandValidator
from src.pipeline.evaluator import EvaluationLoop
from src.pipeline.conversational import ConversationalBranch
from src.pipeline.cli_branch import CLIBranch
from src.storage.session_store import SessionStore


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "src" / "config" / "system_config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            config_path = PROJECT_ROOT / "src" / "config" / "system_config.yaml"
    return yaml.safe_load(config_path.read_text())


def build_sandbox(config: dict) -> BaseSandbox:
    mode = config.get("executor", {}).get("sandbox_mode", "subprocess")
    if mode == "docker":
        return DockerSandbox(
            image=config["executor"].get("docker_image", "cli-intel-sandbox:latest"),
            memory_limit=config["executor"].get("docker_memory_limit", "256m"),
            cpu_limit=config["executor"].get("docker_cpu_limit", "0.5"),
            network_enabled=config["executor"].get("network_enabled", False)
        )
    return RestrictedSubprocess()


class CLIIntelligence:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)

        prompts_dir = self.config["prompts"]["directory"]
        if not Path(prompts_dir).is_absolute():
            prompts_dir = PROJECT_ROOT / prompts_dir

        self.registry = PromptRegistry(
            prompts_dir=prompts_dir,
            hot_reload=self.config["prompts"]["hot_reload"]
        )

        budget_cfg = self.config["budget"]
        self.budget_config = BudgetConfig(
            max_llm_calls=budget_cfg["max_llm_calls"],
            max_cost_usd=budget_cfg["max_cost_usd"],
            max_total_tokens=budget_cfg["max_total_tokens"],
            max_wall_seconds=budget_cfg["max_wall_seconds"],
            max_cli_attempts=budget_cfg["max_cli_attempts"],
            min_confidence=budget_cfg["min_confidence"]
        )

        self.session_store = None
        if self.config.get("session", {}).get("persist", False):
            db_path = self.config["session"].get("db_path", "~/.cli_intel/sessions.db")
            if db_path.startswith("~"):
                db_path = os.path.expanduser(db_path)
            else:
                db_path = PROJECT_ROOT / db_path
            self.session_store = SessionStore(db_path)

        self.sandbox = build_sandbox(self.config)

        self.llm = LLMClient(
            registry=self.registry,
            config=self.config
        )

        self.classifier = Classifier(self.llm)
        self.validator = CommandValidator(self.llm)

        self.eval_loop = EvaluationLoop(
            llm=self.llm,
            sandbox=self.sandbox,
            validator=self.validator,
            config=self.config
        )

        self.conv_branch = ConversationalBranch(
            llm=self.llm,
            config=self.config
        )

        self.cli_branch = CLIBranch(
            llm=self.llm,
            eval_loop=self.eval_loop,
            config=self.config
        )

    def build(
        self,
        user_query: str,
        context: dict = None,
        debug: bool = False,
        session_id: str = None,
        session_context: str = None
    ):
        ctx = context.copy() if context else {}
        budget = BudgetState(config=self.budget_config)
        
        if session_id is None:
            if hasattr(self, '_current_session_id') and self._current_session_id:
                session_id = self._current_session_id
            else:
                session_id = str(uuid.uuid4())[:8]
        
        self._current_session_id = session_id
        self._last_session_id = session_id
        
        if session_context is None and self.session_store and session_id:
            current_session = self.session_store.get_session(session_id)
            if current_session:
                session_context = f"Previous query in this session: {current_session['user_query']}"
            else:
                prev_sessions = self.session_store.get_recent_sessions(3)
                session_context = "; ".join([s["user_query"] for s in prev_sessions if s["id"] != session_id])

        ctx["session_context"] = session_context or "none"

        print(f"\n{'─' * 60}")
        print(f"Query  : {user_query}")
        print(f"Session: {session_id}")
        if session_context and session_context != "none":
            print(f"Context: Previous queries: {session_context[:80]}...")
        print(f"{'─' * 60}\n")

        classification = self.classifier.classify(user_query, budget, ctx)

        if debug:
            print(f"[Intent]  : {classification.intent} "
                  f"(confidence={classification.confidence:.0%})")
            print(f"[Reason]  : {classification.reasoning}")
            print(f"[Prompt v]: {classification.prompt_version}\n")

        if self.session_store:
            self.session_store.save_session(session_id, {
                "user_query": user_query,
                "intent": classification.intent
            })

        if classification.intent == "AMBIGUOUS":
            print(f"[Clarification needed]: {classification.ambiguity_resolution}\n")
            return

        if classification.intent == "CONVERSATIONAL":
            self._handle_conversational(user_query, ctx, budget, debug)
        elif classification.intent == "CLI_DEPENDENT":
            self._handle_cli(user_query, ctx, budget, session_id, debug)

        summary = budget.summary()
        print(f"\n{'─' * 60}")
        print(f"LLM Calls: {summary['llm_calls']}/{summary['max_llm_calls']} | "
              f"CLI Attempts: {summary['cli_attempts']}/{summary['max_cli_attempts']} | "
              f"Time: {summary['elapsed_seconds']:.1f}s")
        print(f"{'─' * 60}\n")

    def _handle_plan_only(self, user_query: str):
        """Handle a query in PLAN mode (read-only)"""
        print(f"\n{'─' * 60}")
        print(f"PLAN MODE (Read-Only)")
        print(f"Query: {user_query}")
        print(f"{'─' * 60}\n")
        
        from src.pipeline.classifier import Classifier
        from src.core.budget import BudgetConfig, BudgetState
        
        budget = BudgetState(config=self.budget_config)
        
        classification = self.classifier.classify(user_query, budget, {"session_context": "none"})
        
        print(f"[Intent Classification]")
        print(f"  - Intent: {classification.intent}")
        print(f"  - Confidence: {classification.confidence:.0%}")
        print(f"  - Reasoning: {classification.reasoning}")
        print()
        
        print(f"[Step-by-Step Plan]")
        print(f"  1. Classify intent → {classification.intent}")
        
        if classification.intent == "CONVERSATIONAL":
            print(f"  2. Route to conversational branch")
            print(f"  3. Generate response (no CLI commands)")
        elif classification.intent == "CLI_DEPENDENT":
            print(f"  2. Generate first command using command_generator prompt")
            print(f"  3. Validate command using validator prompt")
            print(f"  4. Execute in sandbox")
            print(f"  5. Evaluate output using evaluator prompt")
            print(f"  6. If insufficient, iterate with next_command prompt")
        else:
            print(f"  2. Ask for clarification: {classification.ambiguity_resolution}")
        
        print(f"\n{'─' * 60}")
        print(f"PLAN MODE - No execution performed")
        print(f"{'─' * 60}\n")

    def _handle_conversational(
        self,
        user_query: str,
        ctx: dict,
        budget: BudgetState,
        debug: bool
    ):
        print("[CONVERSATIONAL]\n")
        for event in self.conv_branch.handle(user_query, ctx, budget):
            if event["event"] == "token":
                print(event["text"], end="", flush=True)
            elif event["event"] == "complete":
                print()
                if debug and event.get("thinking"):
                    print(f"\n[THINKING]: {event['thinking']}")

    def _handle_cli(
        self,
        user_query: str,
        ctx: dict,
        budget: BudgetState,
        session_id: str,
        debug: bool
    ):
        print("[CLI INVESTIGATION]\n")
        for event in self.cli_branch.handle(user_query, ctx, budget, session_id):
            self._render_event(event, debug)

    def _render_event(self, event: dict, debug: bool):
        ev = event.get("event")

        ICONS = {
            "first_command": "→",
            "executing": "▶",
            "executed": {"success": "✓", "error": "✗", "timeout": "⏱"},
            "evaluated": "◇",
            "complete": "✔",
            "blocked": "🚫",
            "warn": "⚠",
            "budget_hit": "💰",
            "next_command": "⟳"
        }

        if ev == "first_command":
            print(f"  {ICONS['first_command']} Command: {event['command']}")
            if debug:
                print(f"    Purpose: {event.get('purpose', '')}")
                print(f"    Thinking: {event.get('thinking', '')[:200]}")
            print()

        elif ev == "executing":
            print(f"  [{event['attempt']}] {ICONS['executing']} {event['command']}")

        elif ev == "executed":
            icon = ICONS["executed"].get(event["status"], "?")
            status = event.get("status", "unknown")
            print(f"    {icon} {status}")

        elif ev == "evaluated":
            conf = event.get("confidence", 0)
            bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            print(f"    [{bar}] {conf:.0%} — {event.get('feedback', '')}")

        elif ev == "complete":
            print(f"\n{'═' * 60}")
            print(event.get("final_answer", ""))
            print(f"{'═' * 60}")

        elif ev == "blocked":
            print(f"  {ICONS['blocked']} BLOCKED: {event.get('reason', '')}")
            if event.get("alternative"):
                print(f"  Alternative: {event['alternative']}")

        elif ev == "warn":
            print(f"  {ICONS['warn']} WARN: {event.get('reason', '')}")

        elif ev == "budget_hit":
            print(f"\n  {ICONS['budget_hit']} {event.get('reason', '')}")

        elif ev == "next_command":
            print(f"  {ICONS['next_command']} Next: {event['command']}")


PLAN_MODE_CONTEXT = """You are in PLAN MODE - a read-only phase.

### Forbidden:
- Any file edits or modifications
- Running commands that manipulate files (sed, tee, echo, etc.)
- Making any system changes

### Permitted:
- Read/inspect files
- Analyze code
- Search and explore
- Create a plan

### Your Task:
Analyze this user query and create a step-by-step plan to execute it. Do NOT execute anything - just plan.

User Query: {query}

Provide your plan:"""


def main():
    args = sys.argv[1:]
    
    if not args:
        print("Usage: python3 main.py <command> [query]")
        print("Commands:")
        print("  /plan {query}   - Plan only (read-only)")
        print("  /build {query}  - Build/Execute (full mode)")
        print("  -d, --debug   Show details")
        sys.exit(1)
    
    command = args[0]
    query = " ".join(args[1:])
    
    debug = "--debug" in args or "-d" in args
    
    if command not in ["/plan", "/build"]:
        print(f"Unknown command: {command}")
        print("Use /plan {query} or /build {query}")
        sys.exit(1)
    
    if not query or query.startswith("/"):
        if not query:
            print("Error: Query required")
        else:
            print(f"Error: Invalid query '{query}'")
        print("Usage: /plan {query} or /build {query}")
        sys.exit(1)
    
    if command == "/plan":
        from src.core.llm_client import LLMClient
        from src.core.budget import BudgetConfig, BudgetState
        config = load_config()
        registry = PromptRegistry(
            prompts_dir=PROJECT_ROOT / config["prompts"]["directory"],
            hot_reload=config["prompts"]["hot_reload"]
        )
        llm = LLMClient(registry, config)
        budget = BudgetState(config=BudgetConfig.from_dict(config))
        
        print(f"\n{'='*60}")
        print(f"PLAN MODE")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        # Classify intent
        response = llm.call("intent_classifier", {
            "user_query": query, 
            "os_hint": "linux", 
            "session_context": "none", 
            "expertise_level": "intermediate"
        })
        data = response.parse_json()
        intent = data.get("intent")
        
        print(f"[Intent] {intent} (confidence: {data.get('confidence')})")
        print(f"[Reasoning] {data.get('reasoning')}")
        print()
        
        # For CONVERSATIONAL, give the actual response
        if intent == "CONVERSATIONAL":
            print(f"[Plan] Route to conversational branch → generate response\n")
            print(f"{'='*60}")
            print(f"RESPONSE:")
            print(f"{'='*60}")
            conv_response = llm.call("conversational_responder", {
                "user_query": query,
                "expertise_level": "intermediate",
                "session_context": "none",
                "platform": "linux"
            })
            conv_data = conv_response.parse_json()
            # Extract just the response part from the structured output
            response_text = conv_data.get("text", conv_data.get("response", ""))
            if not response_text:
                content = conv_response.content
                import re
                match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
                response_text = match.group(1) if match else content
            print(response_text)
            print(f"\n{'='*60}")
            print(f"(Plan mode - conversational response only)")
            return
        
# For CLI_DEPENDENT, show intent only (no detailed plan)
        if intent == "CLI_DEPENDENT":
            print(f"\n[Plan] Would execute CLI commands to retrieve information")
            print(f"\n{'='*60}")
            print(f"(Use /build to execute)")
            return
        
    elif command == "/build":
        interactive = "-i" in args
        for flag in ["-i", "--interactive"]:
            if flag in args:
                args.remove(flag)
        
        debug = "--debug" in args or "-d" in args
        if debug:
            args.remove("--debug")
        
        if not query:
            query = None
        
        if interactive or not query:
            agent = CLIIntelligence()
            current_session = None
            print("\n" + "="*60)
            print("BUILD MODE - Interactive")
            print("  /plan {query}   - Switch to plan mode")
            print("  quit          - Exit")
            print("="*60 + "\n")
            while True:
                try:
                    user_input = input("> ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("Goodbye!")
                        break
                    if user_input.startswith("/plan "):
                        q = user_input[6:]
                        agent._handle_plan_only(q)
                        continue
                    if user_input.startswith("/plan"):
                        print("Usage: /plan {query}")
                        continue
                    agent.build(user_input, debug=debug, session_id=current_session)
                    current_session = getattr(agent, '_last_session_id', current_session)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nUse 'quit' to exit")
                    break
            return
        
        agent = CLIIntelligence()
        agent.build(query, debug=debug)
    
    elif command == "/plan":
        print("Usage: /plan {query}")
    
    else:
        print(f"Unknown command: {command}")
        print("Use /plan {query} or /build {query}")


if __name__ == "__main__":
    main()