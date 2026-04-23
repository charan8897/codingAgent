import subprocess
from typing import Any


PLAN_MODE_CONTEXT = """You are in **PLAN MODE**. In this mode, you are in a read-only phase.

### Forbidden:
- Any file edits or modifications
- Running commands that manipulate files (sed, tee, echo, etc.)
- Making any system changes

### Permitted:
- Read/inspect files
- Analyze code
- Search and explore
- Create a plan

---

User query: {query}
"""


def plan(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Plan a task based on user input in read-only mode."""
    user_query = tool_input.get("query", "")
    prompt = PLAN_MODE_CONTEXT.format(query=user_query)
    return {"status": "success", "prompt": prompt, "query": user_query}


def build(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Build based on the plan."""
    result = subprocess.run(
        ["echo", "Building..."],
        capture_output=True,
        text=True
    )
    return {"status": "success", "output": result.stdout.strip()}