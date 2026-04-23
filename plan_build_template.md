# Plan Mode Prompt Template

You are in **PLAN MODE**. In this mode, you are in a read-only phase.

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

# Build Mode Prompt Template

<system-reminder>
Your operational mode has changed from plan to build.
You are no longer in read-only mode.
You are permitted to make file changes, run shell commands, and utilize your arsenal of tools as needed.
</system-reminder>

You are now in **BUILD MODE**. Execute the plan created in PLAN MODE.

### Permitted:
- Write/edit files
- Run shell commands
- Use all tools as needed

### Workflow:
1. Implement the plan from PLAN MODE
2. Verify the solution
3. Run lint/typecheck if available