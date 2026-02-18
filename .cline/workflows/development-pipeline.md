# Sphana Development Pipeline - Orchestrator

> **TRIGGER**: The user says: "Follow the development-pipeline workflow to implement: [request]"
> You MUST read this entire file before doing anything else.
> You MUST follow every instruction below. No exceptions. No shortcuts.

---

## Overview

This is a **5-phase development pipeline** where you (Cline) act as different specialized roles
sequentially. Each phase produces artifacts that feed into the next. Review phases can reject
work back to earlier phases for revision.

---

## Pipeline Flow

Phase 1: Requirements Researcher
    |
    v
Phase 2: Python Architect
    |
    v
Phase 3: Architecture Reviewer --[REJECT]--> Phase 1 or Phase 2
    |
    [APPROVED]
    v
Phase 4: Python Developer
    |
    v
Phase 5: Code Reviewer --[REJECT]--> Phase 4 or Phase 2
    |
    [APPROVED]
    v
DONE - Clean up artifacts

---

## Execution Instructions

### Before Starting
1. Read `.clinerules` to understand project conventions
2. Create `.cline/artifacts/` directory if it does not exist

### Phase 1: Requirements Researcher
1. Read `.cline/workflows/requirements-researcher.md` COMPLETELY
2. Adopt the Requirements Researcher role
3. Follow ALL instructions in that file
4. Produce: `.cline/artifacts/requirements.md`

### Phase 2: Python Architect
1. Read `.cline/workflows/python-architect.md` COMPLETELY
2. Adopt the Python Architect role
3. Follow ALL instructions in that file
4. Produce: `.cline/artifacts/architecture.md`

### Phase 3: Architecture Reviewer
1. Read `.cline/workflows/architecture-reviewer.md` COMPLETELY
2. Adopt the Architecture Reviewer role
3. Follow ALL instructions in that file
4. Produce: `.cline/artifacts/arch-review.md`
5. **Decision routing:**
   - **APPROVED**: Proceed to Phase 4
   - **REVISION_NEEDED -> Phase 1**: Return to Phase 1 (increment iteration counter)
   - **REVISION_NEEDED -> Phase 2**: Return to Phase 2 (increment iteration counter)

### Phase 4: Python Developer
1. Read `.cline/workflows/python-developer.md` COMPLETELY
2. Adopt the Python Developer role
3. Follow ALL instructions in that file
4. Produce: implemented code files

### Phase 5: Code Reviewer
1. Read `.cline/workflows/code-reviewer.md` COMPLETELY
2. Adopt the Code Reviewer role
3. Follow ALL instructions in that file
4. Produce: `.cline/artifacts/code-review.md`
5. **Decision routing:**
   - **APPROVED**: Pipeline complete
   - **REVISION_NEEDED -> Phase 4**: Return to Phase 4 (increment iteration counter)
   - **REVISION_NEEDED -> Phase 2**: Return to Phase 2 (increment iteration counter)

---

## Feedback Loop Rules

### Maximum Iterations
- Architecture review loop (Phase 2-3): **3 iterations maximum**
- Code review loop (Phase 4-5): **3 iterations maximum**
- If max iterations exceeded: **STOP and escalate to user**

### Iteration Tracking
- Track iteration count in review artifacts (Section 3: Iteration)
- Each review document shows "[N] of 3"
- When returning to an earlier phase, the phase MUST read the review feedback

---

## Artifact Management

### During Pipeline
All intermediate artifacts are stored in `.cline/artifacts/`:
- `requirements.md` - Phase 1 output
- `architecture.md` - Phase 2 output
- `arch-review.md` - Phase 3 output
- `code-review.md` - Phase 5 output

### On Successful Completion
After Phase 5 produces APPROVED:
1. Delete all files in `.cline/artifacts/`
2. Report final summary to user

### On Escalation (Max Iterations Exceeded)
1. Keep all artifacts for user review
2. Report which phase failed and why
3. Present the latest review document to the user

---

## Cross-Cutting Concerns (Enforced in Phases 2-5)

These concerns MUST be addressed in every phase from architecture through code review:

1. **SOLID Design Principles** - SRP, OCP, LSP, ISP, DIP
2. **Python Best Practices** - PEP 8, modern type hints, dataclasses, proper imports
3. **Observability** - Prometheus metrics, structured logging
4. **Security** - No PII/secrets in logs, input validation, secure defaults

---

## Completion

When the pipeline finishes successfully, use `attempt_completion` with a summary including:
- Development Pipeline Complete
- Original user request
- Phases completed: 5/5
- Review iterations: Architecture=[N], Code=[N]
- Files created and modified
- Artifacts cleaned up status