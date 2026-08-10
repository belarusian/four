# Defrag — Judgment Before Delivery

## The Loop

```
G → V1 → V2* → emit
```

Four functions. Same loop. Two phases.

**Phase 1 — Generation.** G generates code. V1 parses commands. V2 executes bash. Emit commits. The agent floods the canvas: 500+ commits, linear history, additive evolution. No gates. No blockers. Just creation.

**Phase 2 — Judgment.** G generates tickets (auditor). V1 parses issues. V2 executes fixes (validator). Emit commits. The same loop, same algebra, operating on what it created.

The loop is self-referential. It creates, then it judges what it created.

## From 0 to Working

### Step 1: Stream

Start with a goal. Run the stream spoke. The agent writes code, runs tests, commits. Each commit is small: one module, one test, one fix. The runner's EXIT trap creates anchor commits.

```bash
bash run-massive-implementation.sh \
  --goal "Build a personal bookmark manager with search, export, and sync" \
  --variant stream
```

Result: 500+ commits, linear history on `main`. The canvas is flooded. Base knowledge is spread.

### Step 2: Consolidate

The agent scans git history for anchor commits, groups related commits by semantic meaning, creates PRs.

```bash
bash run-massive-implementation.sh \
  --goal "Group commits into semantic PRs" \
  --variant pr-consolidation
```

Result: scattered commits become coherent PRs.

### Step 3: Review

The agent checks out each PR, runs targeted tests, merges if they pass, comments on failures.

```bash
bash run-massive-implementation.sh \
  --goal "Review open PRs" \
  --variant pr-review
```

Result: quality-gated merges to `main`.

### Step 4: Defrag (the higher-order phase)

The same loop, operating at a higher level of abstraction.

**Auditor** (G → V1 → V2* → emit):
- G surveys the codebase, reads modules, runs static analysis
- V1 extracts issues from the evidence
- V2 writes tickets for each issue
- Emit commits the tickets

```bash
bash run-massive-implementation.sh \
  --goal "Audit the codebase. Find issues. Write tickets." \
  --variant auditor
```

**Validator** (G → V1 → V2* → emit):
- G reads tickets, reads code, generates fixes
- V1 extracts commands from the fix plan
- V2 executes fixes, writes tests, runs tests
- Emit commits the fixes

```bash
bash run-massive-implementation.sh \
  --goal "Fix tickets. Write tests. Commit." \
  --variant validator
```

Repeat until the auditor finds no new issues.

**When to run what:**
- Open tickets exist → run the validator first. Clear the debt.
- No open tickets → run the auditor. Find new issues.
- Auditor returns empty → done.

**Prompt template — the goal is the directive:**
```
Auditor:  --goal "Audit the codebase. Find issues: unused imports, dead code, duplicates, broken imports, missing docstrings, syntax errors, type errors. Write tickets/TICKET-NUM.md. Create GitHub issues. Commit each. Output: DONE"

Validator: --goal "Fix tickets TICKET-X through TICKET-Y. Read each ticket. Fix in order. Write tests. Commit. Create PRs. gh pr create. gh issue create. Output: DONE"
```

The goal is not a description. It is the directive. Vague goals produce idle agents. Specific goals produce execution.

### The Ticket Contract

The interface between auditor and validator:

**Auditor writes** `tickets/TICKET-NUM.md`:
```
# TICKET-N: Title

## Evidence
What was read in the code. File paths, line numbers, ruff/mypy output.

## Impact
What breaks or is at risk.

## Suggestion
How to fix it.
```

**Validator reads** `tickets/TICKET-*.md`, expects:
- Sequential numbering (TICKET-1, TICKET-2, ...)
- One issue per file
- Evidence it can verify by reading the code
- A fix it can implement with tests

Both spokes create GitHub issues via `gh issue create` to sync the ticket to the surface.

### Step 5: Delivery

When the auditor finds no new issues and the validator has nothing to fix, the codebase is ready. The loop has judged itself.

## Why It Works

The loop doesn't change. The prompt changes.

- Stream prompt: "Write code. Commit. Test."
- Auditor prompt: "Read code. Find issues. Write tickets."
- Validator prompt: "Read tickets. Fix issues. Write tests."

Same G. Same V1. Same V2. Same emit. Different system prompt.

The agent is the same. The loop is the same. Only the goal changes.

## The Beauty

The same algebra that created the code is now judging the code. Generation and validation are the same process, separated by time and driven by different prompts.

This is judgment before delivery. The loop validates itself.

## Evidence

10 iterations of auditor → validator:
- 45 tickets found and fixed
- 50+ commits of physical fixes
- Dead code removed, duplicates consolidated, type errors resolved, unused imports cleaned
- 2589 tests passing, 0 failures

The loop works. It creates. It judges. It delivers.
