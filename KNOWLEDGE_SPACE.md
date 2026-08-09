# Four — Invariant Knowledge

## What This Is

Four functions compose. The loop is the evaluator.

This is a minimal agent framework. Any autonomous workflow is the same loop with different functions plugged in. No config files, no SDK, no boilerplate. Four callables, chained.

The thesis: **commits grow out, PRs shrink in.** Autonomous development is a three-stage pipeline — unconstrained creation, post-hoc compression, quality validation — separated so they don't entangle.

---

## The Algebra

```
invoke   : G    -- messages → Result[raw]
parse    : V1   -- raw → Result[list[action]]
validate : V2   -- action → Result[observation | Exit]
emit     : IO   -- (messages, outcome) → Path
```

Loop: `(G → V1 → [V2, V2, ...])* → emit`

One iteration: G queries the model, V1 extracts commands, V2 executes each command and captures output, output feeds back into the conversation, loop repeats. When the model returns no code block or max steps is reached, emit persists the trajectory.

Format errors become user messages — the model sees its own mistake and self-corrects. No retry logic needed.

The evaluator is 22 lines. It lives in `src/four/core.py`. It does not know what it's evaluating. It only chains `Result` types.

---

## The Four Functions (Swappable, Always)

**G — invoke.** Queries an LLM. Returns `Ok(text)` or `Err(reason)`.
- `litellm_invoke()` — plain text, no tools
- `litellm_toolcall_invoke()` — structured tool calls
- `context_aware_invoke(fast, large, ...)` — switches models by context size
- `summarizing_invoke(wrap, ...)` — progressive history compression
- `retry_invoke(wrap, ...)` — exponential backoff on transient errors

**V1 — parse.** Extracts actions from raw LLM output. Returns `Ok(list)` or `Err(reason)`.
- `regex_parse()` — finds \`\`\`bash code blocks, returns all matches
- `toolcall_parse()` — parses JSON tool call payloads into commands

**V2 — validate.** Executes each action. Returns `Ok(observation)` or `Err(exit)`.
- `local_env(timeout, max_output)` — subprocess with capture, truncation, exit signal

**emit — IO.** Persists the run. Returns `Path`.
- `save_trajectory(dir)` — JSON trajectory files, auto-incrementing

Remove any one and the loop breaks. Four is the minimum.

---

## Spokes — The Control Surface

A spoke is a Python script that wires four functions together with a system prompt and calls `run()`. The system prompt is the primary control surface — it drives everything the agent does.

Each spoke lives in `examples/spokes/`. The runner (`run-massive-implementation.sh`) selects one by `--variant`.

### Variants

| Variant | What it does |
|---|---|
| `stream` | Unimpeded code flow. Agent writes, tests, commits linearly. No blockers. |
| `long-running` | Same as stream, with `summarizing_invoke` for 200+ step runs |
| `documentation` | Post-hoc documentation of existing code |
| `pr-consolidation` | Phase 2: scans git anchors, creates semantic PRs |
| `pr-review` | Phase 3: reads open PRs, runs tests, merges |
| `context-aware` | Two-model fallback for growing context |
| `durable-state` | State persistence across runs |
| `massive` | Large-scale feature implementation |

### Creating a spoke

Copy any spoke. Change the system prompt, set max_steps, add the variant to the runner's case statement. The loop never changes.

---

## The Three-Phase Pipeline

### Phase 1: Stream — Grow

The agent flows freely. It writes code, runs tests, commits after each logical unit. Each commit is small: one module, one test, one fix. The runner's EXIT trap creates an anchor commit (`autonomous: save progress`) after every run, pushing to remote.

Result: linear history, hundreds of commits, additive evolution.

### Phase 2: Consolidate — Compress

The agent scans the git history for anchor commits. It reads the commits between anchors, groups them by semantic meaning, cherry-picks each group onto a new branch, and creates a PR.

Result: scattered commits become coherent PRs.

### Phase 3: Review — Validate

The agent checks out each PR, runs only the tests that changed in that PR, rebases onto main, and merges. If tests fail, it comments with the specific failures.

Result: quality-gated merges to main.

---

## Model Infrastructure

Two local LLM instances. One fast (throughput), one large (context). The `context_aware_invoke` wrapper switches automatically:

- Small context → fast model
- Medium context → large model
- Overflow → compress history to system prompt + last N messages

The `summarizing_invoke` wrapper adds progressive summarization: when context exceeds a threshold, older messages are distilled into a compact summary by the LLM itself, preserving decisions and structure while reducing token count.

Configure via environment variables: `FIVE_MODEL`, `FIVE_BASE_URL`, `FIVE_LARGE_MODEL`, `FIVE_LARGE_URL`, `FIVE_MAX_TOKENS`.

---

## Trajectories

Every run produces one trajectory JSON file. It contains the full conversation: system prompt, every assistant response, every tool observation, the final outcome.

Trajectories are the only persistent state. They are gitignored. The runner commits them on exit as proof of work.

If you lose everything else, the trajectories let you reconstruct what happened.

---

## How to Make Changes

**Framework changes** — edit `src/four/`. The loop in `core.py` must stay 22 lines. New components go in their module (`chat_model.py`, `parse.py`, `env.py`). Test with a trivial spoke first.

**Spoke changes** — edit the system prompt. It is the control surface. One clear goal, explicit rules, example commands. Test with `--max-steps 10`.

**Runner changes** — add variant to `run-massive-implementation.sh` case statement, point to the spoke.

---

## Invariants

1. **The algebra never changes.** G → V1 → V2* → emit. Four functions, always.
2. **The system prompt drives behavior.** The loop is dumb. The prompt is everything.
3. **Small steps, frequent commits.** Each step is one bash command. Each commit is one logical unit.
4. **Trajectories are truth.** If it's not in a trajectory, it didn't happen.
5. **Additive evolution.** Never delete working code. Copy, enhance, commit.
6. **Separate creation from validation.** Stream grows. Consolidation compresses. Review validates. Never mix phases.
