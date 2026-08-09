# Four Algebra — Knowledge Space Guide

## What This System Is

A four-function algebra for autonomous agents that compose into pipelines. The loop is the evaluator.

```
invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[list[action]]
validate : V2  -- action → Result[observation | Exit]
emit     : IO  -- (messages, outcome) → Path
```

The loop: `(G → V1 → [V2, V2, ...])* → emit`

## Our Vision

**Commits grow out the history. PRs shrink it.**

We discovered that autonomous agents work best when they flow unimpeded — recording from latent space into a linear commit history. Gating mechanisms (PRs, tickets, stages) during the stream are process blockers, not facilitators. The agent ignores them.

The three-phase pipeline:
1. **Stream** — Unimpeded code flow. The agent writes code, commits incrementally, pushes to main. 500+ commits proven.
2. **PR Consolidation** — Post-hoc. Reads the commit history, finds semantic groups, creates PRs. Proven.
3. **Review** — Post-hoc. Reviews open PRs, runs tests, merges. Proven.

## How We Make Changes

**Additive only. Never mutate.**

- Clone existing code, diverge the clone
- New spokes go in `examples/spokes/`
- New generators go in `src/four/generators/`
- The algebra remains invariant — G, V1, V2, emit compose

## What Drives This Operation

The belief that software evolution is like DNA — additive, incremental, preservable. Each commit is a physical gate. The git timeline is durable memory. We don't summarize conversation history (lossy); we read the evidence (commits, diffs, code).

## The Servers

- `.157:8080` — Fast model (fast-qwen), short context, high throughput
- `.161:8081` — Large model (qwen), long context, heavy reasoning

## Running Experiments

```bash
# Stream pipeline (unimpeded)
bash run-massive-implementation.sh --goal "Add feature X" --variant stream

# PR consolidation (post-hoc)
bash run-massive-implementation.sh --goal "Group commits into semantic PRs" --variant pr-consolidation

# Review pipeline (post-hoc)
bash run-massive-implementation.sh --goal "Review open PRs" --variant pr-review

# Documentation pipeline (post-hoc)
bash run-massive-implementation.sh --goal "Document the codebase" --variant documentation
```

## Key Principles

1. **The beam records from latent space** — Don't block it with gates during recording
2. **Git history is the durable record** — Not LLM summaries, not state files
3. **Additive evolution** — Copy-enhance, never delete working code
4. **Functions compose** — G, V1, V2, emit are building blocks
5. **Evidence over plausibility** — Read what actually happened, not what an LLM thinks happened

## Common Pitfalls

- Don't add process blockers to the stream (stages, tickets, PR gates during recording)
- Don't squash merge — it destroys commit DNA
- Don't mutate existing spokes — clone and diverge
- Don't run full test suites during PR review — run targeted tests for changed modules
- The trajectories directory must exist — it's where the system saves evidence
