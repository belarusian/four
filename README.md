# five

Five functions compose. The loop is the evaluator.

```python
from five import run, litellm_invoke, regex_parse, local_env, format_fix, save_trajectory

run(
    G=litellm_invoke("anthropic/claude-sonnet-4-5-20250929"),
    V1=regex_parse(),
    V2=local_env(),
    G_prime=format_fix,
    emit=save_trajectory(),
    system="You are a helpful assistant that executes bash commands.",
    prompt="Find all Python files in /tmp and count lines in each",
    max_steps=50,
)
```

## The algebra

```
invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[action]
validate : V2  -- action → Result[observation | Exit]
fix      : G'  -- (error, messages) → message | None
emit     : IO  -- (messages, outcome) → Path
```

The loop chains them: `G → V1 → V2 → (G' → G)* → emit`

Each function is first-class — swap the model, change the parser, run commands in a container, save trajectories differently. Same loop. Different functions.

## Why five?

The same algebra generates notebooks, Python code, CLI tools, and agents. The only difference is what V2 validates and what emit produces. The loop doesn't know what it's evaluating.

See [letsplot-analysis](https://github.com/belarusian/letsplot-analysis) for the generator variant.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  run(G, V1, V2, G', emit, system, prompt)       │
│                                                  │
│  messages = [system, prompt]                     │
│  while steps < max_steps:                        │
│    raw = G(messages)             # LLM query     │
│    action = V1(raw)            # extract command │
│    obs = V2(action)            # execute it      │
│    messages += obs                               │
│  return emit(messages, outcome)                  │
└─────────────────────────────────────────────────┘
```

## Components

### G — invoke
Queries the LLM. Two implementations:
- `litellm_invoke()` — plain text responses with markdown code blocks
- `litellm_toolcall_invoke()` — structured tool calls (bash function)

### V1 — parse
Extracts actions from raw LLM output. Two implementations:
- `regex_parse()` — extracts ````mswea_bash_command` blocks`
- `toolcall_parse()` — parses JSON tool call payloads

### V2 — validate
Executes the action and returns an observation. One implementation:
- `local_env()` — runs `subprocess.run()`, detects exit signals, truncates output

### G' — fix
Formats parse errors as retry messages. One implementation:
- `format_fix(error, messages)` — wraps error in a user message for re-prompting

### emit — IO
Saves the trajectory. One implementation:
- `save_trajectory()` — writes JSON files with outcome and full message history

## Extending

Every component is swappable. Examples:

```python
# Use tool-calling instead of regex
G=litellm_toolcall_invoke("openai/gpt-4o"),
V1=toolcall_parse(),

# Run commands in a container
V2=docker_env(image="python:3.12", volumes={"/tmp": "/workspace"}),

# Save as LLMF format for benchmarking
emit=save_trajectory("/results", fmt="llmf"),

# Stop on format error instead of retrying
G_prime=lambda err, msgs: None,  # return None = stop
```

## Philosophy

The framework doesn't call itself category theory. It calls itself algebra. Five functions compose. The loop is the evaluator. No config files. No YAML. No SDK. Just functions that take and return well-typed values, chained in a loop that doesn't care what it's evaluating.

#agenticcoding #functional-programming #python #llm #agents #monads
