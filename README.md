# four

Four functions compose. The loop is the evaluator.

An agent that runs bash commands. A generator that produces notebooks. A tool that writes Python code. Same loop. Different functions.

```python
from four import run, litellm_invoke, regex_parse, local_env, save_trajectory

run(
    G=litellm_invoke("anthropic/claude-sonnet-4-5-20250929"),
    V1=regex_parse(),
    V2=local_env(),
    emit=save_trajectory(),
    system="You are a helpful assistant that executes bash commands.",
    prompt="Find all Python files in /tmp and count lines in each",
    max_steps=50,
)
```

## The algebra

```
invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[list[action]]
validate : V2  -- action → Result[observation | Exit]
emit     : IO  -- (messages, outcome) → Path
```

The loop chains them: `(G → V1 → [V2, V2, ...])* → emit`

Each step: G queries the model, V1 extracts all actions, V2 executes each one. If V1 fails, the error becomes a user message and the loop continues — the model sees its mistake and self-corrects on the next turn. Four functions compose.

## The loop

The entire evaluator is 22 lines:

```python
def run(G, V1, V2, emit, system, prompt, max_steps=100, max_format_errors=3):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    consecutive_format_errors = 0

    for step in range(max_steps):
        raw = G(messages)
        if isinstance(raw, Err):
            return emit(messages, f"model_error: {raw.error}")

        actions = V1(raw.value)
        if isinstance(actions, Err):
            consecutive_format_errors += 1
            if 0 < max_format_errors <= consecutive_format_errors:
                return emit(messages, f"repeated_format_error: {actions.error}")
            messages.append({"role": "user", "content": f"Format error: {actions.error}..."})
            continue

        consecutive_format_errors = 0
        for action in actions.value:
            result = V2(action)
            if isinstance(result, Err):
                return emit(messages, result.error)
            messages.append(result.value)

    return emit(messages, "max_steps_reached")
```

That's it. No config files. No YAML. No SDK. No Pydantic models. No Jinja2 templates baked into the code. Just four functions that take and return well-typed values, chained in a loop.

## What it replaces

| mini-swe-agent | four |
|---|---|
| YAML config with 40+ parameters | Four function arguments |
| Pydantic model configs | Plain functions |
| Jinja2 templates in config | Templates passed as strings |
| FormatError + InterruptAgentFlow hierarchy | `Ok` \| `Err` |
| Inner retry loop for format errors | Format error as user message, outer loop continues |
| 1000+ lines of boilerplate | 22-line loop |

Same capability. Different shape.

## Components

**G — invoke.** Queries the LLM. Returns `Ok(text)` or `Err(reason)`.
- `litellm_invoke()` — plain text with markdown code blocks
- `litellm_toolcall_invoke()` — structured tool calls
- `retry_invoke(fn)` — wraps any G with exponential backoff retry

**V1 — parse.** Extracts actions from raw output. Returns `Ok(list[command])` or `Err(reason)`.
- `regex_parse()` — extracts ```mswea_bash_command, ```bash, or ```sh blocks (returns all matches)
- `toolcall_parse()` — parses JSON tool call payloads (returns all bash commands)

**V2 — validate.** Executes each action. Returns `Ok(observation)` or `Err(exit)`.
- `local_env()` — subprocess execution with output truncation and exit signal detection

**emit — IO.** Saves the trajectory. Returns `Path`.
- `save_trajectory()` — JSON files with outcome and full message history

## Extending

Every component is swappable. The loop doesn't care:

```python
# Tool-calling instead of regex
G=litellm_toolcall_invoke("openai/gpt-4o"),
V1=toolcall_parse(),

# Container execution instead of local
V2=docker_env(image="python:3.12"),

# Retry on transient errors
G=retry_invoke(litellm_invoke("openai/gpt-4o")),

# Abort after 2 format errors instead of 3
max_format_errors=2,
```

## Generator examples

### Generate a four-style agent
```bash
python -m four.generators.four_agent.generate \
    --prompt "Create a bash agent with chat variant" \
    --output-dir ./agents
```

### Generate a pipeline spoke (multi-stage workflow)
```bash
python -m four.generators.pipeline_spoke.generate \
    --prompt "Create an essay pipeline with outline, draft, review, revision stages" \
    --output-dir ./spokes
```

### Generate a bash pipeline spoke
```bash
python -m four.generators.bash_pipeline_spoke.generate \
    --prompt "Create a bash pipeline that processes text through multiple stages" \
    --output-dir ./spokes
```

### Generate an idea-forge template
```bash
python -m four.generators.idea_forge_template.generate \
    --prompt "Create a python-research template with pytest and GitHub workflows" \
    --output-dir ./templates
```

### Generate an orchestrator system
```bash
python -m four.generators.orchestrator.generate \
    --prompt "Create a distributed orchestrator with spoke registration and auto-dispatch" \
    --output-dir ./orchestrators
```

## The same algebra, different domains

The four-function loop generates notebooks, agents, Python code, CLI tools, and bash scripts. The only difference is what V2 validates and what emit produces:

| Domain | V2 validates | emit produces |
|---|---|
| [Notebooks](https://github.com/belarusian/letsplot-analysis) | AST + chart execution | `.ipynb` with embedded PNGs |
| Agents | bash execution | JSON trajectory |
| Python code | type checking | `.py` files |
| CLI tools | compilation | binary + man page |
| Bash pipelines | bash syntax + stage chain | `.sh` scripts |

The loop doesn't know what it's evaluating. It only chains `Result` types.

## Generators: four composed with four

The generators framework is `four` composed with `four`:

```
G (invoke) → V1 (parse) → V2 (validate) → G' (fix) → emit
```

Where **G' = G** — the fix node is just another model invocation with error context.

This creates a two-tier generation loop:
1. **Outer loop (rounds)**: G → V1 → V2 → emit (or retry with feedback)
2. **Inner loop (fixes)**: G' (same as G) with error context → V1 → V2

Each generator is a specific instance of this pattern:
- `four_agent`: generates Python agent specs using `four.core.run`
- `pipeline_spoke`: generates multi-stage workflow spokes using `four.core.run`
- `bash_pipeline_spoke`: generates bash pipeline scripts using curl + jq
- `idea_forge_template`: generates idea-forge templates
- `orchestrator`: generates orchestrator systems

All use the same `generation_loop` function from `_loop.py`, differentiated only by their five functions:
- `invoke`: Context → Result[raw]
- `parse`: raw → Result[Spec]
- `validate`: Spec → Result[Artifact]
- `fix`: (Spec, error, ctx) → Spec | None
- `emit`: (Spec, Artifact, rounds, fixes, version, prompt) → Result[Path]

The generators don't implement the loop — they implement the functions that *feed into* the loop.

## Why four?

Four is the minimum. Remove any one and the loop breaks:

- No **G** → nothing to evaluate
- No **V1** → can't extract actions from raw text
- No **V2** → can't execute or observe
- No **emit** → can't persist results

Format recovery is built into the loop — no separate function needed.

## Philosophy

The framework doesn't call itself category theory. It calls itself algebra. Four functions compose. The loop is the evaluator.

#agenticcoding #functional-programming #python #llm #agents #monads
