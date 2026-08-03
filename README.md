# five

Five functions compose. The loop is the evaluator.

An agent that runs bash commands. A generator that produces notebooks. A tool that writes Python code. Same loop. Different functions.

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

Each function is first-class. Swap the model. Change the parser. Run commands in a container. Save trajectories differently. The loop doesn't know what it's evaluating — it only knows that every phase returns `Ok(value)` or `Err(reason)`. Errors flow through the pipeline. No try/catch. No state machines. Five functions compose.

## The loop

The entire evaluator is 25 lines:

```python
def run(G, V1, V2, G_prime, emit, system, prompt, max_steps=100):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]

    for step in range(max_steps):
        raw = G(messages)
        if isinstance(raw, Err):
            return emit(messages, f"model_error: {raw.error}")

        action = V1(raw.value)
        if isinstance(action, Err):
            fix = G_prime(action.error, messages)
            if fix:
                messages.append(fix)
                continue
            return emit(messages, f"format_error: {action.error}")

        result = V2(action.value)
        if isinstance(result, Err):
            return emit(messages, result.error)

        messages.append(result.value)

    return emit(messages, "max_steps_reached")
```

That's it. No config files. No YAML. No SDK. No Pydantic models. No Jinja2 templates baked into the code. Just five functions that take and return well-typed values, chained in a loop.

## What it replaces

| mini-swe-agent | five |
|---|---|
| YAML config with 40+ parameters | Five function arguments |
| Pydantic model configs | Plain functions |
| Jinja2 templates in config | Templates passed as strings |
| FormatError + InterruptAgentFlow hierarchy | `Ok` \| `Err` |
| 1000+ lines of boilerplate | 25-line loop |

Same capability. Different shape.

## Components

**G — invoke.** Queries the LLM. Returns `Ok(text)` or `Err(reason)`.
- `litellm_invoke()` — plain text with markdown code blocks
- `litellm_toolcall_invoke()` — structured tool calls

**V1 — parse.** Extracts the action from raw output. Returns `Ok(command)` or `Err(reason)`.
- `regex_parse()` — extracts ```mswea_bash_command blocks
- `toolcall_parse()` — parses JSON tool call payloads

**V2 — validate.** Executes the action. Returns `Ok(observation)` or `Err(exit)`.
- `local_env()` — subprocess execution with output truncation and exit signal detection

**G' — fix.** Formats errors as retry messages. Returns `message` to re-prompt, or `None` to stop.
- `format_fix()` — wraps parse errors in user messages

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

# Stop on format error instead of retrying
G_prime=lambda err, msgs: None,
```

## The same algebra, different domains

The five-function loop generates notebooks, agents, Python code, and CLI tools. The only difference is what V2 validates and what emit produces:

| Domain | V2 validates | emit produces |
|---|---|---|
| [Notebooks](https://github.com/belarusian/letsplot-analysis) | AST + chart execution | `.ipynb` with embedded PNGs |
| Agents | bash execution | JSON trajectory |
| Python code | type checking | `.py` files |
| CLI tools | compilation | binary + man page |

The loop doesn't know what it's evaluating. It only chains `Result` types.

## Why five?

Five is the minimum. Remove any one and the loop breaks:

- No **G** → nothing to evaluate
- No **V1** → can't extract actions from raw text
- No **V2** → can't execute or observe
- No **G'** → can't recover from format errors
- No **emit** → can't persist results

Add a sixth and it's redundant — the loop already closes.

## Philosophy

The framework doesn't call itself category theory. It calls itself algebra. Five functions compose. The loop is the evaluator.

#agenticcoding #functional-programming #python #llm #agents #monads
