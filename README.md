# five

Five functions compose. The loop is the evaluator.

```python
from five import run, litellm_invoke, regex_parse, local_env, format_fix, save_trajectory

system = """You are a helpful assistant that executes bash commands.
Respond with exactly one command in ```mswea_bash_command blocks."""

run(
    G=litellm_invoke("anthropic/claude-sonnet-4-5-20250929"),
    V1=regex_parse(),
    V2=local_env(),
    G_prime=format_fix,
    emit=save_trajectory(),
    system=system,
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
