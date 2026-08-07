"""Context builder for the four-agent generator.

Injects four-framework knowledge, available transports, and best practices
into the model's system prompt.
"""

from __future__ import annotations

from four.generators._types import DomainSection, GenerationContext

_FOUR_FRAMEWORK = """\
## Four-Framework API

The four-framework implements a four-function algebra for agents:

    invoke   : G   -- messages -> Result[raw]
    parse    : V1  -- raw -> Result[list[action]]
    validate : V2  -- action -> Result[observation | Exit]
    emit     : IO  -- (messages, outcome) -> Path

The loop: (G -> V1 -> [V2, V2, ...])* -> emit

Format errors are appended as user messages with no inner retry loop.
Consecutive format errors are tracked and abort after N failures.

### Core API

```python
from four.core import run, Ok, Err

path = run(
    G=invoke_fn,       # messages -> Result[raw]
    V1=parse_fn,       # raw -> Result[list[action]]
    V2=validate_fn,    # action -> Result[observation | Exit]
    emit=emit_fn,      # (messages, outcome) -> Path
    system=system_prompt,
    prompt=user_prompt,
    max_steps=10,
)
```

### Invoke Transports (G)

#### litellm_invoke (Chat Completions)
```python
from four.chat_model import litellm_invoke

G = litellm_invoke(
    model="openai/{model_id}",
    base_url=base_url,
    temperature=0.3,
    max_tokens=1024,
    api_key="dummy",
)
```

#### litellm_toolcall_invoke (Chat Completions with tool calls)
```python
from four.chat_model import litellm_toolcall_invoke

tools = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"],
            },
        },
    },
]

G = litellm_toolcall_invoke(
    model="openai/{model_id}",
    base_url=base_url,
    temperature=0.3,
    max_tokens=1024,
    api_key="dummy",
    tools=tools,
)
```

#### http_response_invoke (OpenAI Responses API)
```python
from four.response_model import http_response_invoke

G = http_response_invoke(
    base_url=base_url,
    model=model_id,
    api_key="dummy",
    max_output_tokens=1024,
)
```

### Parse Functions (V1)

#### regex_parse (extract bash blocks from text)
```python
from four.parse import regex_parse

V1 = regex_parse()
```

#### toolcall_parse (extract tool calls from structured response)
```python
from four.parse import toolcall_parse

V1 = toolcall_parse()
```

### Validate Functions (V2)

#### local_env (execute bash commands locally)
```python
from four.env import local_env

V2 = local_env()
```

### Emit Functions

#### save_trajectory (save to JSON)
```python
from four.core import save_trajectory

emit = save_trajectory(output_dir="trajectories")
```

#### Custom emit
```python
def emit(messages, outcome):
    import json
    from pathlib import Path
    path = Path("trajectory.json")
    path.write_text(json.dumps({"outcome": outcome, "messages": messages}, indent=2))
    return path
```
"""

_FOUR_PATTERNS = """\
## Agent Code Patterns

### Complete Chat Variant Agent
```python
#!/usr/bin/env python3
import argparse
import os
import sys
import time

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--endpoint", default=os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1"))
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "fast-qwen")
    BASE_URL = args.endpoint
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))
    MAX_STEPS = int(os.getenv("FIVE_MAX_STEPS", "10"))

    step_num = [0]

    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = litellm_invoke(
            model=f"openai/{MODEL_ID}",
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
        )(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = result.value[:120].replace("\\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result

    v1 = regex_parse()
    v2 = local_env()

    system = (
        "You are a bash agent. You solve tasks by executing bash commands. "
        "Wrap each command in a ```bash ... ``` block. "
        "When the task is fully done, respond with plain text instead of a code block."
    )

    prompt = args.prompt or "List all .py files in the current directory."

    def emit(messages, outcome):
        import json
        from pathlib import Path
        path = Path("trajectory.json")
        path.write_text(json.dumps({"outcome": outcome, "messages": messages}, indent=2))
        return path

    path = run(G=debug_g, V1=v1, V2=v2, emit=emit, system=system, prompt=prompt, max_steps=MAX_STEPS)
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")

if __name__ == "__main__":
    main()
```

### Complete Toolcall Variant Agent
```python
#!/usr/bin/env python3
import argparse
import os
import sys
import time

from four.core import run, Ok, Err
from four.chat_model import litellm_toolcall_invoke
from four.parse import toolcall_parse
from four.env import local_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--endpoint", default=os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1"))
    args = parser.parse_args()

    MODEL_ID = os.getenv("FIVE_MODEL", "fast-qwen")
    BASE_URL = args.endpoint
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))
    MAX_STEPS = int(os.getenv("FIVE_MAX_STEPS", "10"))

    step_num = [0]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"}
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    def debug_g(messages):
        step_num[0] += 1
        t0 = time.time()
        result = litellm_toolcall_invoke(
            model=f"openai/{MODEL_ID}",
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
            tools=tools,
        )(messages)
        elapsed = time.time() - t0
        if isinstance(result, Ok):
            preview = result.value[:120].replace("\\n", " ")
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...")
        else:
            print(f"  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}")
        return result

    v1 = toolcall_parse()
    v2 = local_env()

    prompt = args.prompt or "List all .py files in the current directory."

    def emit(messages, outcome):
        import json
        from pathlib import Path
        path = Path("trajectory.json")
        path.write_text(json.dumps({"outcome": outcome, "messages": messages}, indent=2))
        return path

    path = run(G=debug_g, V1=v1, V2=v2, emit=emit, system="", prompt=prompt, max_steps=MAX_STEPS)
    print(f"Trajectory saved to: {path}")
    print(f"Total G calls: {step_num[0]}")

if __name__ == "__main__":
    main()
```
"""

_FOUR_PRINCIPLES = """\
## Agent Authoring Principles

### Structure
1. Shebang and imports
2. Argument parser for --prompt, --endpoint
3. Environment configuration (MODEL_ID, BASE_URL, MAX_TOKENS, MAX_STEPS)
4. Debug G function (wraps the transport with timing and logging)
5. V1 (parse) and V2 (validate) setup
6. System prompt definition
7. Emit function definition
8. run() call

### Transport selection
- Use `chat` variant when the model responds with markdown code blocks
- Use `toolcall` variant when the model supports function calling natively
- Use `responses` variant when targeting OpenAI Responses API endpoints

### System prompt rules
- For chat variant: instruct the model to wrap commands in ```bash blocks
- For toolcall variant: system prompt can be empty (tool definition is implicit)
- For responses variant: same as chat variant

### Error handling
- Use Ok/Err result types throughout
- Debug G should log timing and preview of responses
- Emit should save trajectory as JSON for inspection

### Code quality
- Use argparse for CLI arguments
- Use os.getenv() for configuration with sensible defaults
- Use f-strings for formatted output
- Keep the code under 100 lines
"""


def _discover_available_packages() -> str:
    """List installed Python packages relevant to four agents."""
    try:
        from importlib.metadata import distributions
        pkgs = sorted(
            {d.metadata["Name"] for d in distributions() if d.metadata["Name"]},
            key=str.lower,
        )
        return ", ".join(pkgs)
    except Exception:
        return "four, litellm, openai"


def build_four_agent_context(
    prompt: str | None = None,
) -> GenerationContext:
    """Build context for the four-agent generator."""
    return GenerationContext(
        domain_context=(
            DomainSection("Four-Framework API", _FOUR_FRAMEWORK),
            DomainSection("Agent Code Patterns", _FOUR_PATTERNS),
            DomainSection("Authoring Principles", _FOUR_PRINCIPLES),
        ),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
        default_task=(
            "Generate a four-style agent runner that can execute bash commands "
            "using the four-function algebra (G -> V1 -> V2* -> emit)."
        ),
    )
