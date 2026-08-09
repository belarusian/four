"""Four-function algebra for agents.

Four functions compose. The loop is the evaluator.

invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[list[action]]
validate : V2  -- action → Result[observation | Exit]
emit     : IO  -- (messages, outcome) → Path
"""

from .core import Err, Ok, Result, run, save_trajectory, AbortError, retry_invoke, Invoke, Validate
from .chat_model import BASH_TOOL, litellm_invoke, litellm_toolcall_invoke, context_aware_invoke, summarizing_invoke
from .response_model import BASH_TOOL_RESPONSE_API, http_response_invoke
from .parse import regex_parse, toolcall_parse, toolcall_response_parse
from .env import local_env, local_env_response, pr_gate_env
from .super_spoke import (
    super_env,
    super_invoke,
    super_response_invoke,
    super_spoke_chat,
    super_spoke_responses,
)

__all__ = [
    "Err",
    "Ok",
    "Result",
    "run",
    "save_trajectory",
    "AbortError",
    "retry_invoke",
    "Invoke",
    "Validate",
    "BASH_TOOL",
    "BASH_TOOL_RESPONSE_API",
    "litellm_invoke",
    "litellm_toolcall_invoke",
    "context_aware_invoke",
    "summarizing_invoke",
    "http_response_invoke",
    "regex_parse",
    "toolcall_parse",
    "toolcall_response_parse",
    "local_env",
    "local_env_response",
    "pr_gate_env",
    "super_env",
    "super_invoke",
    "super_response_invoke",
    "super_spoke_chat",
    "super_spoke_responses",
]
