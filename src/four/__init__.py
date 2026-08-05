"""Four-function algebra for agents.

Four functions compose. The loop is the evaluator.

invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[list[action]]
validate : V2  -- action → Result[observation | Exit]
emit     : IO  -- (messages, outcome) → Path
"""

from .core import Err, Ok, Result, run, save_trajectory
from .core import Err, Ok, Result, run, save_trajectory
from .model import LitellmModel, AbortError
from .chat_model import BASH_TOOL, litellm_invoke, litellm_toolcall_invoke
from .response_model import BASH_TOOL_RESPONSE_API, http_response_invoke, litellm_response_invoke
from .parse import regex_parse, toolcall_parse, toolcall_response_parse
from .env import local_env

__all__ = [
    "Err",
    "Ok",
    "Result",
    "run",
    "save_trajectory",
    "LitellmModel",
    "AbortError",
    "BASH_TOOL",
    "BASH_TOOL_RESPONSE_API",
    "litellm_invoke",
    "litellm_toolcall_invoke",
    "litellm_response_invoke",
    "http_response_invoke",
    "regex_parse",
    "toolcall_parse",
    "toolcall_response_parse",
    "local_env",
]
