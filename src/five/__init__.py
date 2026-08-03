"""Five-function algebra for agents.

Five functions compose. The loop is the evaluator.

invoke   : G   -- messages → Result[raw]
parse    : V1  -- raw → Result[action]
validate : V2  -- action → Result[observation | Exit]
fix      : G'  -- (error, messages) → message | None
emit     : IO  -- (messages, outcome) → Path
"""

from .core import Err, Ok, Result, run, save_trajectory
from .model import BASH_TOOL, litellm_invoke, litellm_toolcall_invoke
from .parse import regex_parse, toolcall_parse
from .env import format_fix, local_env

__all__ = [
    "Err",
    "Ok",
    "Result",
    "run",
    "save_trajectory",
    "BASH_TOOL",
    "litellm_invoke",
    "litellm_toolcall_invoke",
    "regex_parse",
    "toolcall_parse",
    "local_env",
    "format_fix",
]
