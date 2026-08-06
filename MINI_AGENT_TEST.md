# mini-swe-agent Test Results

## Summary

mini-swe-agent **works** against our local llama.cpp model at `http://192.168.1.157:8080/v1`.

Our four framework does NOT work the same way until we fix tool_call_id correlation.

## How We Got mini-swe-agent Working

### 1. Install mini-swe-agent
```bash
cd /Users/av4nda/Research/mini-swe-agent
pip install -e .
```

### 2. Create Config File
`~/.config/mini-swe-agent/config.yaml`:
```yaml
model:
  model_class: minisweagent.models.litellm_model.LitellmModel
  model_name: openai/sunny
  model_kwargs:
    base_url: http://192.168.1.157:8080/v1
    api_key: not-needed
    temperature: 0.3
    max_tokens: 4096
    drop_params: true
  cost_tracking: ignore_errors

agent:
  agent_class: minisweagent.agents.default.DefaultAgent
  mode: yolo
  step_limit: 10
  output_path: /tmp/mini_traj.json
```

### 3. Run Test
```bash
export MSWEA_MODEL_NAME="openai/sunny"
export MSWEA_CONFIGURED="1"

mini -c src/minisweagent/config/mini.yaml \
     -c ~/.config/mini-swe-agent/config.yaml \
     -t "List all .py files in /Users/av4nda/Research/five, then count lines in the largest one. Show the final count." \
     -o /tmp/mini_traj.json \
     -y --exit-immediately
```

## Results

| Metric | Value |
|--------|-------|
| Exit status | `Submitted` ✅ |
| API calls | 3 |
| Steps | 3 |

### Step-by-step execution:

1. **Step 2**: Model returns tool call → `find /Users/av4nda/Research/five -name "*.py" -type f`
2. **Step 4**: Model returns tool call → `wc -l` on all files to find largest
3. **Step 6**: Model returns tool call → `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

### Raw model responses (from trajectory):
```json
{
  "role": "assistant",
  "content": "I'll start by listing all .py files...",
  "tool_calls": [
    {
      "function": {
        "arguments": "{\"command\":\"find /Users/av4nda/Research/five -name \\\"*.py\\\" -type f\"}",
        "name": "bash"
      },
      "id": "MoQPLC6KYzgsOiUXzAWLleJ3cvdOMtTP",
      "type": "function"
    }
  ]
}
```

## The Gap: Why four doesn't work (INITIAL STATE)

When we call the same model with tool calls, our four framework gets:
- `tool_calls: None`
- Text output instead of structured tool calls
- Garbled markdown that looks like tool calls but isn't valid JSON

Direct `litellm.completion()` returns proper tool_calls.
Our `litellm_toolcall_invoke` wrapper does NOT return tool_calls.

**The question we need to answer:**
> What is the difference between:
> 1. mini-swe-agent calling `litellm.completion()` successfully with tool calls
> 2. our four.chat_model.LitellmModel._query() getting text instead of tool_calls
>
> Why does the same API call produce different responses?

## Root Cause Analysis

### Finding: Response Format Mismatch

The model at `:8080` uses a **custom Responses API format**:
```json
{
  "call_id": "call_xxx",
  "call_type": "execute_bash", 
  "args": {"command": "..."}
}
```

NOT the OpenAI-style tool_calls format.

### Finding: Tool Call ID Correlation

When we use `http_response_invoke()` with `toolcall_parse()`, it returns:
```json
[{"tool_call_id": "...", "name": "bash", "arguments": "{...}"}]
```

This IS the correct format! The issue was that **observations didn't include tool_call_id**.

When V2 returned `{role: "tool", content: "..."}` without `tool_call_id`, the model couldn't correlate results to calls, so it output text instead of structured tool calls on subsequent steps.

## The Fix

We modified two files:

### 1. `src/four/parse.py` - toolcall_parse() returns dicts with tool_call_id
```python
def toolcall_parse() -> Parse:
    def _parse(raw: str) -> Ok[list[dict]] | Err[str]:
        # Returns [{"command": "...", "tool_call_id": "xxx"}]
```

### 2. `src/four/core.py` - run() attaches tool_call_id to observations
```python
for action in actions.value:
    command = action["command"] if isinstance(action, dict) else action
    tool_call_id = action.get("tool_call_id") if isinstance(action, dict) else None
    result = V2(command)
    observation = result.value
    if tool_call_id:
        observation["tool_call_id"] = tool_call_id  # ← FIX
    messages.append(observation)
```

## Verification

With the fix, `--variant responses` now works:
- All G calls return proper JSON with tool_call_id
- All tool observations include tool_call_id  
- Tool results are properly correlated to calls

**Question remaining:**
> Why did mini-swe-agent work without this explicit tool_call_id passing?

Mini-swe-agent likely handles this differently - either:
1. The observation template includes tool_call_id automatically
2. The model receives messages in a different format that preserves correlation
3. There's something else we're missing about how the message flow works

We need to compare mini-swe-agent's `format_observation_messages()` to understand how it handles this.

## Files Compared

- mini-swe-agent: `/Users/av4nda/Research/mini-swe-agent/src/minisweagent/models/litellm_model.py`
- our four: `/Users/av4nda/Research/four-pr/src/four/chat_model.py`

Both use `litellm.completion()` with `tools=[BASH_TOOL]`.
Both seem to have identical tool definitions.
Both pass the same model kwargs.

**We need to identify the exact difference that causes this behavioral divergence.**
