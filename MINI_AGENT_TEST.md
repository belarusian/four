# mini-swe-agent Test Results

## Summary

mini-swe-agent **works** against our local llama.cpp model at `http://192.168.1.157:8080/v1`.

Our four framework does NOT work the same way.

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

## The Gap: Why four doesn't work

When we call the same model with tool calls, our four framework gets:
- `tool_calls: None`
- Text output instead of structured tool calls
- Garbled markdown that looks like tool calls but isn't valid JSON

Direct `litellm.completion()` returns proper tool_calls.
Our `litellm_toolcall_invoke` wrapper does NOT return tool_calls.

**The question we need to answer:**
> What is the difference between:
> 1. mini-swe-agent calling `litellm.completion()` successfully with tool calls
> 2. our `four.chat_model.LitellmModel._query()` getting text instead of tool_calls
>
> Why does the same API call produce different responses?

## Files Compared

- mini-swe-agent: `/Users/av4nda/Research/mini-swe-agent/src/minisweagent/models/litellm_model.py`
- our four: `/Users/av4nda/Research/four-pr/src/four/chat_model.py`

Both use `litellm.completion()` with `tools=[BASH_TOOL]`.
Both seem to have identical tool definitions.
Both pass the same model kwargs.

**We need to identify the exact difference that causes this behavioral divergence.**
