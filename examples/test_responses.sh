#!/bin/bash
# Test Qwen3.6-27B on any endpoint using Responses API with tool calls
# Run: FIVE_BASE_URL=http://192.168.1.XXX:808X/v1 ./examples/test_responses.sh
# Output: examples/log_responses.txt

set -e

cd "$(dirname "$0")/.."

LOG="examples/log_responses.txt"
BASE_URL="${FIVE_BASE_URL:-http://192.168.1.157:8080/v1}"
PROMPT="List all .py files in the current directory, then count how many lines are in the largest one. Show the final count."

echo "Running: Responses API on $BASE_URL"
echo "Prompt: $PROMPT"
echo "Log: $LOG"
echo "========================================"

python -c "
import os, sys, time

sys.path.insert(0, 'src')

from four.core import run, Ok, Err
from four.response_model import http_response_invoke
from four.parse import toolcall_parse
from four.env import local_env

MODEL_ID = 'C:\\\\Users\\\\kodep\\\\models\\\\unsloth\\\\Qwen3.6-27B-MTP-GGUF\\\\Qwen3.6-27B-UD-Q4_K_XL.gguf'
BASE_URL = '${BASE_URL}'
PROMPT = '''$PROMPT'''

step_num = [0]

def debug_g(messages):
    step_num[0] += 1
    t0 = time.time()
    result = http_response_invoke(
        base_url=BASE_URL,
        model=MODEL_ID,
        api_key='dummy',
        max_output_tokens=1024,
    )(messages)
    elapsed = time.time() - t0
    if isinstance(result, Ok):
        preview = result.value[:120].replace('\n', ' ')
        print(f'  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...')
    else:
        print(f'  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}')
    return result

g = debug_g
v1 = toolcall_parse()
v2 = local_env()

system = (
    'You are a bash agent. You solve tasks by executing bash commands. '
    'Wrap each command in a \`\`\`bash ... \`\`\` block. '
    'When the task is fully done, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
)

path = run(G=g, V1=v1, V2=v2, emit=lambda m, o: open('/dev/null', 'w'), system=system, prompt=PROMPT, max_steps=10)

import json
data = json.loads(path.read_text())
print(f'\nOutcome: {data[\"outcome\"]}')
print(f'G calls: {step_num[0]}')
for i, m in enumerate(data['messages']):
    role = m['role']
    content = m.get('content', '')
    if isinstance(content, str):
        preview = content[:200].replace('\n', ' ')
    else:
        preview = str(content)[:200]
    print(f'  {i}: [{role}] {preview[:120]}')
" 2>&1 | tee "$LOG"

echo ""
echo "Done. Log saved to $LOG"
