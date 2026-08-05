#!/bin/bash
# Test Qwen3-Coder-Next Q8 (80B) on :8082 using Chat Completions API
# Run: ./examples/test_coder_next_q8_chat.sh
# Output: examples/log_coder_next_q8_chat.txt

set -e

cd "$(dirname "$0")/.."

LOG="examples/log_coder_next_q8_chat.txt"
BASE_URL="http://192.168.1.161:8082/v1"
PROMPT="List all .py files in the current directory, then count how many lines are in the largest one. Show the final count."

echo "Running: Coder-Next Q8 + Chat Completions on $BASE_URL"
echo "Prompt: $PROMPT"
echo "Log: $LOG"
echo "========================================"

python -c "
import os, sys, time

sys.path.insert(0, 'src')

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory

MODEL_ID = '/Users/kodep/models/bartowski/qwen3-coder-next-q4/Qwen_Qwen3-Coder-Next-Q4_K_M/Qwen_Qwen3-Coder-Next-Q4_K_M.gguf'
BASE_URL = '$BASE_URL'
PROMPT = '''$PROMPT'''

step_num = [0]

def debug_g(messages):
    step_num[0] += 1
    t0 = time.time()
    result = litellm_invoke(
        model=f'openai/{MODEL_ID}',
        base_url=BASE_URL,
        temperature=0.3,
        max_tokens=1024,
        api_key='dummy',
    )(messages)
    elapsed = time.time() - t0
    if isinstance(result, Ok):
        preview = result.value[:120].replace('\n', ' ')
        print(f'  [G step {step_num[0]}] ({elapsed:.1f}s) -> {preview}...')
    else:
        print(f'  [G step {step_num[0]}] ({elapsed:.1f}s) ERR: {result.error[:100]}')
    return result

g = debug_g
v1 = regex_parse()
v2 = local_env()
emit = save_trajectory('.')

system = (
    'You are a bash agent. You solve tasks by executing bash commands. '
    'Wrap each command in a \`\`\`bash ... \`\`\` block. '
    'When the task is fully done, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
)

path = run(G=g, V1=v1, V2=v2, emit=emit, system=system, prompt=PROMPT, max_steps=10)

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
