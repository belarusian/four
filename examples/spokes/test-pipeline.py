#!/usr/bin/env python3
"""
Autonomous pipeline: test-pipeline

just a test

This spoke implements an autonomous development loop:
- Takes a high-level goal as input
- Runs multiple stages, each with multiple LLM turns
- Uses bash tool to write code, run git, execute commands
- Validates each step with bash commands
- Loops until goal achieved or max_steps reached
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory


def slugify(text: str) -> str:
    """Convert text to a filesystem-friendly slug."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    return text


PIPELINE = {
    "name": "test-pipeline",
    "description": "just a test",
    "stages": {
        "test": {
            "name": "test",
            "description": """just a test""",
            "max_turns": 10,
            "validation_command": """true""",
        },
    },
    "output_dir": "output",
}


def execute_turn(turn_num: int, stage_name: str, stage_config: dict[str, Any], input_text: str) -> tuple[str, Path]:
    """Execute a single turn using four.core.run() with bash tool."""
    model_id = "granite4.1:8b"
    MODEL_ID = os.getenv("FIVE_MODEL", model_id)
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1")
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))

    system_prompt = f"""You are an autonomous development agent.
    
Stage: stage_config['description']
Turn: {"turn_num"}/10

Task: Make progress toward completing this stage.
Use bash tool to:
- Write/modify files in the codebase
- Run git commands (add, commit, status)
- Execute build/test commands

Output your final answer in the response body (not as a tool call).
"""

    def invoke(messages):
        result = litellm_invoke(
            model=f"openai/{MODEL_ID}",
            base_url=BASE_URL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            api_key="dummy",
        )(messages)
        return result

    def parse(raw):
        return regex_parse()(raw)

    def validate(action):
        return local_env()(action)

    def emit(messages, outcome):
        return Path("trajectory.json")

    path = run(
        G=invoke,
        V1=parse,
        V2=validate,
        emit=emit,
        system=system_prompt,
        prompt=input_text,
        max_steps=10,
    )

    trajectory = Path("trajectory.json")
    if trajectory.exists():
        data = json.loads(trajectory.read_text())
        for msg in reversed(data.get("messages", [])):
            if msg.get("role") == "assistant":
                return msg.get("content", ""), path

    return "Turn output not found", path


def execute_stage(stage_name: str, stage_config: dict[str, Any], input_text: str) -> str:
    """Execute all turns in a stage."""
    output_dir = Path(PIPELINE["output_dir"])
    stage_dir = output_dir / slugify(input_text) / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    current_input = input_text
    final_output = ""

    for turn_num in range(1, stage_config["max_turns"] + 1):
        output_file = stage_dir / f"turn_{turn_num}.md"
        output, _ = execute_turn(turn_num, stage_name, stage_config, current_input)

        output_file.write_text(output)
        print(f"  Turn {turn_num}/{stage_config['max_turns']}: {output_file}")

        final_output = output

        if turn_num < stage_config["max_turns"]:
            current_input = output

    return final_output


def validate_stage_output(stage_name: str, stage_config: dict[str, Any], output: str) -> bool:
    """Validate stage output using bash command."""
    cmd = stage_config.get("validation_command", "true")
    if not cmd or cmd == "true":
        return True

    try:
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        print(f"  Validation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=PIPELINE["description"])
    parser.add_argument("--topic", required=True, help="Pipeline topic/goal")
    parser.add_argument("--endpoint", default=os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1"))
    args = parser.parse_args()

    topic = args.topic
    slug = slugify(topic)
    output_dir = Path(PIPELINE["output_dir"]) / slug

    print(f"Goal: {topic}")
    print(f"Slug: {slug}")
    print(f"Output dir: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "topic.txt").write_text(topic)

    current_input = topic
    stage_names = list(PIPELINE["stages"].keys())

    for i, stage_name in enumerate(stage_names):
        stage_config = PIPELINE["stages"][stage_name]
        output_file = output_dir / f"{stage_name}.md"

        print(f"Stage {i+1}/{len(stage_names)}: {stage_name}")
        print("-" * 40)

        t0 = time.time()
        output = execute_stage(stage_name, stage_config, current_input)
        elapsed = time.time() - t0

        output_file.write_text(output)
        print(f"  Output: {output_file} ({elapsed:.1f}s)")

        if validate_stage_output(stage_name, stage_config, output):
            print(f"  Validation: PASSED")
        else:
            print(f"  Validation: FAILED")

        print()
        current_input = output

    final_file = output_dir / "final.md"
    final_file.write_text(current_input)

    print("=" * 40)
    print("PIPELINE COMPLETE")
    print("=" * 40)
    print(f"Goal: {topic}")
    print()
    for stage_name in stage_names:
        file_path = output_dir / f"{stage_name}.md"
        print(f"{stage_name.capitalize()}: {file_path}")
    print()


if __name__ == "__main__":
    main()
