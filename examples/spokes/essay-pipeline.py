#!/usr/bin/env python3
"""
Pipeline spoke: essay-pipeline

Multi-stage essay writing pipeline

This spoke implements a multi-stage pipeline where each stage
uses a specific model and passes its output to the next stage.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add four to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "AI" / "four"))

from four.core import run, Ok, Err
from four.chat_model import litellm_invoke
from four.parse import regex_parse
from four.env import local_env
from four.core import save_trajectory

# Pipeline configuration
PIPELINE = {
    "name": "essay-pipeline",
    "description": "Multi-stage essay writing pipeline",
    "stages": {
        "outline": {"model": "granite4.1:3b", "prompt": """Pipeline stage: outline"""},
        "draft": {"model": "granite4.1:3b", "prompt": """Pipeline stage: draft"""},
        "review": {"model": "granite4.1:8b", "prompt": """Pipeline stage: review"""},
        "revision": {"model": "granite4.1:3b", "prompt": """Pipeline stage: revision"""},
    },
    "output_dir": "output/{topic-slug}",
}


def slugify(text: str) -> str:
    """Convert text to a filesystem-friendly slug."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    return text


def execute_stage(stage_name: str, stage_config: dict[str, Any], input_text: str) -> str:
    """Execute a single pipeline stage."""
    model_name = stage_config["model"]
    model_id = model_name.split(":")[0] if ":" in model_name else model_name
    MODEL_ID = os.getenv("FIVE_MODEL", model_id)
    BASE_URL = os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1")
    MAX_TOKENS = int(os.getenv("FIVE_MAX_TOKENS", "1024"))

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

    system_prompt = stage_config["prompt"]
    prompt = input_text

    path = run(
        G=invoke,
        V1=parse,
        V2=validate,
        emit=emit,
        system=system_prompt,
        prompt=prompt,
        max_steps=10,
    )

    # Read the trajectory to get the output
    trajectory = Path("trajectory.json")
    if trajectory.exists():
        import json
        data = json.loads(trajectory.read_text())
        # Get the final assistant message
        for msg in reversed(data.get("messages", [])):
            if msg.get("role") == "assistant":
                return msg.get("content", "")

    return "Stage output not found"


def main():
    parser = argparse.ArgumentParser(description=PIPELINE["description"])
    parser.add_argument("--topic", required=True, help="Pipeline topic/input")
    parser.add_argument("--endpoint", default=os.getenv("FIVE_BASE_URL", "http://localhost:8080/v1"))
    args = parser.parse_args()

    topic = args.topic
    slug = slugify(topic)
    output_dir = Path(PIPELINE["output_dir"]) / slug

    print(f"Topic: {topic}")
    print(f"Slug: {slug}")
    print(f"Output dir: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the topic
    (output_dir / "topic.txt").write_text(topic)

    # Execute pipeline stages
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
        print()

        current_input = output

    # Save final output
    final_file = output_dir / "essay.md"
    final_file.write_text(current_input)

    print("=" * 40)
    print("COMPLETE")
    print("=" * 40)
    print(f"Topic: {topic}")
    print()
    for stage_name in stage_names:
        file_path = output_dir / f"{stage_name}.md"
        print(f"{stage_name.capitalize()}: {file_path}")
    print()


if __name__ == "__main__":
    main()
