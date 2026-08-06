"""Super Spoke demo - Most powerful spoke variant.

Usage:
    python examples/super_spoke_demo.py "List all .py files in the current directory, then count how many lines are in the largest one. Show the final count."
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from four.super_spoke import super_spoke_responses


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/super_spoke_demo.py <prompt>")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    # Use the Responses API with deep-qwen model
    # This is the most powerful spoke variant:
    # - Responses API (most advanced reasoning)
    # - Auto-retry on transient failures (15 attempts)
    # - Large output limits (100k chars, 5 min timeout)
    # - Tool calls for structured execution
    
    print(f"Super Spoke (Responses API + deep-qwen)")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    # Get the runner function
    runner = super_spoke_responses(
        model="deep-qwen",
        base_url=os.getenv("FIVE_BASE_URL", "http://192.168.1.161:8082/v1"),
        max_output_tokens=8192,
        max_steps=100,
        max_format_errors=5,
        output_dir="examples",
    )
    
    # Run the task
    trajectory_path = runner(prompt)
    
    print(f"\nTrajectory saved to: {trajectory_path}")


if __name__ == "__main__":
    main()
