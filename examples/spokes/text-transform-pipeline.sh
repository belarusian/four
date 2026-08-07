#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# Helper to call the LLM API via curl
call_llm() {
    local prompt="$1"
    local model="${2:-granite4.1:8b}"
    local response
    response=$(curl -s -X POST "$FIVE_BASE_URL" \\
        -H "Content-Type: application/json" \\
        -d "{\\"prompt\\": \\"$prompt\\", \\"model\\": \\"$model\\"}" 2>/dev/null) || {
        echo "Error: Failed to call API at $FIVE_BASE_URL" >&2
        return 1
    }
    # Safely extract response content from common JSON structures
    echo "$response" | jq -r '.response // .content // .text // . // empty' 2>/dev/null || echo "$response"
}

# Stage 1: Uppercase
stage_uppercase() {
    local input_file="${1:-}"
    local output_file="$OUTPUT_DIR/uppercase.md"
    local input_text

    if [[ -n "$input_file" && -f "$input_file" ]]; then
        input_text=$(<"$input_file")
    else
        read -r -p "Enter initial text: " input_text
    fi

    local prompt="Convert the following text to uppercase:\\n\\n${input_text}"
    echo ">>> Stage: uppercase"
    local result
    result=$(call_llm "$prompt") || { echo "Failed at uppercase stage." >&2; return 1; }
    echo "$result" > "$output_file"
    echo "<<< Stage: uppercase complete"
}

# Stage 2: Wordcount
stage_wordcount() {
    local input_file="$OUTPUT_DIR/uppercase.md"
    local output_file="$OUTPUT_DIR/wordcount.md"

    if [[ ! -f "$input_file" ]]; then
        echo "Error: Missing input file $input_file" >&2
        return 1
    fi

    local input_text
    input_text=$(<"$input_file")
    local prompt="Count the words in the following text. Return only the number:\\n\\n${input_text}"
    echo ">>> Stage: wordcount"
    local result
    result=$(call_llm "$prompt") || { echo "Failed at wordcount stage." >&2; return 1; }
    echo "$result" > "$output_file"
    echo "<<< Stage: wordcount complete"
}

# Stage 3: Reverse
stage_reverse() {
    local input_file="$OUTPUT_DIR/wordcount.md"
    local output_file="$OUTPUT_DIR/reverse.md"

    if [[ ! -f "$input_file" ]]; then
        echo "Error: Missing input file $input_file" >&2
        return 1
    fi

    local input_text
    input_text=$(<"$input_file")
    local prompt="Reverse the following text character by character:\\n\\n${input_text}"
    echo ">>> Stage: reverse"
    local result
    result=$(call_llm "$prompt") || { echo "Failed at reverse stage." >&2; return 1; }
    echo "$result" > "$output_file"
    echo "<<< Stage: reverse complete"
}

# Pipeline orchestrator
run_pipeline() {
    echo "Starting multi-stage pipeline..."
    stage_uppercase "$@" || { echo "Pipeline failed at uppercase stage." >&2; exit 1; }
    stage_wordcount || { echo "Pipeline failed at wordcount stage." >&2; exit 1; }
    stage_reverse || { echo "Pipeline failed at reverse stage." >&2; exit 1; }
    echo "Pipeline finished successfully."
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_pipeline "$@"
fi