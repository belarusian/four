#!/bin/bash

# Read input from input.txt
INPUT=$(cat input.txt)

# 1. Apply uppercase transformation and save to output.txt
echo "$INPUT" | tr '[:lower:]' '[:upper:]' > output.txt

# 2. Reverse the text (reverse the entire string)
echo "$INPUT" | rev >> output.txt

# 3. Count words and save to word_count.txt
echo "$INPUT" | wc -w > word_count.txt
