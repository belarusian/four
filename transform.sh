#!/bin/bash

# Read input file
INPUT="input.txt"
OUTPUT="output.txt"
WORD_COUNT="word_count.txt"

# Apply uppercase transformation
UPPERCASE=$(cat "$INPUT" | tr '[:lower:]' '[:upper:]')

# Reverse the text (character-wise reversal)
REVERSED=$(echo "$UPPERCASE" | rev)

# Write the reversed uppercase text to output.txt
echo "$REVERSED" > "$OUTPUT"

# Count words
WORDS=$(cat "$INPUT" | wc -w | tr -d ' ')
echo "$WORDS" > "$WORD_COUNT"

echo "Transformations complete."
echo "Output: $(cat $OUTPUT)"
echo "Word count: $(cat $WORD_COUNT)"
