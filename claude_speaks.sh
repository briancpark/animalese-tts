#!/bin/bash
# Claude Code Animalese Hook
# This script speaks Claude's responses in Animalese
#
# Usage as a hook: Add to Claude Code settings
# Or run manually: echo "text" | ./claude_speaks.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="animalese-tts"

# Activate conda environment
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null
fi

# Read text from stdin or arguments
if [ -p /dev/stdin ]; then
    TEXT=$(cat)
else
    TEXT="$@"
fi

# Speak the text
if [ -n "$TEXT" ]; then
    # Try daemon first (faster), fall back to direct mode
    if [ -S /tmp/animalese.sock ]; then
        echo "$TEXT" | nc -U /tmp/animalese.sock -w 1 2>/dev/null
    else
        echo "$TEXT" | python "$SCRIPT_DIR/speak.py" --stdin
    fi
fi
