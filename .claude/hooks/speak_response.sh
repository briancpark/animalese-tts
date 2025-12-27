#!/bin/bash
# Claude Code Hook: Speak responses in Animalese
# This hook runs after Claude finishes responding and speaks the text

SCRIPT_DIR="${CLAUDE_PROJECT_DIR:-/home/katana/animalese-tts}"

# Source conda
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate animalese-tts 2>/dev/null
fi

# Read stdin and extract text using Python (no jq dependency)
LAST_TEXT=$(python3 -c '
import sys, json

try:
    # Read hook input from stdin
    hook_input = json.load(sys.stdin)
    transcript_path = hook_input.get("transcript_path")

    if not transcript_path:
        sys.exit(0)

    # Read transcript (JSONL format)
    with open(transcript_path, "r") as f:
        lines = f.readlines()

    # Find the last assistant message with text content
    for line in reversed(lines[-50:]):  # Check last 50 lines
        try:
            msg = json.loads(line)
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                print(text)
                                sys.exit(0)
                elif isinstance(content, str):
                    print(content)
                    sys.exit(0)
        except:
            continue
except:
    pass
')

if [ -n "$LAST_TEXT" ]; then
    # Speak the text (run in background to not block)
    echo "$LAST_TEXT" | "$SCRIPT_DIR/speak.py" --stdin 2>/dev/null &
fi

exit 0
