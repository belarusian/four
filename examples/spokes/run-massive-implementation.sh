#!/bin/bash
set -e

# ── Defaults ────────────────────────────────────────────────────────────────
VARIANT="${VARIANT:-long-running}"           # massive | context-aware | long-running
PROJECT_DIR="${PROJECT_DIR:-$HOME/Research/autonomous-project}"
PROJECT_NAME="${PROJECT_NAME:-autonomous-project}"
GOAL="${GOAL:-}"

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --goal GOAL          Project description (required, or set GOAL env var)"
    echo "  --variant VARIANT    'massive' (single model), 'context-aware' (two-model), or 'long-running' (two-model + summarization, default)"
    echo "  --project-dir DIR    Working directory (default: ~/Research/autonomous-project)"
    echo "  --project-name NAME  GitHub repo name (default: autonomous-project)"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --goal 'Build a CLI tool called foo with bar and baz features'"
    echo "  VARIANT=massive $0 --goal 'Quick prototype'"
    echo "  GOAL='Build a search engine' $0"
    exit 0
}

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --goal) GOAL="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$GOAL" ]; then
    echo "Error: --goal is required (or set GOAL env var)"
    usage
fi

# ── Select spoke ────────────────────────────────────────────────────────────
SPOKE_DIR="$HOME/Research/four/examples/spokes"
case "$VARIANT" in
    massive)
        SPOKE="$SPOKE_DIR/massive-feature-implementation.py"
        ;;
    context-aware)
        SPOKE="$SPOKE_DIR/context-aware-implementation.py"
        ;;
    long-running)
        SPOKE="$SPOKE_DIR/long-running-implementation.py"
        ;;
    *)
        echo "Error: unknown variant '$VARIANT' (use 'massive', 'context-aware', or 'long-running')"
        exit 1
        ;;
esac

if [ ! -f "$SPOKE" ]; then
    echo "Error: spoke not found: $SPOKE"
    exit 1
fi

# ── Setup project dir ───────────────────────────────────────────────────────
echo "Setting up project at $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if [ ! -d ".git" ]; then
    git init
    git commit --allow-empty -m "initial"
fi

# ── Push to remote on exit ──────────────────────────────────────────────────
cleanup() {
    echo "Pushing to remote..."
    git add -A 2>/dev/null || true
    git commit -m "autonomous: save progress" 2>/dev/null || true
    if ! git remote -v | grep -q "origin"; then
        REMOTE_URL="https://github.com/belarusian/${PROJECT_NAME}.git"
        echo "Creating GitHub repo: $PROJECT_NAME"
        gh repo create "belarusian/$PROJECT_NAME" --public --source=. --remote=origin --push 2>/dev/null || \
        (git remote add origin "$REMOTE_URL" 2>/dev/null; git push -u origin main 2>/dev/null) || true
    else
        git push 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Run ─────────────────────────────────────────────────────────────────────
echo "Variant: $VARIANT"
echo "Spoke:   $SPOKE"
echo "Goal:    $GOAL"
echo "---"

FIVE_BASE_URL=http://192.168.1.157:8080/v1 \
FIVE_MODEL=fast-qwen \
FIVE_LARGE_URL=http://192.168.1.161:8081/v1 \
FIVE_LARGE_MODEL=qwen \
FIVE_MAX_TOKENS=65536 \
python "$SPOKE" --goal "$GOAL"
