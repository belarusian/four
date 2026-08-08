#!/bin/bash
set -e

PROJECT_DIR="$HOME/Research/autonomous-project"

echo "Setting up project at $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if [ ! -d ".git" ]; then
  git init
  git commit --allow-empty -m "initial"
fi

echo "Starting autonomous pipeline..."
FIVE_BASE_URL=http://192.168.1.157:8080/v1 FIVE_MODEL=fast-qwen \
python "$HOME/Research/four/examples/spokes/massive-feature-implementation.py" \
  --goal "Build a new Python CLI tool called 'taskflow' — a lightweight project management system with: (1) Ticket/issue tracking with labels, priority, and assignee support. (2) Real-time collaboration features with WebSocket-based live updates and notification engine. Target: 1000 commits across 200 PRs with 100% test coverage. Each PR must be small, focused, and include tests."
