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
FIVE_LARGE_URL=http://192.168.1.161:8081/v1 FIVE_LARGE_MODEL=qwen \
FIVE_MAX_TOKENS=65536 \
python "$HOME/Research/four/examples/spokes/massive-feature-implementation.py" \
  --goal "Build a local search engine called 'personal-index' — a personal web search engine where you define your interests and the system scans, filters, and indexes the web for you. Features: (1) Interest configuration — define topics, keywords, and URL patterns to track. (2) Web crawler — configurable depth, politeness, and rate limiting. (3) Local search index — full-text search with relevance scoring. (4) Content filtering — only store what matches your interests. (5) CLI interface — add interests, run crawls, search, view results. (6) Scheduled crawling — periodic re-scanning of tracked topics. Work in the current directory only. DO NOT create generator scripts. Write real code, one file at a time. Make small commits with tests. Target: 50 commits minimum."
