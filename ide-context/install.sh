#!/usr/bin/env bash
set -e

echo ""
echo "RapidFire AI — IDE Context Installer"
echo "======================================"
echo "Installing into: $(pwd)"
echo "Make sure you are running this from your project root."
echo ""

REPO_BASE="https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/ide-context"

mkdir -p .claude/rules
mkdir -p .cursor/rules

curl -fsSL "$REPO_BASE/CLAUDE.md" -o CLAUDE.md
curl -fsSL "$REPO_BASE/.claude/rules/rapidfireai-api.md" -o .claude/rules/rapidfireai-api.md
curl -fsSL "$REPO_BASE/.cursor/rules/rapidfireai.mdc" -o .cursor/rules/rapidfireai.mdc

echo ""
echo "✅ Installed:"
echo "   CLAUDE.md"
echo "   .claude/rules/rapidfireai-api.md"
echo "   .cursor/rules/rapidfireai.mdc"
echo ""
echo "Claude Code and Cursor will now automatically pick up the RapidFire AI context."
echo "Restart your IDE or Claude Code session to activate."
