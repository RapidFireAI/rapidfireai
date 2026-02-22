$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "RapidFire AI — IDE Context Installer"
Write-Host "======================================"
Write-Host "Installing into: $PWD"
Write-Host "Make sure you are running this from your project root."
Write-Host ""

$RepoBase = "https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/ide-context"

New-Item -ItemType Directory -Force -Path ".claude\rules" | Out-Null
New-Item -ItemType Directory -Force -Path ".cursor\rules" | Out-Null

Invoke-WebRequest "$RepoBase/CLAUDE.md" -OutFile "CLAUDE.md"
Invoke-WebRequest "$RepoBase/.claude/rules/rapidfireai-api.md" -OutFile ".claude\rules\rapidfireai-api.md"
Invoke-WebRequest "$RepoBase/.cursor/rules/rapidfireai.mdc" -OutFile ".cursor\rules\rapidfireai.mdc"

Write-Host ""
Write-Host "✅ Installed:"
Write-Host "   CLAUDE.md"
Write-Host "   .claude\rules\rapidfireai-api.md"
Write-Host "   .cursor\rules\rapidfireai.mdc"
Write-Host ""
Write-Host "Claude Code and Cursor will now automatically pick up the RapidFire AI context."
Write-Host "Restart your IDE or Claude Code session to activate."
