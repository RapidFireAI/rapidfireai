# RapidFire AI — IDE Context Files

These files teach **Claude Code** and **Cursor** how to work with the `rapidfireai` Python package — covering the full API, usage patterns, common mistakes, and correct workflows for both fine-tuning and RAG experiments.

## Install (One Command)

**Run this from your project root** — the directory where you write your rapidfireai experiments:

```bash
# Recommended: use the CLI you already have (works on all platforms)
rapidfireai install-ide-context
```

Fallback if you don't have the CLI yet:

```bash
# macOS / Linux
curl -sSL https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/ide-context/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/ide-context/install.ps1 | iex
```

## What Gets Installed

```
your-project/
├── CLAUDE.md                          ← Claude Code: loaded every session
├── .claude/
│   └── rules/
│       └── rapidfireai-api.md         ← Claude Code: full API ref (loaded on demand)
└── .cursor/
    └── rules/
        └── rapidfireai.mdc            ← Cursor: auto-applied to .py and .ipynb files
```

## What the AI Learns

- **Environment setup** — Python 3.12+, `pip install rapidfireai`, `hf-xet` fix, port usage
- **Server lifecycle** — `init`, `start`, `stop`, `doctor`
- **Core mental model** — Experiment → Config Group → Run → Chunks/Shards
- **Full API** — `Experiment`, `RFGridSearch`, `RFRandomSearch`, `List()`, `Range()`
- **SFT/RFT classes** — `RFModelConfig`, `RFLoraConfig`, `RFSFTConfig`, `RFDPOConfig`, `RFGRPOConfig`
- **RAG classes** — `RFLangChainRagSpec`, `RFvLLMModelConfig`, `RFOpenAIAPIModelConfig`, `RFPromptManager`
- **User function contracts** — signatures and return types for all callbacks
- **IC Ops** — Stop, Resume, Clone-Modify, Delete semantics
- **Common mistakes** — `Range()` in GridSearch, wrong mode, `kill -9`, etc.
- **LoRA guidance** — which knobs matter and in what order

## Updating

When RapidFire AI releases a new version, re-run the install command to pull the latest context files:

```bash
rapidfireai install-ide-context
```
