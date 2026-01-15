---
name: Trackio Integration Article
overview: Fork the Trackio repository, create a new integration article for RapidFire AI following the existing Transformers integration pattern, and submit a pull request to add it to the Integration section of the Trackio docs.
todos:
  - id: fork-clone
    content: Fork gradio-app/trackio repo and clone locally with upstream remote
    status: completed
  - id: create-branch
    content: Create feature branch docs/rapidfireai-integration
    status: completed
  - id: write-article
    content: Create rapidfireai_integration.md with code examples from notebook
    status: completed
    dependencies:
      - create-branch
  - id: update-toctree
    content: Add entry to _toctree.yml under Integration section
    status: completed
    dependencies:
      - write-article
  - id: commit-push
    content: Commit changes and push to fork
    status: completed
    dependencies:
      - update-toctree
  - id: create-pr
    content: Create pull request to gradio-app/trackio
    status: cancelled
    dependencies:
      - commit-push
---

# RapidFire AI Integration Article for Trackio

## Overview

Contribute a new integration document (`rapidfireai_integration.md`) to the [Trackio repository](https://github.com/gradio-app/trackio) under the "Integration" section. The article will educate ML engineers on how to use Trackio with RapidFire AI for experiment tracking.

## Git Contribution Workflow

### Step 1: Fork and Clone Repository

```powershell
# Fork the repository using GitHub CLI
gh repo fork gradio-app/trackio --clone

# Navigate to the cloned repo
cd trackio

# Add upstream remote
git remote add upstream https://github.com/gradio-app/trackio.git
```

### Step 2: Create Feature Branch

```powershell
git checkout -b docs/rapidfireai-integration
```

## Document Creation

### Step 3: Create Integration Article

Create `docs/source/rapidfireai_integration.md` following the [Transformers Integration](https://huggingface.co/docs/trackio/en/transformers_integration) pattern.

**Document Structure:**

1. **Title and Introduction** - Brief intro explaining RapidFire AI and the integration
2. **Installation** - Quick install command for both packages
3. **Configuration** - How to enable Trackio in RapidFire AI via environment variables
4. **Code Example** - Minimal working example from the tutorial notebook
5. **What Gets Tracked** - Training metrics, eval metrics, hyperparameters
6. **Viewing the Dashboard** - `trackio show` command
7. **Learn More** - Links to tutorials and documentation

**Key Content Sources:**

- [Co-Announcement Blog](tutorial_notebooks/fine-tuning/Co-Announcement%20Blog%20Trackio%20and%20RapidFire%20AI.md) - Introduction text and feature descriptions
- [Tutorial Notebook](tutorial_notebooks/fine-tuning/rf-tutorial-sft-trackio.ipynb) - Code snippets for configuration and usage

**Code Snippet (from notebook):**

```python
import os

# Enable Trackio as the tracking backend
os.environ["RF_TRACKIO_ENABLED"] = "true"
os.environ["RF_MLFLOW_ENABLED"] = "false"

from rapidfireai import Experiment
from rapidfireai.automl import RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

experiment = Experiment(experiment_name="my-experiment", mode="fit")
experiment.run_fit(config_group, create_model_fn, train_dataset, eval_dataset)
```

### Step 4: Update Table of Contents

Add entry to `docs/source/_toctree.yml` under the Integration section (line ~32):

```yaml
- sections:
  - local: transformers_integration
    title: Transformers Trainer
  - local: rapidfireai_integration
    title: RapidFire AI
  title: Integration
```

## Submit Pull Request

### Step 5: Commit and Push

```powershell
git add docs/source/rapidfireai_integration.md docs/source/_toctree.yml
git commit -m "docs: add RapidFire AI integration guide"
git push origin docs/rapidfireai-integration
```

### Step 6: Create Pull Request

```powershell
gh pr create --title "docs: Add RapidFire AI integration guide" --body "..."
```

## Notes

- The article will NOT include Trackio Space ID configuration (per your specification)
- Links to 4 future tutorial notebooks will be added as placeholders with a note that more examples are coming
- Follow the concise style of the Transformers integration (minimal prose, code-focused)
