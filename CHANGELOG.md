# Change Log

All notable changes `rapidfireai` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

## Types of changes

- Added for new features.
- Changed for changes in existing functionality.
- Deprecated for soon-to-be removed features.
- Removed for now removed features.
- Fixed for any bug fixes.
- Security in case of vulnerabilities.

## [v0.11.1]
### Added
- Support for Google Colab
- Links for RapidFireAI documentation, GitHub, and Discord in notebooks
- Support for Tensorboard, currently only Colab

### Changes
- Relaxed `pip` dependency versions


## [v0.10.2]
### Changes

- Update links in README file for docs
- Add helpful links to beginning of notebooks
- Add comprehensive CLAUDE.md documentation files to guide AI-assisted development across all RapidFire modules
- Refreshes README hero with centered logo and buttons, adds PyPI badge, clarifies local-only ports, community links, and updates tagline to 16-24x throughput without extra GPUs
- Refactor bump_version.sh, include option to specify a version
- Changed manual workflow to handle release candidates
- Update BUILD.md instructions


## [v0.10.1]
### Changes

- Upgrade to MLFlow 3 
- Comment out installing `vllm` and `flash-attn` for now
- Changed to be multiples of eff. batch and chunks
- Adjusted logging steps and eval steps to show more points on curves
- Unpins the TRL version requirement to >= 0.21.0 and PyTorch versions to >= 2.7.0
- Include GH templates for PRs and new issues
- Documentation: standardize to Python venv (.venv) and update setup guidance and readme

### Fixes
- Added cloned_from key in the DB for the runs table.
- Code to correctly set cloned_from as well as warm_stared_from details for all runs in Controller, Worker and Trainer. [Issue 24 - IC-OP Clone(Non-Warm) fails with GRPO use case ](https://github.com/RapidFireAI/rapidfireai/issues/24)
- Commented out _clear_run_from_shm for deleted runs [Issue 22 - IC-OP Clone and Delete on a same parent run ](https://github.com/RapidFireAI/rapidfireai/issues/22)
- warm_started_from key was assigned a value that didn't exist. Updated it to hold the value of parent's run_id
- Modified chunks.py, specifically the _create_chunk_indices() method to distribute batches instead of examples
Chunks now prioritize having sizes as multiples of batch_size with at most 1 chunk having a non-multiple size
The non-multiple chunk is positioned last to contain any partial batch [Issue 37 - Dataset chunks ignore batch_size, breaking algorithmic equivalence.](https://github.com/RapidFireAI/rapidfireai/issues/37)
- Fixed step calculation: Updated total step count to correctly multiply dataset_len / effective_batch_size by num_generations in controller.py and trainer.py [Issue 34 - Total Steps/Number of Chunks of runs for GRPO-Lite notebook is off ](https://github.com/RapidFireAI/rapidfireai/issues/34)
- Fixed datatype issue in controller.py by setting default value to 0
- Modified GRPO-lite notebook to run for only 1 epoch instead of 3

## [v0.9.11]
### Changes
- Add option to create a Test PyPI package
- Added logging of rf version number in Experiment class's init method.
- Do not install old version of flash-attn on NVIDIA 7 GPUs
- Add `RF_` prefix and accept existing values for MLFLOW_PORT, MLFLOW_HOST, FRONTEND_PORT, FRONTEND_HOST, API_PORT, API_HOST, PID_FILE
- Allow use of RF_DB_PATH for DBConfig.DB_PATH in constants
- Added lite versions of tutorial_notebooks that will run under 2 hours on T4 GPUs
- rapidfireai init Copies tutorial notebooks to RF_TUTORIAL_PATH or ./tutorial_notebooks

### Fixes
- Added checks in create_warm_start_checkpoint method in shm_manager.py. This method is only triggered during clone-modify warm start operations.
- Creates mlflow.db file in RF_DB_PATH or ~/db
- If netcat's nc not found rapidfireai start now falls back to a python port checker
- `rapidfireai start` now tries python3/pip3 if possible then falls back to python/pip

## [v0.9.10]
### Changes
- Updated main name of CLI prog to rapidfireai

## [v0.9.9] - 09-03-2025
### Added
- Initial open source release
