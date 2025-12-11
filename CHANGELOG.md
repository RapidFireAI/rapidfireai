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

## [v0.12.7]

### Changes
- Added comment cells to gsm8k and scifact notebooks on openai api costs
- Downsampled data further in both notebooks to reduce cotsts
- Now api costs are $3 and $5 only
- Reasoning effort knob adjusted in gsm8k, reducing to 6 configs
- Automatic check if running in Google Colab for simpler install/init process
- Add some checks to rapidfireai doctor to give warning or errors if some required items not installed/compatible
- Add additional Python packages to rapidifreai doctor
- Add Torch version and Torch CUDA version to rapidfireai doctor
- Support for evals of newer versions of Torch based on Cuda version
- Move OPENAI_API_KEY to first line of Notebooks
- Set default Ray console port to 8855, and allow customizing the port RF_RAY_PORT
- Updates to notebook headers and README.md
- Moved most RapidFire AI files into new RF_HOME location that defaults to ~/rapidfireai or /content/rapidfireai (for Colab)
- Consolidated environment variables and common constants to rapidfireai/utils/constants.py
- Add RF_HOME/rf_mode.txt file to store fit or evals depending on flag to rapidfireai init
- Altered Colab notebooks to not need to run commands from the terminal anymore, default to T4 session
- Altered Colab notebooks remove need for the Hugging Face token
- For evals if system has 2 or fewer CPUs default Ray number of CPUs to 1
- For evals if system has 1 or fewer GPUs default Ray number of GPUs to 0.5
- Added get_log_file_path() to Experiment class to get log file location
- Fix locations in JavaScript files for RF_ environment variable HOST and PORT
- Add RF_TIMEOUT_TIME to change default (30s) timeout for service startups
- Moved rapidfireai doctor code out of cli.py into dedicated module
- Expanded rapidfireai doctor to provide more information including default of 10 lines of log files (override with --log-lines -1 for all lines, 0 for no lines)
- Alter fit hard torch 2.5.1 version requirement

### Fixes
- IC Ops panel for evals fixed
- Re-Support run_fit() on Google Colab
- Updated IC Ops Clone code to create new run's config with the user's modifications of the parent run's config.
- Fixed rapidfireai status output to not always show success
- Fixed rapidfire start from automatically stopping services if run again, now prompts to shutdown

### Added
- run_evals() support on Google Colab
- Add rapidfireai jupyter command, including outputting tunneling recommendations based on if in VSCode or not
- Jupyter default URL is /tree
- Add rapidfireai --test-notebooks to copy test notebook to tutorial_notebooks folder
- Add additional items to startup issues check
- Added section to README.md for RapidFire AI environment variables
- For colab notebooks add confirmation button before experiment.end()


## [v0.12.6]
*NOTE:* Colab is not working in this release, use 0.11.1 for fine tuning

### Changes
- Added Gantt charts in images
- Update documentation links for HuggingFace

### Fixed
- RAG/Evals notebook issues with rate limit, no CUA GPUs, and unpickle messages
- Experiment.end() for RAG
- Datagrame for results

## [v0.12.5]
*NOTE:* Colab is not working in this release, use 0.11.1 for fine tuning

### Changes
- Update colab notebook to point to 0.11.1 release
- Updated readme
- Reorganized `tutorial_notebook` folder


## [v0.12.4]
*NOTE:* Colab is not working in this release, use 0.11.1 for fine tuning
### Fixed
- Fix problems running rapidfireai start
- Additional filesystem fixes for RAG/FIT split

### Changes
- Changed default ports for Dispatcher = 8851, MLFlow = 8852, Frontend = 8853
- New log directory variable RF_LOG_PATH
- Additional changes to allow customization of ports used

## [v0.12.3]
### Fixed
- Image locations in tuorial notebooks


## [v0.12.1]
### Added
- Initial open source release of RapidFire AI RAG for RAG inference and context engineering experimentation
- Expanded Experiment API to add `run_evals` method
- Added evals folder under rapidfireai with new controller, scheduler, workers, automl, etc. for RAG pipelines
- Added 3 new tutorial notebooks for RAG and context engineering experiments: FiQA, GSM8K, and SciFact
- Added evals-specific additional setup and requirements

### Changes
- Refactored rapidfireai folder to move sft/rft workflows under fit subfolder
- Refactored tutorial_notebooks folder to move notebooks to evals vs. fit subfolders
- Refactored setup folder to for specific additional requirements and comands for evals vs. fit
- README updated to reflect support for RAG/context engineering workflows and evals installation

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
