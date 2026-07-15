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

## [Unreleased]

### Fixed
- Fixed `#219`: `rapidfireai --version` and `--help` no longer implicitly create `$RF_HOME` or `$RF_HOME/logs`. Directory initialization is now deferred until non-pure-info subcommands are run.

## [v0.16.1]

### Additions
- `rapidfireai/utils/mode_utils.py`: `get_installed_mode()` and `assert_mode_matches()` helpers for validating the installed RapidFire mode (`$RF_HOME/rf_mode.txt`).
- New RF_API_WORKERS / RF_API_WORKER_CLASS / RF_API_THREADS / RF_API_TIMEOUT / RF_API_GRACEFUL_TIMEOUT / RF_API_KEEPALIVE env vars on DispatcherConfig to tune the dispatcher gunicorn server.
- New include_decoded_config flag on RFDatabase.get_pipeline* / get_all_pipelines so HTTP callers can skip the expensive (and SIGSEGV-prone) dill decode of FAISS / embedder blobs.
- Autopilot/chat-agent clone path now stamps requests with source: "autopilot" and refuses to launch a clone whose effective config duplicates an existing run.
- Per-step intermediate checkpoint folders (`checkpoint-<step>`) for `save_strategy="chunk"`, preserving one snapshot per chunk.
- Configurable eval generation via `generation_config`: `max_input_length`, `add_generation_prompt`, and `truncation_side`.
- New `Status` column on the experiment runs table (pinned left after Run Name) showing dispatcher-vocabulary lifecycle state with per-state color.
- New `Shards` column showing live shard / chunk progress (`current/total`) for both evals (per-shard) and fit (per-chunk-per-epoch) runs.
- New mutable MLflow tags `rapidfire.progress.current` and `rapidfire.progress.total` emitted by both the fit and evals controllers; `MetricLogger.set_tag()` API across all backends.
- Stranded-run recovery (`rf_db.py` `reset_all_tables` path) now also terminates orphaned MLflow runs as `FAILED`.
- New unit-test suite `tests/test_mlflow_status_parity.py` covering every status-parity transition, idempotency path, and the new helpers.

### Changes
- Experiment now validates that the installed mode matches the requested mode at initialization — before Ray/DB setup — blocking early with a clear, actionable error on a mismatch instead of silently producing a dead IC Ops panel.
- `doctor` reads the installed mode via `get_installed_mode()` (single source of truth).
- Dispatcher gunicorn defaults switched from sync worker to gthread (4 threads, 120 s timeout, 5 s keepalive) for both evals/ and fit/ dispatchers to survive long-lived polling connections.
- HTTP polling endpoints (/get-all-pipelines, /get-all-runs, /get-pipeline, IC validate calls) no longer dill-decode the pipeline_config blob; get_all_runs recovers per-row instead of failing the whole poll.
- Frontend proxy strips hop-by-hop headers, returns 502 for upstream failures, and absorbs ChunkedEncodingError / ProtocolError mid-stream instead of logging tracebacks.
- Eval generation now left-pads and forwards an explicit `attention_mask`; input truncation length is derived from the tokenizer/model instead of a fixed 512 tokens.
- Worker reads the user's `save_strategy` as the single source of truth; training args use a shallow copy so the `"chunk"` sentinel is preserved on `config_leaf`.
- Changed default behavior for `rapidfireai init` to be `--evals`, to do training use `rapidfireai init --train`
- Pin datasets==3.6.0 and pyarrow>=18.1.0
- Remove output from rag-fiqa notebook
- Dashboard relabels MLflow's native enum to the dispatcher's vocabulary across the experiment runs table and the run detail page: `Finished` → `COMPLETED`, `Killed` → `STOPPED`, `Running` → `ONGOING`, `Scheduled` → `NEW`.
- `KILLED` (dispatcher `STOPPED`) now renders with a `StopCircleIcon` in warning amber instead of the red error icon used for `FAILED` — an intentional halt is no longer visually conflated with a failure.
- `DateCellRenderer` no longer prefixes the timestamp with the run-status icon (the new `Status` column owns that signal).
- `experiment_utils.create_experiment` / `end_experiment` and `MLflowMetricLogger.clear_context()` now read the active run's server-side status before calling `mlflow.end_run()` so a terminal state already set by the controller / worker / IC handler is preserved instead of being overwritten with `FINISHED`.
- RunFit Notebooks updated with param mode='fit'

### Fixes
- A missing `rf_mode.txt` no longer blocks `run_fit()`; it now defaults to fit mode, matching the dispatcher startup default (`run_evals()` remains correctly blocked in that case).
- Fixes dispatcher worker WORKER TIMEOUT / "no URI read" SIGKILLs caused by the sync worker blocking on keep-alive sockets.
- Fixes dispatcher worker SIGSEGVs when the autopilot polling loop re-hydrated ~25 FAISS-bearing pipeline blobs per poll.
- Fixes Converge autopilot launching byte-identical duplicate clones and silently dropping cloned runs because their flattened_config was empty.
- Fixes AttributeError in _transform_pipeline_to_run when SQLite returned NULL for string columns (.get(key, "") doesn't apply the default to None values).
- Optuna label building no longer crashes on `dict`/`None` categorical knob choices (#268).
- Optuna leaf configs now carry the `pipeline` key so the controller can launch them (#268).
- Query actor reuses its inference engine only when it actually exists, fixing `AttributeError` under sharding (#268).
- Replacement (relaunched) pipelines now inherit the parent's rate limiter and max-completion-tokens, fixing inference-engine init failures (#268).
- Fixed intermediate checkpoints overwriting each other every chunk save.
- Fixed `save_strategy="chunk"` being lost/mutated before the worker's save step.
- Fixed corrupted batched eval generation when `pad_token_id == eos_token_id` (missing attention mask + right padding).
- Fixed missing `add_generation_prompt` in eval chat templating, which degraded eval quality for instruction-tuned models.
- Dashboard MLflow status now matches the dispatcher status shown in the notebook progress table across all terminal transitions in both `run_fit()` and `run_evals()`: clean completion, IC-stop, Optuna prune, worker exception, runtime / init / dispatch failure, create-run failure, and stranded-run recovery after a crash.
- Reused query actors no longer mark a still-running pipeline as `FINISHED` on the MLflow server when switching to the next pipeline.
- Orphaned `ONGOING` / `NEW` runs left over from a crashed experiment no longer remain stuck as `RUNNING` in MLflow after recovery.
- Cloned pipelines (IC clone) and Optuna replacement runs now render shard progress in the `Shards` column instead of `-`.
- Failed run-creation paths now record the MLflow run as `FAILED` instead of letting it default to `FINISHED`.
- Downgrade `numpy` to 2.0.1 on Colab for SFT notebook issue with datasets


## [v0.16.0]

### Additions
- Pinecone `source_tag` was added for integration analytics
- Conditional MLFlow tracing: MLFlow logging can now be toggled through `RF_MLFLOW_ENABLED` environment variable. Accepts true or false
- MLFlow traces now show on RapidFire AI dashboard
- Google Gemini API support through `RFGeminiAPIModelConfig` (mirrors `RFOpenAIAPIModelConfig` interface)
- Dedicated `GoogleGeminiRateLimiter` with character-based token approximation
- `gemini_config` key support in `RFGridSearch` and `RFRandomSearch` configs
- Separate per-backend `RateLimiterActor` instances so OpenAI and Gemini pipelines can coexist in the same experiment
- MLFlow tracing for gemini generations
- Pretty error display for `Experiment` creation and `run_evals()`
- Added `num_gpus` and `num_cpus` parameters to `Experiment()` initialization
- Option to disable starting frontend
- Add flags to specify NVIDIA CUDA and Compute Capability versions
- Mirroring colab tutorial notebooks under tests/staging for build process
- Additional notebooks placed under tests/notebooks for testing features, not to be used for tutorials
- Add option to specify or disable CUDA, change error for no CUDA to just a warning
- **MLflow AI Gateway integration**: new `RFAPIModelConfig` works with any provider (OpenAI, Gemini, Anthropic, Azure, Triton) through an OpenAI-compatible gateway; `MLflowGatewayClient` idempotently provisions secrets, models, and endpoints.
- **AI Gateway dashboard** in the frontend — Endpoints, API Keys, Usage charts, Try-It playground, plus a GenAI Experiment Overview page with assessment/tool-call/trace cost-latency-token charts (sidebar gains an "AI Gateway" tab; replaces "Prompts").
- **Anthropic provider support** via `anthropic>=0.96.0` and a new `AnthropicRateLimiter` built on a fine-grained sliding-window limiter with separate ITPM/OTPM limits.
- New `rf-tutorial-scifact-generators.ipynb` and updated colab/GCE notebooks demonstrating unified `RFAPIModelConfig` across providers.
- Frontend proxy forwards `/gateway` requests to the MLflow target.
- Multimodal RAG: `LangChainRagSpec.multimodal_processor` runs per-modality text/image/table summarizers over `UnstructuredBaseLoader` chunks.
- `LangChainRagSpec.artifact_storage_cfg` and new `CloudStorage` (S3 / GCS) for offloading large image and table HTML artifacts out of vector-store metadata.
- `LangChainRagSpec.document_loader` accepts a list of loaders (with `None` entries skipped).
- New tutorial: `rf-tutorial-MMDocIR.ipynb` (multimodal RAG)
- New `LocalStorage` class in `rapidfireai.evals.utils.storage_utils` mirroring `CloudStorage`'s `put_*` / `get_*` / `read_bytes` interface; writes under `{bucket}/{prefix}/{type}/{rf_doc_id}/document` and returns absolute filesystem paths.
- `artifact_storage_cfg` accepts `False` to explicitly disable offloading (drops `*_source` from metadata after summarization).

### Changes
- Ray resource allocation overhaul. Resources are spared for other processes to avoid overloading the node.
- `num_actors`, `gpus_per_actor`, `cpus_per_actor` are auto calculated using available resources but can be overridden by user. 
- Infeasible user overrides are scaled down to the closest feasible hardware allocation
-  deprecated methods removed
- `num_actors` removed from tutorial notebooks
- `RateLimiterActor` now accepts a `backend` parameter (`"openai"` | `"gemini"`) to select the appropriate rate limiter
- `pipeline_type` in the database is now derived from the pipeline object (`"vllm"` / `"openai"` / `"gemini"`) instead of being hardcoded
- IC Ops clone now selects the rate limiter by matching the cloned pipeline's backend type
- `sampling_params_to_dict()` on API model configs now uses a blocklist (exclude `"model"`) instead of a hardcoded key whitelist, making it forward-compatible with new API parameters
- default limit_safety_ratio for rate limiters changed from 0.90 to 0.95
-`set_aside` logic now reserves ~10% of CPUs by default for OS and other processes
- If using Converge, sets RF_MLFLOW_ENABLED and RF_TENSORBOARD_ENABLED to true
- Handle SIGTERM event on parent process of `rapidfireai start` to kill child processes (start.sh, and each sub-python process).  Fixes #217 
- Change default HOST for most variables to be 0.0.0.0 instead of 127.0.0.1
- Update URL displayed for frontend to be `localhost` if RF_FRONTEND_HOST is set to 0.0.0.0.  Fixes #230
- Unified `OpenAIInferenceEngine` + `GoogleGeminiInferenceEngine` into a single `APIInferenceEngine` that talks to the gateway via `AsyncOpenAI`.
- `RFGridSearch`/`RFRandomSearch`: pipeline key `openai_config`/`gemini_config` → `api_config`; reinstantiated configs get `verbose=False` to suppress duplicate gateway-provisioning logs.
- Rate limiter API: `estimate_total_tokens()` → `count_prompt_tokens()`; `update_actual_usage()` takes input/output tokens separately; one actor per provider with an OpenAI/tiktoken fallback for unknown providers.
- Dispatcher pipeline-clone payload uses `pipeline_type: "api"` (with required `endpoint_config`) instead of separate `"openai"`/`"gemini"` types; serialization emits a single `"api"` branch and strips `api_key` from `endpoint_config`.
- Bumped `mlflow` to `>=3.11.1` (`mlflow[genai]` in core) across `pyproject.toml` and all `setup/*/requirements*.txt`.
- Output ports using tcp:// to allow auto port forwarding of VS Code/Cursor
- If no GPU or Nvidia GPU changes error to a warning and disable GPU tasks
- Support install and init on a non-Nvidia GPU machine
- Installs faiss-cpu on a non-Nvidia machine
- `LangChainRagSpec.artifact_storage_cfg=None` now defaults to local-disk offload under `$RF_HOME/artifacts` instead of being a silent no-op.
- Added `"local"` to the accepted `backend` values alongside `"s3"` and `"gcs"`.
- MMDocIR tutorial notebook updated to use the new local-disk default with cloud / opt-out alternatives shown inline.
- Adds a `--multimodal` flag to `rapidfireai init --evals` to check if required OS packages are installed, and will then install the needed Python packages

### Fixes
- Dispatcher running out file descriptors error fixed
- get_runs now checks for Textsplitter encoding
- init script now works on cpu-only machines. NVML error fixed.
- Missing pipeline_id and model_name attributes added to traces in prompt manager
- RFLogger file handling bug fixed
- On startup show correct log information depending on frontend source
- Update readme with correct default ports
- Addresses issue #220 by flushing jupyter output
- HTTP 500 on `/dispatcher/get-all-runs` (and `/dispatcher/get-all-pipelines`)
- Fix issue where RF_FRONTEND_HOST was never actually used
- Decoupled `RFvLLMModelConfig` and `RFAPIModelConfig` so each is gated only by its own optional deps.
- Optional-dep placeholders in `rapidfireai.automl` are now sentinel **classes** via `_make_unavailable_class` instead of `None`, so `isinstance(x, RFLoraConfig)` etc. no longer raises `TypeError: isinstance() arg 2 must be a type`.
- Colab `torchcodec` ABI mismatch: pinned `torchcodec<=0.7.0` in `setup/evals/requirements-colab.txt`.
- Rate-limit wait-time is now deterministic — RPM/TPM/ITPM/OTPM all use `minimum_wait_time` instead of depending on the oldest in-flight request.
- Controller reads `pipeline.model_name` (property) instead of `pipeline.model_config["model"]`, fixing display/logging for gateway endpoint configs.
- `APIInferenceEngine.update_actual_usage` for `FAILED` requests now passes both input and output token counts (previously dropped one arg, raising `TypeError`).
- `BaseRateLimiter.register_model` now normalises rate-limit dicts (accepts either `{"rpm","tpm"}` or `{"rpm","itpm","otpm"}`), matching its `FineGrainedBaseRateLimiter` sibling and removing a latent `KeyError`.
- `RFAPIModelConfig.sampling_params_to_dict` no longer duplicates `max_completion_tokens` in the IC Ops JSON view — now excludes `max_completion_tokens`, `model`, and `messages` (mirrors the keys `APIInferenceEngine.__init__` pops).
- Install issues on systems with CUDA and without a Nvidia GPU
- Fixes issue with dependencies upgrading numpy, by pinning numpy 2.0.1
- Restricts peft<0.19.0 until transformers can be upgraded to >5.0
- Pins cupy-cuda12x to 14.0.1 to override issue in 14.1.0 that requires `pytest` the issue is fixed in `https://github.com/cupy/cupy/pull/9965` but there is not a current release with this fix.


## [v0.15.2]

### Additions
- MLFlow tracing for rag retrieval, prompt manager, vllm and openai generate accessible through the mlflow dashboard (8852)

### Changes
- Updates to pinecone vector store: index_name is now index_namespace and accepts a tuple of type (str, str)
- Updated GitHub action versions

### Fixes
- flashinfer upgrade flag set for rapidfireai init command
- Resolve Torch compatibility issues with Flash Attention

## [v0.15.1]

### Changes
- Re-added Refresh button(sync icon) to MLflow dashboard
- Updated pinecone and pgvector rag notebooks
- Remove flashinfer and pin numpy to 2.0.1 for Colab to handle restrictions on limits of Google Colab


## [v0.15.0]

### Additions
- combines new vector store integrations for postgres and pinecone along with a round of bugfixes

### Changes
- Add community notebooks from AI Winter 2025 competition winners
- Integrated multi-GPU training via Fully Sharded Data Parallelism (FSDP), enabling large models to be distributed across multiple GPUs.
- Added Sanity Automation Test for SFT
- Disable starting frontend with rapidfireai start if RF_MLFLOW_ENABLED is not enabled
- Standardized MLflow and TensorBoard naming throughout
- Downsampled fsdp large and fsdp medium notebooks for demo purposes, Increased eval batch size for fsdp-large notebook, Changed fsdp evaluation such that each GPU's model.generate() will handle different batch
- Added component independence to rag pipeline.
- Coupled "_cls" and "kwargs" attributes into a "_cfg" attribute for embedding, search, and reranker in RFLangChainRagSpec, RFPromptManager
- FSDP LoRA State Dict Collection (_collect_fsdp_peft_state_dict)
- GPU Memory Cleanup Utilities
- Added constraint to enforce top_n <= top_k across reranker and search categories in grid search and random search
- Fixed cloned run metrics not appearing in the notebook display table and result_dict
- Fixed display table and result_dict to show all user-configurable knobs
- run_evals changes: Added plots for confidence intervals for metrics, Removed the lower_bound and upper_bound plots
- Added plots for confidence intervals for metrics, Removed the lower_bound and upper_bound plots


### Fixes
- Various changes to tutorial notebooks [Issue #173](https://github.com/RapidFireAI/rapidfireai/issues/173)
- [Issue #187](https://github.com/RapidFireAI/rapidfireai/issues/187)
- [Issue #188](https://github.com/RapidFireAI/rapidfireai/issues/188)
- [Issue #189](https://github.com/RapidFireAI/rapidfireai/issues/189)
- Type checking for default search knobs


## [v0.14.0]

### Changes
- Disable IC Ops button in MLflow if no active experiment
- Add useIsExperimentRunning hook in frontend to check if specific experiment is running
- Update frontend build packages
- Update `fiqa` notebook to use max_model_len: 4096
- Use logger instead of prints in metric loggers
- Added support for Trackio delete_run()
- Add Trackio tutorial and documentation
- Change evals run names to not prefix Pipeline #_
- Not use transformers>=5.0 for now

### Fixes
- Bug fixes to Colab notebook
- Fix rf_db.py to return status as string instead of enum for JSON serialization
- Consolidate tensorboard logs to single directory for each run
- Fix dependency version problems with new updates by pinning dill, numpy, tensorboard, setuptools, and huggingface-hub
- Fixed Trackio's log() method to accept step as a separate keyword argument instead of including it within the metrics dictionary.
- Fix cloned runs in evals mod
- IC Logs for evals looking for same string as training

## [v0.12.9]

### Changes
- Added mlflow logging to evals
- Added Trackio logging to both fit and evals
- Deprecated RF_TRACKING_BACKEND environment variable for backends
- Environment variable RF_MLFLOW_ENABLED to enable MLFLOW tracking (default to true for non-Colab environments)
- Environment variable RF_TENSORBOARD_ENABLED to enable TENSORBOARD tracking (default to true for Colab environments)
- Environment variable RF_TRACKIO_ENABLED to enable TRACKIO tracking (default to false for all environments)
- `--tracking-backend` flag for `rapidfireai start` changed to `--tracking-backends` to now allow multiple flags with one of the following: mlflow, tensorboard, trackio
- Remove more pinning of dependent modules
- Rename `mlflow_run_id` column in database to `metric_run_id` to be more generic for metrics logger
- Rename `mlflow_experiment_id` column in databasse to `metric_experment_id` to be more generic for metrics logger
- Consolidated mlflow_manager, tensorboard_manager, and trackio_manager for both `evals` and `fit`
- New default `metric_rfmetric_manager` as Metrics manager that accepts one or more Metrics loggers
- Added more documentation to `rf-colab-rag-fiqa-tutorial.ipynb` notebook to help beginners

### Fixes
- Fix issue when triggering a warm-clone (IC_CLONE_MODIFY_WARM) on a run that has completed its last chunk, the controller throws: ValueError: Invalid chunk_id 4
- (#132) Fix when a run completes an epoch before other runs, it gets scheduled repeatedly while other runs are starved. This causes trials to run sequentially instead of in parallel after the first epoch. This is because num_checks get reset after the epoch completes, causing that run to get repeatedly rescheduled till it catches up.

## [v0.12.8]

### Fixes
- Fix issue with 2nd call to Experiment() crashing python process for RAG

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
- Changed default ports for Dispatcher = 8851, MLflow = 8852, Frontend = 8853
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
- Support for TensorBoard, currently only Colab

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

- Upgrade to MLflow 3 
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
- Creates rapidfire_mlflow.db file in RF_DB_PATH or ~/db
- If netcat's nc not found rapidfireai start now falls back to a python port checker
- `rapidfireai start` now tries python3/pip3 if possible then falls back to python/pip

## [v0.9.10]
### Changes
- Updated main name of CLI prog to rapidfireai

## [v0.9.9] - 09-03-2025
### Added
- Initial open source release
