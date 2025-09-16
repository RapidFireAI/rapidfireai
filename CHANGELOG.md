# Change Log
All notable changes `rapidfireai` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

## [v0.9.11]
### Changes:
* Add option to create a Test PyPI package
* Added logging of rf version number in Experiment class's init method.
* Do not install old version of flash-attn on NVIDIA 7 GPUs
* Add RF_ prefix and accept existing values for MLFLOW_PORT, MLFLOW_HOST, FRONTEND_PORT, FRONTEND_HOST, API_PORT, API_HOST, PID_FILE
* Allow use of RF_DB_PATH for DBConfig.DB_PATH in constants
* Added lite versions of tutorial_notebooks that will run under 2 hours on T4 GPUs
* rapidfireai init Copies tutorial notebooks to RF_TUTORIAL_PATH or ./tutorial_notebooks

### Fixes:
* Added checks in create_warm_start_checkpoint method in shm_manager.py. This method is only triggered during clone-modify warm start operations.
* Creates mlflow.db file in RF_DB_PATH or ~/db
* If netcat's nc not found rapidfireai start now falls back to a python port checker
* rapidfireai start now tries python3/pip3 if possible then falls back to python/pip


## [v0.9.10]

Updated main name of CLI prog to `rapidfireai`

### Changes
* Updated main name of CLI prog to rapidfireai


## [v0.9.9]

Initial open source release

