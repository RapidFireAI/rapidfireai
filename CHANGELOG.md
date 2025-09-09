# Change Log
All notable changes `rapidfireai` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

## [v0.9.11]

`rapidfireai` command fixes and improvements

### Fixes
* Creates mlflow.db file in RF_DB_PATH or `~/db`
* If netcat's `nc` not found `rapidfireai start` now falls back to a python port checker
* `rapidfireai start` now tries python3/pip3 if possible then falls back to python/pip

### Changes
* `rapidfireai init` Copies tutorial notebooks to RF_TUTORIAL_PATH or `./tutorial_notebooks`

## [v0.9.10]

Updated main name of CLI prog to `rapidfireai`

### Changes
* Updated main name of CLI prog to rapidfireai


## [v0.9.9]

Initial open source release

