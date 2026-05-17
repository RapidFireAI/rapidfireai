# Developer Guide — Code Flow

A walkthrough of how RapidFire AI executes a training experiment end-to-end. Written for engineers landing in the repo for the first time. All file paths are relative to the repo root and citations use `path:line` so you can jump straight to the code.

---

## 1. Repo Layout (what's actually there)

```
rapidfireai/
├── experiment.py              # User-facing Experiment class (run_fit, get_results)
├── cli.py                     # rapidfireai start / stop / init / doctor
├── version.py
├── automl/                    # GridSearch, RandomSearch — expand a config into runs
├── fit/                       # Training pipeline (this guide focuses here)
│   ├── backend/               # Controller, Scheduler, Worker, Chunks
│   ├── db/                    # SQLite layer + tables.sql
│   ├── dispatcher/            # Flask REST API (IC Ops + UI reads)
│   ├── ml/                    # Trainer, callbacks, checkpoint utils
│   └── utils/                 # SHM, worker manager, IC controller, logging
├── evals/                     # Evaluation pipeline (actors, RAG, metrics) — separate
├── frontend/                  # React dashboard (MLflow fork + IC Ops panel)
└── utils/                     # CLI-side helpers (doctor, gpu_info, colab, ping)
```

> Note: an older version of `CLAUDE.md` references `backend/`, `db/`, `ml/`, etc. at the package root. They live under `fit/` now.

---

## 2. The 30-second Mental Model

1. The user calls `Experiment.run_fit(...)` from a notebook.
2. A **Controller** runs in that user process. It expands configs into runs, persists them to SQLite, and spawns one **Worker** subprocess per GPU.
3. A **Scheduler** decides which `(run_id, chunk_id)` should go to which worker(s) next, using Monte-Carlo simulated fairness.
4. **Workers** poll the DB, train one chunk, save a checkpoint (shared memory by default, disk on the last chunk), and report back.
5. The **Dispatcher** (Flask, `:8851`) exposes a REST API the React frontend uses to read state and submit **IC Ops** (Stop / Resume / Clone-Modify / Delete). Ops are written to the DB; the Controller polls and applies them between scheduling ticks.
6. **MLflow** (`:8852`) stores metrics/params/artifacts. The frontend (`:8853`) is a fork of MLflow's UI with an IC Ops panel bolted on.

---

## 3. End-to-End Walkthrough: `Experiment.run_fit()`

### 3.1 User entry

`rapidfireai/experiment.py:355` — `Experiment.run_fit(param_config, create_model_fn, train_dataset, eval_dataset, num_chunks, num_gpus, seed, ...)`.

Constructs a `Controller` at `experiment.py:446` and hands it the datasets + config.

### 3.2 Controller initialization

`fit/backend/controller.py:44` — `Controller.__init__`:
- Detects GPUs via `torch.cuda.device_count()` → sets `num_workers`.
- Builds a `SharedMemoryManager` (controller.py:73) — a `multiprocessing.Manager` dict + lock used as the cross-process model registry.
- Builds a `WorkerManager` (controller.py:79) for spawning/teardown.
- Initializes the MLflow `MetricLogger` (controller.py:83).

### 3.3 Persisting datasets and runs

- Datasets are serialized once and saved to disk (controller.py:598). Workers will read them per-task via `decode_db_payload()` (worker.py:185).
- `Controller._create_models()` (controller.py:97) expands `param_config` (plain dict or AutoML object) into individual run rows in the `runs` table. Each gets a UUID, a `config_leaf` BLOB, `total_steps`, and `status=NEW`. An MLflow run is created in parallel and its ID stored on the row (controller.py:188).
- The experiment's current task is set to `RUN_FIT` (controller.py:226).

### 3.4 Spawning workers

`WorkerManager.create_workers()` (`fit/utils/worker_manager.py:110`) spawns one `multiprocessing.Process` per GPU. Each subprocess runs `worker_process_target` (worker_manager.py:20) which constructs a `Worker` and enters `Worker.serve_forever()` (`fit/backend/worker.py:648`).

Workers receive the same SHM registry + lock so checkpoints can be passed in-memory.

### 3.5 Building the Scheduler

`fit/backend/scheduler.py` — the `Scheduler` is constructed with the list of `NEW` run IDs, per-run required GPUs, and `num_chunks`. Internal state:

- `worker_running_current_run` — worker → run mapping (`-1` = idle).
- `run_visited_num_chunks` — per-run progress counter.
- `run_estimated_runtime` — initial random estimates, updated as chunks complete.

### 3.6 The main loop (Controller)

`Controller.run_fit()` (controller.py:576) loops until every run is `COMPLETED` or `FAILED`. Each tick does:

1. **Reap finished tasks** (controller.py:676) — read `worker_task` rows. For `COMPLETED`: bump `run_visited_num_chunks`, possibly mark the run complete (all chunks done AND `completed_steps >= total_steps`) or "epoch boundary" (chunks done, more epochs to go). For `FAILED`: drop the run from the scheduler and clear its SHM entries via `_clear_run_from_shm()` (controller.py:231).
2. **Apply IC Ops** (controller.py:792) — `_process_interm_ic_ops_states()` polls the `interactive_control` table and dispatches to `_process_interactive_control()` (controller.py:265):
   - **STOP** → mark run `STOPPED`, free workers.
   - **RESUME** → mark `ONGOING`, re-add to scheduler.
   - **DELETE** → mark `DELETED`, delete from MLflow.
   - **CLONE_MODIFY** → call `_create_models()` again with parent config + overrides.
   - **CLONE_MODIFY_WARM** → same, but seeds the new run from the parent's checkpoint.
3. **Pick next assignment** (controller.py:841) — `Scheduler.schedule()` (scheduler.py:364) runs a Monte-Carlo simulation across candidate `(run, workers, chunk)` triples and returns the best. Sentinel returns: `run_id=-1` means no eligible workers right now; `run_id=None` means everything's done.
4. **Dispatch to workers** (controller.py:862) — flips the run to `ONGOING` and calls `db.create_worker_task(...)` for each worker in the assignment, with `WorkerTask.TRAIN_VAL`, `TaskStatus.SCHEDULED`, multi-worker rendezvous info (`world_size`, `master_address`, `master_port`), and the pickled `create_model_fn`.
5. **Sleep briefly** (controller.py:900), repeat.

When the loop exits, the experiment task is set to `IDLE` (controller.py:903) and `WorkerManager.shutdown()` signals workers to terminate.

### 3.7 What workers do

`Worker.serve_forever()` (worker.py:648) is a tight poll loop:

1. `db.get_worker_scheduled_task(worker_id)` (worker.py:654). On miss, sleep 1s.
2. Pull `run_id`, `chunk_id`, multi-worker rendezvous info, and `create_model_fn` (worker.py:660).
3. Mark task `IN_PROGRESS` (worker.py:670).
4. Call `Worker.run_fit()` (worker.py:197):
   - Load the run's `config_leaf` (worker.py:210).
   - If FSDP, set `CUDA_VISIBLE_DEVICES` and init `torch.distributed` with the rendezvous info (worker.py:237).
   - Build a `DatasetChunks` (`fit/backend/chunks.py:8`) over the train dataset and pull the slice for this `chunk_id` (chunks.py:99). Chunks are aligned to the effective batch size.
   - Build the trainer via `create_trainer_instance()` in `fit/ml/trainer.py`. If this isn't the first chunk for the run, the model is loaded from SHM (or disk fallback).
   - `trainer.train()` runs one pass over the chunk (worker.py:386).
   - `db.set_completed_steps(run_id, ...)` (worker.py:393).
   - **Checkpoint policy** (worker.py:408): SHM for intermediate chunks; disk on the final chunk or when `save_strategy="chunk"`.
   - GPU cleanup: drop trainer, clear CUDA cache, GC.
5. Mark task `COMPLETED` (worker.py:679) and loop.

### 3.8 Retrieving results

`Experiment.get_results()` (experiment.py:558) fetches each run's `metric_run_id` from the DB and pulls timeseries metrics from MLflow into a pandas DataFrame.

---

## 4. Database Schema (the source of truth for coordination)

Defined in `fit/db/tables.sql`, accessed via `fit/db/rf_db.py`. The tables you'll see referenced everywhere:

| Table | Role |
|---|---|
| `experiments` | One row per experiment. Holds `current_task`, `status`, error info. |
| `runs` | One row per training run. `config_leaf` (BLOB), `status`, `completed_steps`, `total_steps`, `metric_run_id` (MLflow), parent linkage for clones. |
| `worker_task` | Queue between Controller and Workers. `(worker_id, run_id, chunk_id, status, multi_worker_details)`. |
| `interactive_control` | IC Ops queue. `(run_id, ic_op, status, config_leaf)` — written by Dispatcher, drained by Controller. |
| `controller_progress`, `worker_progress` | Liveness/progress counters. |

There are no long-held connections between Controller ↔ Worker — coordination is entirely through these tables.

---

## 5. Shared Memory (Why You Won't Find a Pickle on Disk Between Chunks)

`fit/utils/shm_manager.py` wraps a `multiprocessing.Manager().dict()` plus a process lock. Workers `put` checkpoints under a `(run_id, chunk_id)` key after each non-final chunk; the next chunk's worker pulls the same key. This is what makes chunk-based interleaving cheap — without it you'd be paying disk-write + disk-read per chunk per run.

`USE_SHARED_MEMORY=False` (env) forces the disk fallback path. Disk is also used for the final chunk so the artifact survives shutdown.

---

## 6. Interactive Control — The Loop

The IC Ops panel in the frontend is the differentiating feature. The cycle:

1. User clicks **Stop** (or Resume/Clone/Delete) on a run in the React UI.
2. Frontend POSTs to the Dispatcher (`fit/dispatcher/dispatcher.py`, e.g. `stop_run` at dispatcher.py:105).
3. Dispatcher inserts an `interactive_control` row.
4. The Controller's next loop iteration drains it (controller.py:419) and applies the operation — which may mean updating run status, freeing workers, calling `_create_models()` again for a clone, or deleting from MLflow.
5. Worker behavior changes on its next poll because the DB state has changed (e.g. its run is now `STOPPED`, so it gets no new tasks for it).

Important: nothing kills a worker mid-chunk. Stops take effect at the next chunk boundary. Don't add code that assumes synchronous IC.

---

## 7. Process Topology

```
user notebook (python)
└─ Experiment
   └─ Controller (main loop)              ← in-process, owns SHM + scheduler
      ├─ Worker subprocess (GPU 0)        ← spawned by WorkerManager
      ├─ Worker subprocess (GPU 1)
      └─ ...

separately, started by `rapidfireai start` (cli.py):
- Dispatcher (gunicorn @ :8851)           ← reads/writes the same SQLite DB
- MLflow server (@ :8852)
- Frontend dev/static server (@ :8853)
```

The user-process side and the server-process side share state only through the SQLite DB and the MLflow store. There is no socket or IPC between them.

---

## 8. CLI Surface

`rapidfireai/cli.py` wires up:

- `init` — pull tutorial notebooks, install Jupyter kernel, basic env checks.
- `start` — boot dispatcher, MLflow, frontend (`fit/start.sh` / `start_dev.sh`).
- `stop` — kill the three services.
- `doctor` — `rapidfireai/utils/doctor.py` runs CUDA / GPU / Python diagnostics.

The CLI does **not** start the Controller — that's instantiated by `Experiment.run_fit()` inside whatever process the user is running.

---

## 9. Where to Start When You're Debugging

| Symptom | First place to look |
|---|---|
| Run never starts | `runs` table status, then `worker_task` rows. Controller scheduling tick logs. |
| Worker hung | Worker logs (`worker_{N}` logger), then `worker_task` row status. |
| IC Op did nothing | `interactive_control` row status; Controller log for `_process_interactive_control` errors. |
| Checkpoint missing on next chunk | SHM registry contents; check `USE_SHARED_MEMORY` and disk fallback path. |
| Metrics not in UI | MLflow run ID on the `runs` row, then MLflow server logs. |
| Port conflict | `lsof -t -i:8851` / `:8852` / `:8853`, kill stale processes. |

---

## 10. Conventions

- **Python 3.12** required (pyproject.toml).
- **Ruff** with `line-length: 120` (`ruff.toml`). Run `ruff format . && ruff check --fix .` before pushing.
- **Tests**: `pytest tests/`. The fit pipeline tests live alongside `tests/test_chunks.py` etc.
- **Multiprocessing start method**: `spawn`. Notebooks run fine; pure-CLI scripts hit the usual `if __name__ == "__main__"` requirements.
- **Logging**: loguru, with separate sinks per component (`experiment`, `controller`, `worker_{N}`, `user`, `interactive-control`). Files land in the experiment directory.

---

## 11. Code Flow Diagram

See [`RapidFire-CodeFlow.png`](./images/RapidFire-CodeFlow.png) for complete flow daigram.
See [`code-flow.md`](./code-flow.md) for the full mermaid diagram of the same flow described above.
