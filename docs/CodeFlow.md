# RapidFire AI — Code Flow

```mermaid
flowchart TB
    %% User Entry
    User([User])
    Notebook[Jupyter Notebook / Python Script]
    User --> Notebook

    %% Top-level API
    subgraph API["User-Facing API (experiment.py)"]
        Exp[Experiment]
        RunFit["run_fit(param_config,<br/>create_model_fn,<br/>train/eval datasets,<br/>num_chunks)"]
        GetResults["get_results()"]
        Exp --> RunFit
        Exp --> GetResults
    end
    Notebook --> Exp

    %% AutoML expansion
    subgraph AutoML["automl/"]
        Grid[GridSearch]
        Random[RandomSearch]
    end
    RunFit -. expands configs .-> AutoML

    %% Controller
    subgraph Ctrl["Controller (backend/controller.py)"]
        CreateModels[Create Models<br/>from configs]
        SpawnWorkers[Spawn Worker<br/>processes per GPU]
        SchedLoop[Scheduling Loop]
        ICHandler[IC Ops Handler]
        Monitor[Progress Monitor]
    end
    RunFit --> Ctrl
    CreateModels --> SpawnWorkers --> SchedLoop

    %% Scheduler
    subgraph Sched["Scheduler (backend/scheduler.py)"]
        Assign["Assign (run_id, chunk_id)<br/>round-robin + fairness"]
    end
    SchedLoop --> Assign

    %% Database
    subgraph DB["SQLite DB (db/rf_db.py)"]
        TExp[(experiments)]
        TRuns[(runs)]
        TTasks[(tasks:<br/>Experiment/Controller/Worker)]
        TCkpt[(checkpoints)]
    end
    Ctrl <--> DB
    Assign -- writes WorkerTask --> TTasks

    %% Workers
    subgraph Workers["Workers (backend/worker.py)<br/>separate GPU processes"]
        Poll[Poll DB for tasks]
        LoadCkpt[Load checkpoint<br/>from SHM or disk]
        Train[Train on chunk<br/>ml/trainer.py]
        SaveCkpt[Save checkpoint]
        Report[Report status]
        Poll --> LoadCkpt --> Train --> SaveCkpt --> Report --> Poll
    end
    SpawnWorkers --> Workers
    Poll <--> TTasks

    %% Shared Memory
    subgraph SHM["Shared Memory (utils/shm_manager.py)"]
        Registry[Model Registry]
        Locks[Process Locks]
        DiskFallback[Disk Fallback]
    end
    LoadCkpt <--> SHM
    SaveCkpt <--> SHM

    %% Chunks
    subgraph Data["Chunks (backend/chunks.py)"]
        Split[Split dataset<br/>into N chunks]
    end
    RunFit --> Split
    Split -. chunk refs .-> Train

    %% MLflow
    subgraph ML["MLflow (utils/mlflow_manager.py)"]
        MLServer[MLflow Server<br/>:8852]
        Metrics[(metrics, params,<br/>artifacts)]
        MLServer --> Metrics
    end
    Report --> MLServer
    Ctrl --> MLServer

    %% Dispatcher / Frontend
    subgraph UI["UI Layer"]
        Dispatcher[Dispatcher Flask API<br/>dispatcher/dispatcher.py<br/>:8851]
        Frontend[React Dashboard<br/>frontend/<br/>:8853]
    end
    Frontend <--> Dispatcher
    Frontend <--> MLServer
    Dispatcher <--> DB

    %% IC Ops Loop
    UserOps([User clicks<br/>Stop/Resume/Clone/Delete])
    UserOps --> Frontend
    Dispatcher -- writes IC Op state --> TRuns
    ICHandler -- polls --> TRuns
    ICHandler --> SchedLoop

    %% Results
    GetResults --> DB
    GetResults --> MLServer
    GetResults --> User

    %% CLI
    subgraph CLI["CLI (cli.py)"]
        StartCmd["rapidfireai start<br/>(start.sh)"]
        StopCmd["rapidfireai stop"]
        InitCmd["rapidfireai init"]
        DoctorCmd["rapidfireai doctor"]
    end
    User --> CLI
    StartCmd --> Dispatcher
    StartCmd --> MLServer
    StartCmd --> Frontend

    %% Styling
    classDef userClass fill:#e1f5ff,stroke:#0288d1,color:#000
    classDef ctrlClass fill:#fff4e1,stroke:#f57c00,color:#000
    classDef workerClass fill:#e8f5e9,stroke:#388e3c,color:#000
    classDef dbClass fill:#fce4ec,stroke:#c2185b,color:#000
    classDef uiClass fill:#f3e5f5,stroke:#7b1fa2,color:#000

    class User,Notebook,UserOps userClass
    class Ctrl,Sched,AutoML ctrlClass
    class Workers,SHM,Data workerClass
    class DB,ML dbClass
    class UI,API,CLI uiClass
```

## Flow Summary

1. **Entry** — User invokes `Experiment.run_fit()` from a notebook; AutoML expands param configs into multiple runs.
2. **Controller** spawns Worker processes (one per GPU) and runs the scheduling loop.
3. **Scheduler** assigns `(run_id, chunk_id)` pairs to workers via the `tasks` table using round-robin + fairness.
4. **Workers** poll the DB, load checkpoints from shared memory (disk fallback), train on the chunk, save the checkpoint back, and report metrics.
5. **Shared Memory** avoids disk I/O between chunks.
6. **MLflow** records metrics/params/artifacts; served on `:8852`.
7. **UI Layer** — Dispatcher (`:8851`) + React frontend (`:8853`) read from DB and MLflow; user IC Ops (Stop/Resume/Clone/Delete) write state changes back into the DB, which the Controller polls.
8. **CLI** (`rapidfireai start`) boots dispatcher, MLflow, and frontend.
