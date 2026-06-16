"""Gunicorn configuration file for the RapidFire dispatcher"""

from rapidfireai.evals.db import RFDatabase
from rapidfireai.utils.constants import DispatcherConfig

# Bind address
bind = f"{DispatcherConfig.HOST}:{DispatcherConfig.PORT}"

# Worker / timeout knobs come from DispatcherConfig (env-var overridable;
# see rapidfireai/utils/constants.py for the full rationale and the
# RF_API_* variables that override each default).
#
# Defaults at a glance:
#   * workers=1, worker_class="gthread", threads=4 -- one process keeps the
#     SQLite footprint low; `gthread` is required because polling clients
#     hold long-lived keep-alive TCP connections (otherwise the `sync`
#     worker blocks in `sock.recv()` and the master SIGKILLs it with
#     WORKER TIMEOUT / "no URI read" -- the original api.log crash).
#   * timeout=120 -- safety margin above gunicorn's 30s default for the
#     first-request SQLite migrations on a cold start.
#   * keepalive=5 -- short enough that a misbehaving client cannot tie up
#     a thread, long enough for normal browser polling.
workers = DispatcherConfig.WORKERS
worker_class = DispatcherConfig.WORKER_CLASS
threads = DispatcherConfig.THREADS
timeout = DispatcherConfig.TIMEOUT
graceful_timeout = DispatcherConfig.GRACEFUL_TIMEOUT
keepalive = DispatcherConfig.KEEPALIVE

wsgi_app = "rapidfireai.evals.dispatcher.dispatcher:serve_forever()"


def on_starting(server):
    """
    This function is called once before the master process is initialized.
    We use it to create tables, ensuring this happens only once.
    """
    # `flush=True` is important: stdout is block-buffered when redirected
    # to a log file (e.g. `gunicorn ... > api.log`). Without an explicit
    # flush, this line stays in the master's stdio buffer, gets inherited
    # by every forked worker, and is re-flushed each time a worker exits,
    # making it look in api.log like the database is being re-initialized
    # on every worker restart.
    print("Initializing database tables...", flush=True)
    try:
        rf_db = RFDatabase()
        rf_db.create_tables()
        print("Database tables initialized successfully", flush=True)
    except Exception as e:
        print(f"Error initializing database tables: {e}", flush=True)
        raise
