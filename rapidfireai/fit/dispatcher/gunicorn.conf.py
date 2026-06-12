"""Gunicorn configuration file for the RapidFire dispatcher"""

from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.utils.constants import DispatcherConfig

# Bind address
bind = f"{DispatcherConfig.HOST}:{DispatcherConfig.PORT}"

# Worker / timeout knobs come from DispatcherConfig (env-var overridable;
# see rapidfireai/utils/constants.py for the full rationale and the
# RF_API_* variables that override each default). See the evals dispatcher
# gunicorn.conf.py for a verbose breakdown of why each default was picked.
workers = DispatcherConfig.WORKERS
worker_class = DispatcherConfig.WORKER_CLASS
threads = DispatcherConfig.THREADS
timeout = DispatcherConfig.TIMEOUT
graceful_timeout = DispatcherConfig.GRACEFUL_TIMEOUT
keepalive = DispatcherConfig.KEEPALIVE

wsgi_app = "rapidfireai.fit.dispatcher.dispatcher:serve_forever()"


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
        rf_db = RfDb()
        rf_db.create_tables()
        print("Database tables initialized successfully", flush=True)
    except Exception as e:
        print(f"Error initializing database tables: {e}", flush=True)
        raise
