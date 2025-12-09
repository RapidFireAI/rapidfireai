"""Gunicorn configuration file for the RapidFire Evals dispatcher"""

from rapidfireai.evals.db.rf_db import RFDatabase
from rapidfireai.utils.constants import DispatcherConfig

# Gunicorn settings
bind = f"{DispatcherConfig.HOST}:{DispatcherConfig.PORT}"
workers = 1  # Single worker for Colab/single-user environments to save memory

wsgi_app = "rapidfireai.evals.dispatcher.dispatcher:serve_forever()"


def on_starting(server):
    """
    This function is called once before the master process is initialized.
    We use it to create tables, ensuring this happens only once.
    """
    print("Initializing Evals database tables...")
    try:
        rf_db = RFDatabase()
        rf_db.create_tables()
        print("Evals database tables initialized successfully")
    except Exception as e:
        print(f"Error initializing Evals database tables: {e}")
        raise
