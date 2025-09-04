"""Gunicorn configuration file for the RapidFire dispatcher"""

from rapidfireai.db.rf_db import RfDb
from rapidfireai.utils.constants import DispatcherConfig

# Other Gunicorn settings...
bind = f"{DispatcherConfig.HOST}:{DispatcherConfig.PORT}"
workers = 2

wsgi_app = "rapidfireai.dispatcher.dispatcher:serve_forever()"


def on_starting(server):
    """
    This function is called once before the master process is initialized.
    We use it to create tables, ensuring this happens only once.
    """
    print("Initializing database tables...")
    try:
        rf_db = RfDb()
        rf_db.create_tables()
        print("Database tables initialized successfully")
    except Exception as e:
        print(f"Error initializing database tables: {e}")
        raise
