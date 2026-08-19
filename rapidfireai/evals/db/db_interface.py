"""Interface for the database."""

import os
import sqlite3
from typing import Any

from rapidfireai.evals.utils.constants import DBConfig


class DatabaseInterface:
    """Interface for the database."""

    def __init__(self):
        try:
            if not os.path.exists(DBConfig.DB_PATH):
                path = os.path.dirname(DBConfig.DB_PATH)
                os.makedirs(path, exist_ok=True)
                print(f"Created directory for database at {path}")

            self.conn: sqlite3.Connection = sqlite3.connect(
                DBConfig.DB_PATH,
                timeout=DBConfig.CONNECTION_TIMEOUT,
                check_same_thread=False,
                isolation_level=None,
            )

            # Configure database with all PRAGMA settings
            pragma_sql = f"""
            PRAGMA cache_size={DBConfig.CACHE_SIZE};
            PRAGMA mmap_size={DBConfig.MMAP_SIZE};
            PRAGMA page_size={DBConfig.PAGE_SIZE};
            PRAGMA busy_timeout={DBConfig.BUSY_TIMEOUT};
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA foreign_keys=ON;
            PRAGMA wal_autocheckpoint={DBConfig.WAL_AUTO_CHECKPOINT};
            """
            _ = self.conn.executescript(pragma_sql)

            self.cursor: sqlite3.Cursor = self.conn.cursor()

        except sqlite3.Error as e:
            raise Exception(f"Failed to initialize database connection: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error during database initialization: {e}") from e

    def close(self) -> None:
        """Close the database connection properly"""
        try:
            if self.conn:
                self.conn.close()
        except sqlite3.Error as e:
            raise Exception(f"Error closing database connection: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error closing database connection: {e}") from e

    def optimize_periodically(self) -> None:
        """Run periodic optimization - call this occasionally, not on every query"""
        try:
            _ = self.conn.execute("PRAGMA optimize")
        except sqlite3.Error as e:
            raise Exception(f"Failed to optimize database: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error during database optimization: {e}") from e

    def checkpoint(self) -> None:
        """Run a PASSIVE WAL checkpoint. Non-blocking: checkpoints as many
        frames as possible without waiting on readers/writers. Safe to call
        from the hot writer to bound WAL growth so auto-checkpoint doesn't
        ambush a burst."""
        try:
            _ = self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error as e:
            raise Exception(f"Failed to checkpoint WAL: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error during WAL checkpoint: {e}") from e

    def _execute_once(
        self,
        query: str,
        params: dict[str, Any] | tuple[Any, ...] | None,
        fetch: bool,
        commit: bool,
    ) -> list[Any] | tuple[Any] | None:
        """Execute a query a single time without retry handling."""
        # Execute the query with parameters if provided
        result = self.cursor.execute(query, params) if params else self.cursor.execute(query)

        # Commit the transaction if commit is True
        if commit:
            self.conn.commit()

        # Return the result if fetch is True
        if fetch:
            return result.fetchall()
        return None

    def execute(
        self,
        query: str,
        params: dict[str, Any] | tuple[Any, ...] | None = None,
        fetch: bool = False,
        commit: bool = False,
    ) -> list[Any] | tuple[Any] | None:
        """Execute a query a single time.

        Lock-waiting is handled entirely by SQLite's ``busy_timeout`` PRAGMA
        (set on the connection): on a locked database SQLite blocks and retries
        the lock internally for up to ``BUSY_TIMEOUT`` ms, succeeding the moment
        the lock frees. There is no Python-level retry on top — that would just
        stack more ``busy_timeout`` windows and add latency. A write that still
        fails after ``busy_timeout`` is raised as a wrapped ``Exception``.
        """
        if not fetch and not commit:
            raise ValueError("Either fetch or commit must be True")

        try:
            return self._execute_once(query, params, fetch, commit)
        except sqlite3.OperationalError as e:
            raise Exception(f"Database error executing query '{query[:50]}...': {e}") from e
        except sqlite3.Error as e:
            raise Exception(f"Database error executing query '{query[:50]}...': {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error executing query '{query[:50]}...': {e}") from e
