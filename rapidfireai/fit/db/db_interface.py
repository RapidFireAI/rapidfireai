"""Interface for the database."""

import os
import sqlite3
import time
from typing import Any

from rapidfireai.fit.utils.constants import DBConfig
from rapidfireai.fit.utils.exceptions import DBException


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
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA foreign_keys=ON;
            """
            _ = self.conn.executescript(pragma_sql)

            self.cursor: sqlite3.Cursor = self.conn.cursor()

        except sqlite3.Error as e:
            raise DBException(f"Failed to initialize database connection: {e}") from e
        except Exception as e:
            raise DBException(
                f"Unexpected error during database initialization: {e}"
            ) from e

    def close(self) -> None:
        """Close the database connection properly"""
        try:
            if self.conn:
                self.conn.close()
        except sqlite3.Error as e:
            raise DBException(f"Error closing database connection: {e}") from e
        except Exception as e:
            raise DBException(
                f"Unexpected error closing database connection: {e}"
            ) from e

    def optimize_periodically(self) -> None:
        """Run periodic optimization - call this occasionally, not on every query"""
        try:
            _ = self.conn.execute("PRAGMA optimize")
        except sqlite3.Error as e:
            raise DBException(f"Failed to optimize database: {e}") from e
        except Exception as e:
            raise DBException(
                f"Unexpected error during database optimization: {e}"
            ) from e

    def _execute_once(
        self,
        query: str,
        params: dict[str, Any] | tuple[Any, ...] | None,
        fetch: bool,
        commit: bool,
    ) -> list[Any] | tuple[Any] | None:
        """Execute a query a single time without retry handling."""
        # Execute the query with parameters if provided
        if params:
            result = self.cursor.execute(query, params)
        else:
            result = self.cursor.execute(query)

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
        max_retries: int = DBConfig.DEFAULT_MAX_RETRIES,
        base_delay: float = DBConfig.DEFAULT_BASE_DELAY,
        max_delay: float = DBConfig.DEFAULT_MAX_DELAY,
    ) -> list[Any] | tuple[Any] | None:
        """Execute a query with automatic retry on database locked errors.

        The retry logic lives inside this method (rather than as a decorator)
        because the ``sqlite3.OperationalError`` must be inspected *before* it is
        wrapped into a ``DBException``; otherwise the "database is locked" signal
        is lost and the caller sees an immediate 500.
        """
        # Validate that either fetch or commit is True
        if not fetch and not commit:
            raise ValueError("Either fetch or commit must be True")

        last_locked_exception: sqlite3.OperationalError | None = None
        for attempt in range(max_retries):
            try:
                return self._execute_once(query, params, fetch, commit)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    last_locked_exception = e
                    if attempt < max_retries - 1:
                        # Exponential backoff with a small jitter
                        delay = min(base_delay * (2**attempt), max_delay)
                        delay += time.time() % 0.1
                        time.sleep(delay)
                        continue
                raise DBException(
                    f"Database error executing query '{query[:50]}...': {e}"
                ) from e
            except sqlite3.Error as e:
                raise DBException(
                    f"Database error executing query '{query[:50]}...': {e}"
                ) from e
            except Exception as e:
                raise DBException(
                    f"Unexpected error executing query '{query[:50]}...': {e}"
                ) from e

        # All retries exhausted on a locked database
        raise DBException(
            f"Database error executing query '{query[:50]}...': {last_locked_exception}"
        ) from last_locked_exception
