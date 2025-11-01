"""Interface for the database."""

import functools
import os
import sqlite3
import time
from collections.abc import Callable
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
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA foreign_keys=ON;
            """
            _ = self.conn.executescript(pragma_sql)

            self.cursor: sqlite3.Cursor = self.conn.cursor()

        except sqlite3.Error as e:
            raise Exception(f"Failed to initialize database connection: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error during database initialization: {e}") from e

    @staticmethod
    def retry_on_locked(
        max_retries: int = DBConfig.DEFAULT_MAX_RETRIES,
        base_delay: float = DBConfig.DEFAULT_BASE_DELAY,
        max_delay: float = DBConfig.DEFAULT_MAX_DELAY,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to retry operations when database is locked"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e).lower():
                            last_exception = e
                            if attempt < max_retries - 1:
                                # Exponential backoff with jitter
                                delay = min(base_delay * (2**attempt), max_delay)
                                delay += time.time() % 0.1  # Add small jitter
                                time.sleep(delay)
                                continue
                        # Re-raise if it's not a "database is locked" error
                        raise
                # If we get here, all retries failed
                if last_exception:
                    raise last_exception
                else:
                    raise RuntimeError("All retries failed but no exception was captured")

            return wrapper

        return decorator

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

    @retry_on_locked()
    def execute(
        self,
        query: str,
        params: dict[str, Any] | tuple[Any, ...] | None = None,
        fetch: bool = False,
        commit: bool = False,
    ) -> list[Any] | tuple[Any] | None:
        """Execute a query with automatic retry on database locked errors"""
        # Validate that either fetch or commit is True
        if not fetch and not commit:
            raise ValueError("Either fetch or commit must be True")

        try:
            # Execute the query with parameters if provided
            result = self.cursor.execute(query, params) if params else self.cursor.execute(query)

            # Commit the transaction if commit is True
            if commit:
                self.conn.commit()

            # Return the result if fetch is True
            if fetch:
                return result.fetchall()

        except sqlite3.Error as e:
            raise Exception(f"Database error executing query '{query[:50]}...': {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error executing query '{query[:50]}...': {e}") from e
