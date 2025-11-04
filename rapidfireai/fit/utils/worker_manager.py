"""
WorkerManager class for managing worker processes with PyTorch multiprocessing.
"""

import os
import signal
import sys
import threading
import time
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import Lock

from torch.multiprocessing import Event, Process

from rapidfireai.fit.backend.worker import Worker
from rapidfireai.fit.utils.logging import RFLogger


def worker_process_target(worker_id: int, model_registry: DictProxy, process_lock: Lock, shutdown_event: EventType):
    """
    Target function that runs in each worker process.
    Creates Worker instance inside the process to avoid pickling issues.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    # Create worker instance inside the process (avoids pickling)
    worker = Worker(worker_id, model_registry, process_lock, shutdown_event)

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        if signum == signal.SIGTERM:
            worker.logger.debug(f"Worker {worker.worker_id} received SIGTERM")
            shutdown_event.set()
            # Don't exit immediately, let the main loop handle shutdown

    signal.signal(signal.SIGTERM, signal_handler)

    try:
        worker.serve_forever()
    except KeyboardInterrupt:
        worker.logger.debug(f"Worker {worker.worker_id} interrupted")
    except Exception as e:
        worker.logger.debug(f"Worker {worker.worker_id} crashed: {e}")
    finally:
        worker.logger.info(f"Worker {worker.worker_id} process exiting")
        # Ensure proper cleanup before exit
        try:
            worker.shutdown()
        except Exception as e:
            worker.logger.debug(f"Error during worker shutdown: {e}")
        # Use sys.exit instead of os._exit for better cleanup
        sys.exit(0)


class WorkerManager:
    """
    WorkerManager class for managing worker processes.
    """

    def __init__(
        self,
        num_workers: int,
        model_registry: DictProxy | None = None,
        process_lock: Lock | None = None,
        parent_check_interval: float = 1.0,
    ):
        self.num_workers = num_workers
        self.parent_check_interval = parent_check_interval
        self.parent_pid = os.getppid()
        self.process_group_id: int | None = None
        self.processes: list[Process] = []
        self.shutdown_events: list[EventType] = []
        self.shutdown_event = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.model_registry: DictProxy = model_registry
        self.process_lock: Lock = process_lock

        self.logger = RFLogger().create_logger("worker_manager")

    def _parent_monitor(self):
        """Monitor parent process and kill process group if parent dies"""
        while not self.shutdown_event.is_set():
            try:
                # Check if parent process is still alive
                try:
                    os.getpgid(self.parent_pid)
                except PermissionError:
                    # Fallback for restricted environments (e.g., Colab)
                    # Use psutil to check if parent process exists
                    import psutil

                    if not psutil.pid_exists(self.parent_pid):
                        raise ProcessLookupError("Parent process no longer exists")
                time.sleep(self.parent_check_interval)
            except ProcessLookupError:
                self.logger.debug(f"Parent process {self.parent_pid} died, shutting down workers...")
                self.shutdown()
                break
            except Exception as e:
                self.logger.error(f"Parent monitor error: {e}")
                time.sleep(self.parent_check_interval)

    def create_workers(self) -> list[int]:
        """
        Create worker processes and return worker IDs.
        Returns list of worker IDs to avoid pickling Worker objects.
        """
        self.logger.debug(f"Creating {self.num_workers} worker processes...")

        # Create new process group (may not be permitted in restricted environments like Colab)
        try:
            os.setpgrp()
            self.process_group_id = os.getpgrp()
            self.logger.debug(f"Starting worker processes in process group {self.process_group_id}")
        except PermissionError:
            self.logger.debug(
                "Cannot create process group (restricted environment) - will use individual process termination"
            )
            self.process_group_id = None

        worker_ids = []

        # Start each worker process
        for i in range(self.num_workers):
            # Create shutdown event for this worker
            shutdown_event = Event()
            self.shutdown_events.append(shutdown_event)

            # Create process - pass only picklable arguments
            process = Process(
                target=worker_process_target, args=(i, self.model_registry, self.process_lock, shutdown_event)
            )
            process.daemon = False
            process.start()

            self.processes.append(process)
            worker_ids.append(i)

            self.logger.debug(f"Started Worker {i} with PID {process.pid}")

        # Start parent monitoring thread
        self.monitor_thread = threading.Thread(target=self._parent_monitor, daemon=True)
        self.monitor_thread.start()

        msg = f"Started {self.num_workers} worker processes successfully"
        self.logger.info(msg)
        print(msg)

        return worker_ids

    def shutdown(self, timeout: float = 10.0):
        """Shutdown all workers gracefully, then force kill if needed"""
        self.logger.debug("WorkerManager shutdown initiated...")
        self.shutdown_event.set()

        # Step 1: Signal all workers to shutdown gracefully
        self.logger.debug("Signaling graceful shutdown...")
        for shutdown_event in self.shutdown_events:
            shutdown_event.set()

        # Wait for processes to finish gracefully
        start_time = time.time()
        all_finished = True

        for i, process in enumerate(self.processes):
            if process:
                remaining_timeout = max(0, timeout - (time.time() - start_time))
                if remaining_timeout > 0:
                    process.join(timeout=remaining_timeout)

                if process.is_alive():
                    self.logger.debug(f"Worker {i} did not shutdown gracefully")
                    all_finished = False

        if all_finished:
            self.logger.debug("All workers shutdown gracefully")
            return

        # Step 2: Send SIGTERM to remaining processes
        self.logger.debug("Sending SIGTERM to remaining workers...")
        for i, process in enumerate(self.processes):
            if process and process.is_alive():
                try:
                    process.terminate()  # Sends SIGTERM
                except Exception as e:
                    self.logger.debug(f"Error sending SIGTERM to Worker {i}: {e}")

        # Wait a bit more
        time.sleep(2)

        # Step 3: Force kill any remaining processes
        remaining_alive = False
        for i, process in enumerate(self.processes):
            if process and process.is_alive():
                remaining_alive = True
                try:
                    process.kill()  # SIGKILL
                    process.join(timeout=1.0)
                except Exception as e:
                    self.logger.debug(f"Error force killing Worker {i}: {e}")

        if remaining_alive:
            # Nuclear option - kill entire process group
            self.logger.debug("Using process group kill as final resort...")
            if self.process_group_id:
                try:
                    os.killpg(self.process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process group already dead
                except Exception as e:
                    self.logger.error(f"Error killing process group: {e}")

        self.logger.debug("WorkerManager shutdown complete")

    def get_worker_status(self) -> dict:
        """Get status of all workers"""
        status = {}
        for i, process in enumerate(self.processes):
            status[i] = {
                "alive": process.is_alive() if process else False,
                "pid": process.pid if process else None,
            }
        return status

    def __del__(self):
        """Cleanup when WorkerManager is destroyed"""
        try:
            self.shutdown(timeout=5.0)
        except:
            pass
