#!/usr/bin/env python3
"""
RapidFire AI startup script for Google Colab environments.

This script starts the three RapidFire services (MLflow, Dispatcher, Frontend)
and exposes them using Colab's native port forwarding or optional tunneling.
"""

import os
import subprocess
import sys
import time
import signal
from typing import List, Optional

# Try to import colab_helper
try:
    from rapidfireai.utils.colab_helper import (
        is_colab,
        get_notebook_environment,
        expose_rapidfire_services
    )
except ImportError:
    print("Warning: Could not import colab_helper. Installing rapidfireai package...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
    from rapidfireai.utils.colab_helper import (
        is_colab,
        get_notebook_environment,
        expose_rapidfire_services
    )


# Configuration
RF_MLFLOW_PORT = int(os.getenv('RF_MLFLOW_PORT', 5002))
RF_MLFLOW_HOST = os.getenv('RF_MLFLOW_HOST', '0.0.0.0')  # Bind to all interfaces for tunneling
RF_FRONTEND_PORT = int(os.getenv('RF_FRONTEND_PORT', 3000))
RF_FRONTEND_HOST = os.getenv('RF_FRONTEND_HOST', '0.0.0.0')
RF_API_PORT = int(os.getenv('RF_API_PORT', 8080))
RF_API_HOST = os.getenv('RF_API_HOST', '0.0.0.0')
RF_DB_PATH = os.getenv('RF_DB_PATH', os.path.expanduser('~/db'))
RF_TUNNEL_METHOD = os.getenv('RF_TUNNEL_METHOD', 'native')  # 'native', 'cloudflare', or 'ngrok'
RF_NGROK_TOKEN = os.getenv('RF_NGROK_TOKEN')


class ServiceManager:
    """Manages RapidFire service processes."""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down RapidFire services...")
        self.cleanup()
        sys.exit(0)

    def start_mlflow(self) -> bool:
        """Start MLflow server."""
        start_time = time.time()
        print(f"🚀 Starting MLflow server on {RF_MLFLOW_HOST}:{RF_MLFLOW_PORT}...")

        # Create database directory
        os.makedirs(RF_DB_PATH, exist_ok=True)

        try:
            proc = subprocess.Popen(
                [
                    'mlflow', 'server',
                    '--host', RF_MLFLOW_HOST,
                    '--port', str(RF_MLFLOW_PORT),
                    '--backend-store-uri', f'sqlite:///{RF_DB_PATH}/mlflow.db'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            self.processes.append(proc)

            # Wait for service to be ready
            if self._wait_for_port(RF_MLFLOW_PORT, 'MLflow', timeout=30):
                elapsed = time.time() - start_time
                print(f"✅ MLflow server started (PID: {proc.pid}) [took {elapsed:.1f}s]")
                return True
            else:
                elapsed = time.time() - start_time
                print(f"❌ MLflow server failed to start [took {elapsed:.1f}s]")
                return False

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error starting MLflow: {e} [took {elapsed:.1f}s]")
            return False

    def start_dispatcher(self) -> bool:
        """Start Dispatcher API server with Gunicorn."""
        start_time = time.time()
        print(f"🚀 Starting Dispatcher API on {RF_API_HOST}:{RF_API_PORT}...")

        try:
            # Get the dispatcher directory
            import rapidfireai
            package_dir = os.path.dirname(rapidfireai.__file__)
            dispatcher_dir = os.path.join(package_dir, 'dispatcher')

            # Ensure dispatcher directory exists
            if not os.path.exists(dispatcher_dir):
                print(f"❌ Dispatcher directory not found: {dispatcher_dir}")
                return False

            # Start Gunicorn
            proc = subprocess.Popen(
                [
                    'gunicorn',
                    '-c', os.path.join(dispatcher_dir, 'gunicorn.conf.py'),
                    '--bind', f'{RF_API_HOST}:{RF_API_PORT}',
                    'rapidfireai.dispatcher.dispatcher:Dispatcher().app'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=dispatcher_dir
            )

            self.processes.append(proc)

            # Wait for service to be ready
            if self._wait_for_port(RF_API_PORT, 'Dispatcher', timeout=60):
                elapsed = time.time() - start_time
                print(f"✅ Dispatcher API started (PID: {proc.pid}) [took {elapsed:.1f}s]")
                return True
            else:
                elapsed = time.time() - start_time
                print(f"❌ Dispatcher API failed to start [took {elapsed:.1f}s]")
                return False

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error starting Dispatcher: {e} [took {elapsed:.1f}s]")
            return False

    def start_frontend(self) -> bool:
        """Start Frontend Flask server."""
        start_time = time.time()
        print(f"🚀 Starting Frontend server on {RF_FRONTEND_HOST}:{RF_FRONTEND_PORT}...")

        try:
            # Get the frontend directory
            import rapidfireai
            package_dir = os.path.dirname(rapidfireai.__file__)
            frontend_dir = os.path.join(package_dir, 'frontend')

            # Ensure frontend directory exists
            if not os.path.exists(frontend_dir):
                print(f"❌ Frontend directory not found: {frontend_dir}")
                return False

            # Check if server.py exists
            server_py = os.path.join(frontend_dir, 'server.py')
            if not os.path.exists(server_py):
                print(f"❌ Frontend server.py not found: {server_py}")
                return False

            # Start Flask server
            proc = subprocess.Popen(
                [sys.executable, 'server.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=frontend_dir,
                env={**os.environ, 'PORT': str(RF_FRONTEND_PORT), 'HOST': RF_FRONTEND_HOST}
            )

            self.processes.append(proc)

            # Wait for service to be ready
            if self._wait_for_port(RF_FRONTEND_PORT, 'Frontend', timeout=30):
                elapsed = time.time() - start_time
                print(f"✅ Frontend server started (PID: {proc.pid}) [took {elapsed:.1f}s]")
                return True
            else:
                elapsed = time.time() - start_time
                print(f"❌ Frontend server failed to start [took {elapsed:.1f}s]")
                return False

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error starting Frontend: {e} [took {elapsed:.1f}s]")
            return False

    def _wait_for_port(self, port: int, service_name: str, timeout: int = 30) -> bool:
        """Wait for a port to become available."""
        import socket

        print(f"⏳ Waiting for {service_name} to be ready...")

        for i in range(timeout):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()

                if result == 0:
                    return True

            except Exception:
                pass

            time.sleep(1)

        return False

    def monitor_processes(self):
        """Monitor running processes and restart if needed."""
        print("\n👀 Monitoring services... Press Ctrl+C to stop\n")

        # Keep track of which processes we've reported as crashed
        reported_crashes = set()

        try:
            while True:
                time.sleep(5)

                # Check if any process has died
                for i, proc in enumerate(self.processes):
                    if proc.poll() is not None and proc.pid not in reported_crashes:
                        print(f"\n⚠️  Process {proc.pid} has stopped unexpectedly")

                        # Try to get output from the crashed process
                        try:
                            output, _ = proc.communicate(timeout=1)
                            if output:
                                print(f"📋 Last output from process {proc.pid}:")
                                print(output[-2000:] if len(output) > 2000 else output)  # Last 2000 chars
                        except Exception as e:
                            print(f"   Could not retrieve process output: {e}")

                        reported_crashes.add(proc.pid)

        except KeyboardInterrupt:
            print("\n🛑 Received stop signal")
            self.cleanup()

    def cleanup(self):
        """Cleanup all processes."""
        print("🧹 Cleaning up processes...")

        for proc in self.processes:
            try:
                if proc.poll() is None:  # Process still running
                    print(f"   Stopping process {proc.pid}...")
                    proc.terminate()
                    proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"   Force killing process {proc.pid}...")
                proc.kill()
            except Exception as e:
                print(f"   Error stopping process {proc.pid}: {e}")

        self.processes.clear()
        print("✅ All services stopped")


def check_environment():
    """Check if running in appropriate environment."""
    env = get_notebook_environment()
    print(f"🔍 Detected environment: {env}")

    if env == 'colab':
        print("✅ Running in Google Colab")
        return True
    elif env == 'jupyter':
        print("⚠️  Running in Jupyter (not Colab)")
        print("   Native port forwarding will not work. Consider using cloudflare or ngrok.")
        return True
    else:
        print("⚠️  Not running in a notebook environment")
        print("   Consider using `rapidfireai start` instead")
        return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Checking dependencies...")

    required = ['mlflow', 'gunicorn', 'flask', 'flask-cors']
    missing = []

    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"📥 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q'] + missing
        )
        print("✅ Dependencies installed")
    else:
        print("✅ All dependencies available")


def cleanup_existing_processes():
    """Kill any existing RapidFire processes."""
    print("🧹 Cleaning up any existing RapidFire processes...")

    killed_any = False

    # Kill by process pattern (safer - only kills RapidFire processes)
    processes_to_kill = [
        ('mlflow server', 'MLflow'),
        ('gunicorn.*rapidfireai', 'Dispatcher'),  # Only kill gunicorn running rapidfireai
        ('server.py', 'Frontend'),
        ('cloudflared', 'Cloudflare Tunnel')
    ]

    for pattern, name in processes_to_kill:
        try:
            result = subprocess.run(
                ['pkill', '-9', '-f', pattern],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"   Stopped existing {name} process")
                killed_any = True
        except Exception:
            pass

    # Also check for Python processes running our specific files (more targeted)
    python_files = [
        ('rapidfireai/frontend/server.py', 'Frontend'),
        ('rapidfireai/start_colab.py', 'Colab starter')
    ]

    for file_pattern, name in python_files:
        try:
            result = subprocess.run(
                ['pkill', '-9', '-f', file_pattern],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"   Stopped existing {name} process")
                killed_any = True
        except Exception:
            pass

    if killed_any:
        # Give processes time to fully terminate
        time.sleep(2)
        print("✅ Cleanup complete\n")
    else:
        print("✅ No existing processes found\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("  RapidFire AI - Google Colab Startup")
    print("=" * 60 + "\n")

    # Cleanup any existing processes first
    cleanup_existing_processes()

    # Check environment
    check_environment()

    # Install dependencies
    install_dependencies()

    # Create service manager
    manager = ServiceManager()

    # Start backend services only (MLflow and Dispatcher)
    print("\n" + "=" * 60)
    print("  Starting Backend Services")
    print("=" * 60 + "\n")

    services_started = 0

    if manager.start_mlflow():
        services_started += 1
    else:
        print("❌ Failed to start MLflow - aborting")
        return 1

    if manager.start_dispatcher():
        services_started += 1
    else:
        print("❌ Failed to start Dispatcher - aborting")
        manager.cleanup()
        return 1

    print(f"\n✅ Backend services started ({services_started}/2)\n")

    # Create tunnels for backend services FIRST (if using tunneling)
    mlflow_url = None
    dispatcher_url = None
    frontend_url = None

    if RF_TUNNEL_METHOD in ['cloudflare', 'ngrok']:
        print("=" * 60)
        print(f"  Creating Backend Tunnels ({RF_TUNNEL_METHOD})")
        print("=" * 60 + "\n")

        if RF_TUNNEL_METHOD == 'cloudflare':
            from rapidfireai.utils.colab_helper import setup_cloudflare_tunnel
            mlflow_url = setup_cloudflare_tunnel(RF_MLFLOW_PORT, "MLflow Tracking UI")
            dispatcher_url = setup_cloudflare_tunnel(RF_API_PORT, "Dispatcher API")
        elif RF_TUNNEL_METHOD == 'ngrok':
            from rapidfireai.utils.colab_helper import setup_ngrok_tunnel
            mlflow_url = setup_ngrok_tunnel(RF_MLFLOW_PORT, RF_NGROK_TOKEN, "MLflow Tracking UI")
            dispatcher_url = setup_ngrok_tunnel(RF_API_PORT, RF_NGROK_TOKEN, "Dispatcher API")

        # Set environment variables for frontend proxy to use
        if mlflow_url:
            os.environ['RF_MLFLOW_URL'] = mlflow_url.rstrip('/') + '/'
        if dispatcher_url:
            os.environ['RF_DISPATCHER_URL'] = dispatcher_url.rstrip('/') + '/'

        if mlflow_url and dispatcher_url:
            print("\n🔗 Frontend proxy will use:")
            print(f"   RF_MLFLOW_URL={os.environ.get('RF_MLFLOW_URL')}")
            print(f"   RF_DISPATCHER_URL={os.environ.get('RF_DISPATCHER_URL')}\n")

    # NOW start frontend (it will inherit the tunnel URL env vars)
    print("=" * 60)
    print("  Starting Frontend Service")
    print("=" * 60 + "\n")

    if manager.start_frontend():
        services_started += 1
    else:
        print("❌ Failed to start Frontend - aborting")
        manager.cleanup()
        return 1

    # Create tunnel for frontend LAST
    if RF_TUNNEL_METHOD in ['cloudflare', 'ngrok']:
        print("\n" + "=" * 60)
        print("  Creating Frontend Tunnel")
        print("=" * 60 + "\n")

        if RF_TUNNEL_METHOD == 'cloudflare':
            from rapidfireai.utils.colab_helper import setup_cloudflare_tunnel
            frontend_url = setup_cloudflare_tunnel(RF_FRONTEND_PORT, "RapidFire Dashboard")
        elif RF_TUNNEL_METHOD == 'ngrok':
            from rapidfireai.utils.colab_helper import setup_ngrok_tunnel
            frontend_url = setup_ngrok_tunnel(RF_FRONTEND_PORT, RF_NGROK_TOKEN, "RapidFire Dashboard")

    # Print summary
    print("\n" + "=" * 60)
    print("  📋 RapidFire Services Summary")
    print("=" * 60)

    if RF_TUNNEL_METHOD in ['cloudflare', 'ngrok']:
        print(f"  FRONTEND: {frontend_url or '❌ Failed'}")
        print(f"  MLFLOW: {mlflow_url or '❌ Failed'}")
        print(f"  DISPATCHER: {dispatcher_url or '❌ Failed'}")
    else:
        print(f"  FRONTEND: http://localhost:{RF_FRONTEND_PORT}")
        print(f"  MLFLOW: http://localhost:{RF_MLFLOW_PORT}")
        print(f"  DISPATCHER: http://localhost:{RF_API_PORT}")

    print("=" * 60 + "\n")

    # Monitor processes
    manager.monitor_processes()

    return 0


if __name__ == '__main__':
    sys.exit(main())
