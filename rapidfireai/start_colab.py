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
                print(f"✅ MLflow server started (PID: {proc.pid})")
                return True
            else:
                print(f"❌ MLflow server failed to start")
                return False

        except Exception as e:
            print(f"❌ Error starting MLflow: {e}")
            return False

    def start_dispatcher(self) -> bool:
        """Start Dispatcher API server with Gunicorn."""
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
                print(f"✅ Dispatcher API started (PID: {proc.pid})")
                return True
            else:
                print(f"❌ Dispatcher API failed to start")
                return False

        except Exception as e:
            print(f"❌ Error starting Dispatcher: {e}")
            return False

    def start_frontend(self) -> bool:
        """Start Frontend Flask server."""
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
                print(f"✅ Frontend server started (PID: {proc.pid})")
                return True
            else:
                print(f"❌ Frontend server failed to start")
                return False

        except Exception as e:
            print(f"❌ Error starting Frontend: {e}")
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

        try:
            while True:
                time.sleep(5)

                # Check if any process has died
                for proc in self.processes:
                    if proc.poll() is not None:
                        print(f"⚠️  Process {proc.pid} has stopped unexpectedly")

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


def main():
    """Main entry point."""
    print("=" * 60)
    print("  RapidFire AI - Google Colab Startup")
    print("=" * 60 + "\n")

    # Check environment
    check_environment()

    # Install dependencies
    install_dependencies()

    # Create service manager
    manager = ServiceManager()

    # Start services
    print("\n" + "=" * 60)
    print("  Starting Services")
    print("=" * 60 + "\n")

    services_started = 0

    if manager.start_mlflow():
        services_started += 1

    if manager.start_dispatcher():
        services_started += 1

    if manager.start_frontend():
        services_started += 1

    if services_started < 3:
        print(f"\n⚠️  Only {services_started}/3 services started successfully")
        print("Check the logs above for errors")
        manager.cleanup()
        return 1

    print("\n" + "=" * 60)
    print("  ✅ All Services Started!")
    print("=" * 60 + "\n")

    # Expose ports using selected method
    print("=" * 60)
    print(f"  Exposing Services (method: {RF_TUNNEL_METHOD})")
    print("=" * 60 + "\n")

    urls = expose_rapidfire_services(
        method=RF_TUNNEL_METHOD,
        mlflow_port=RF_MLFLOW_PORT,
        dispatcher_port=RF_API_PORT,
        frontend_port=RF_FRONTEND_PORT,
        ngrok_token=RF_NGROK_TOKEN
    )

    if not urls or not any(urls.values()):
        print("\n⚠️  Failed to expose services via tunneling")
        print("Services are running locally but not accessible externally")

    # Monitor processes
    manager.monitor_processes()

    return 0


if __name__ == '__main__':
    sys.exit(main())
