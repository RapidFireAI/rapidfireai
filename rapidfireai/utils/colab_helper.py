"""
Helper utilities for running RapidFire AI in Google Colab environments.
"""

import os
import subprocess
import sys
from typing import Optional


def is_colab() -> bool:
    """
    Detect if code is running in Google Colab environment.

    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """
    Detect if code is running in Kaggle notebook environment.

    Returns:
        bool: True if running in Kaggle, False otherwise
    """
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def get_notebook_environment() -> str:
    """
    Detect the current notebook environment.

    Returns:
        str: 'colab', 'kaggle', 'jupyter', or 'unknown'
    """
    if is_colab():
        return 'colab'
    elif is_kaggle():
        return 'kaggle'
    elif 'ipykernel' in sys.modules:
        return 'jupyter'
    else:
        return 'unknown'


def expose_port_native(port: int, description: str = "") -> Optional[str]:
    """
    Expose a port using Google Colab's native port forwarding.

    This method uses Colab's built-in `serve_kernel_port_as_window` function
    to create an authenticated URL that proxies to the local port.

    IMPORTANT: This must be called from within a Colab notebook cell, not from
    a subprocess or script. The Colab output APIs require the notebook kernel context.

    Args:
        port: The local port number to expose
        description: Optional description for the service

    Returns:
        str: The public URL if successful, None otherwise
    """
    if not is_colab():
        print(f"Warning: Not in Colab environment. Cannot use native port forwarding.")
        return None

    try:
        # Check if we're in IPython/notebook context
        try:
            get_ipython()
        except NameError:
            print(f"❌ Error: Native port forwarding must be called from a notebook cell, not a script.")
            print(f"   Please use cloudflare or ngrok tunnel instead, or call from notebook.")
            return None

        from google.colab import output

        print(f"🚀 Exposing {description or f'port {port}'} via Colab native forwarding...")
        print(f"   This will open in a new window/tab")

        # This opens the port in a new window
        output.serve_kernel_port_as_window(port)

        # Get the proxy URL (for reference)
        from google.colab.output import eval_js
        url = eval_js(f"google.colab.kernel.proxyPort({port})")

        print(f"✅ {description or 'Service'} is accessible at: {url}")
        return url

    except Exception as e:
        print(f"❌ Error exposing port {port}: {e}")
        import traceback
        traceback.print_exc()
        return None


def expose_port_iframe(port: int, height: int = 600, description: str = "") -> Optional[str]:
    """
    Expose a port using Google Colab's iframe method (displays inline).

    Args:
        port: The local port number to expose
        height: Height of the iframe in pixels
        description: Optional description for the service

    Returns:
        str: The public URL if successful, None otherwise
    """
    if not is_colab():
        print(f"Warning: Not in Colab environment. Cannot use iframe forwarding.")
        return None

    try:
        from google.colab import output

        print(f"📊 Exposing {description or f'port {port}'} in iframe...")

        # This embeds the content in an iframe within the notebook
        output.serve_kernel_port_as_iframe(port, height=height)

        # Get the proxy URL
        from google.colab.output import eval_js
        url = eval_js(f"google.colab.kernel.proxyPort({port})")

        print(f"✅ {description or 'Service'} is accessible at: {url}")
        return url

    except Exception as e:
        print(f"❌ Error exposing port {port} in iframe: {e}")
        return None


def setup_cloudflare_tunnel(port: int, description: str = "") -> Optional[str]:
    """
    Setup Cloudflare Tunnel (cloudflared) for port forwarding.

    This is a free alternative that doesn't require registration.

    Args:
        port: The local port to tunnel
        description: Optional description for the service

    Returns:
        str: The public URL if successful, None otherwise
    """
    import time
    start_time = time.time()

    print(f"🌐 Setting up Cloudflare Tunnel for {description or f'port {port}'}...")

    # Check if cloudflared is installed
    if not _check_command('cloudflared'):
        print("Installing cloudflared...")
        if not _install_cloudflared():
            print("❌ Failed to install cloudflared")
            return None

    # Start tunnel in background
    try:
        import threading
        import re

        url_container = {'url': None}

        def run_tunnel():
            proc = subprocess.Popen(
                ['cloudflared', 'tunnel', '--url', f'http://localhost:{port}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Parse URL from output
            for line in proc.stdout:
                if 'trycloudflare.com' in line:
                    match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        url_container['url'] = match.group(0)

        thread = threading.Thread(target=run_tunnel, daemon=True)
        thread.start()

        # Wait a moment for URL to be generated
        time.sleep(5)

        elapsed = time.time() - start_time
        if url_container['url']:
            print(f"✅ {description or 'Service'} is accessible at: {url_container['url']} [took {elapsed:.1f}s]")
        else:
            print(f"⚠️  Tunnel created but URL not detected yet [took {elapsed:.1f}s]")

        return url_container['url']

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error setting up Cloudflare Tunnel: {e} [took {elapsed:.1f}s]")
        return None


def setup_ngrok_tunnel(port: int, auth_token: Optional[str] = None, description: str = "") -> Optional[str]:
    """
    Setup ngrok tunnel for port forwarding.

    Note: Requires ngrok auth token for persistent usage.

    Args:
        port: The local port to tunnel
        auth_token: ngrok authentication token (optional for testing)
        description: Optional description for the service

    Returns:
        str: The public URL if successful, None otherwise
    """
    print(f"🔗 Setting up ngrok tunnel for {description or f'port {port}'}...")

    try:
        from pyngrok import ngrok as pyngrok_client

        # Set auth token if provided
        if auth_token:
            pyngrok_client.set_auth_token(auth_token)

        # Start tunnel
        tunnel = pyngrok_client.connect(port)
        url = tunnel.public_url

        print(f"✅ {description or 'Service'} is accessible at: {url}")
        return url

    except ImportError:
        print("Installing pyngrok...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyngrok'])
        return setup_ngrok_tunnel(port, auth_token, description)

    except Exception as e:
        print(f"❌ Error setting up ngrok tunnel: {e}")
        if "authentication" in str(e).lower():
            print("💡 Tip: ngrok requires an auth token. Get one at: https://dashboard.ngrok.com/signup")
            print("    Then set it with: RF_NGROK_TOKEN environment variable")
        return None


def _check_command(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    try:
        subprocess.run([cmd, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _install_cloudflared() -> bool:
    """Install cloudflared binary."""
    try:
        # Detect OS and architecture
        import platform

        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == 'linux':
            if 'x86_64' in machine or 'amd64' in machine:
                url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'
            elif 'arm' in machine or 'aarch64' in machine:
                url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64'
            else:
                print(f"Unsupported architecture: {machine}")
                return False
        else:
            print(f"Unsupported OS: {system}")
            return False

        # Download and install
        print(f"Downloading cloudflared from {url}...")
        subprocess.check_call(['wget', '-q', url, '-O', '/tmp/cloudflared'])
        subprocess.check_call(['chmod', '+x', '/tmp/cloudflared'])
        subprocess.check_call(['sudo', 'mv', '/tmp/cloudflared', '/usr/local/bin/cloudflared'])

        print("✅ cloudflared installed successfully")
        return True

    except Exception as e:
        print(f"❌ Error installing cloudflared: {e}")
        return False


def expose_rapidfire_services(
    method: str = 'native',
    mlflow_port: int = 5002,
    dispatcher_port: int = 8080,
    frontend_port: int = 3000,
    ngrok_token: Optional[str] = None
) -> dict:
    """
    Expose all RapidFire services using the specified method.

    IMPORTANT: If using method='native', this must be called from within a Colab
    notebook cell, not from a script or CLI command.

    Args:
        method: 'native' (Colab built-in), 'cloudflare', or 'ngrok'
        mlflow_port: Port for MLflow server
        dispatcher_port: Port for dispatcher API
        frontend_port: Port for frontend dashboard
        ngrok_token: Required for ngrok method

    Returns:
        dict: Mapping of service names to public URLs
    """
    urls = {}

    if method == 'native':
        if not is_colab():
            print("❌ Native method only works in Google Colab")
            return urls

        # Check if we're in notebook context
        try:
            get_ipython()
        except NameError:
            print("❌ Native method requires running from a Colab notebook cell")
            print("   Use 'cloudflare' or 'ngrok' method instead when running from CLI")
            return urls

        print("🚀 Exposing RapidFire services using Colab native forwarding...\n")

        urls['frontend'] = expose_port_native(frontend_port, "RapidFire Dashboard")
        urls['mlflow'] = expose_port_native(mlflow_port, "MLflow Tracking UI")
        urls['dispatcher'] = expose_port_native(dispatcher_port, "Dispatcher API")

    elif method == 'cloudflare':
        print("🌐 Exposing RapidFire services using Cloudflare Tunnel...\n")

        urls['frontend'] = setup_cloudflare_tunnel(frontend_port, "RapidFire Dashboard")
        urls['mlflow'] = setup_cloudflare_tunnel(mlflow_port, "MLflow Tracking UI")
        urls['dispatcher'] = setup_cloudflare_tunnel(dispatcher_port, "Dispatcher API")

    elif method == 'ngrok':
        print("🔗 Exposing RapidFire services using ngrok...\n")

        if not ngrok_token:
            ngrok_token = os.environ.get('RF_NGROK_TOKEN')

        urls['frontend'] = setup_ngrok_tunnel(frontend_port, ngrok_token, "RapidFire Dashboard")
        urls['mlflow'] = setup_ngrok_tunnel(mlflow_port, ngrok_token, "MLflow Tracking UI")
        urls['dispatcher'] = setup_ngrok_tunnel(dispatcher_port, ngrok_token, "Dispatcher API")

    else:
        print(f"❌ Unknown method: {method}")
        print("   Valid methods: 'native', 'cloudflare', 'ngrok'")
        return urls

    # Print summary
    print("\n" + "="*60)
    print("📋 RapidFire Services Summary:")
    print("="*60)
    for service, url in urls.items():
        if url:
            print(f"  {service.upper()}: {url}")
        else:
            print(f"  {service.upper()}: ❌ Failed to expose")
    print("="*60 + "\n")

    return urls
