"""Custom setup.py to build frontend during package installation from source."""

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithFrontend(build_py):
    """Custom build_py that builds the frontend if not already built."""

    def run(self):
        self.build_frontend_if_needed()
        super().run()

    def build_frontend_if_needed(self):
        """Build the frontend if the build directory doesn't exist."""
        frontend_dir = Path(__file__).parent / "rapidfireai" / "fit" / "frontend"
        build_dir = frontend_dir / "build"
        index_html = build_dir / "index.html"

        # Skip if already built (check for index.html to ensure complete build)
        if build_dir.exists() and index_html.exists():
            print("Frontend already built, skipping...")
            return

        print("Frontend build not found. Attempting to build...")

        # Check for Node.js
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            node_version = result.stdout.strip()
            print(f"Found Node.js {node_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("WARNING: Node.js not found. Frontend will not be built.")
            print("         The dashboard will not work without the frontend.")
            print("         Options:")
            print("         1. Install Node.js 22+ and reinstall this package")
            print("         2. Use 'pip install rapidfireai' from PyPI instead")
            return

        # Check for yarn in the frontend directory
        yarn_path = frontend_dir / "yarn" / "releases" / "yarn-4.9.1.cjs"
        if not yarn_path.exists():
            print(f"WARNING: Yarn not found at {yarn_path}")
            print("         Frontend will not be built.")
            return

        print("Building frontend (this may take a minute)...")

        try:
            # Install dependencies
            print("Running yarn install...")
            subprocess.run(
                ["node", str(yarn_path), "install"],
                cwd=frontend_dir,
                check=True,
                env={**subprocess.os.environ, "NODE_OPTIONS": "--max-old-space-size=4096"},
            )

            # Build frontend
            print("Running yarn build...")
            subprocess.run(
                ["node", str(yarn_path), "build"],
                cwd=frontend_dir,
                check=True,
                env={**subprocess.os.environ, "NODE_OPTIONS": "--max-old-space-size=4096"},
            )

            print("Frontend built successfully!")

        except subprocess.CalledProcessError as e:
            print(f"WARNING: Frontend build failed: {e}")
            print("         The dashboard may not work.")
            print("         Try building manually:")
            print(f"         cd {frontend_dir}")
            print("         yarn install && yarn build")


setup(
    cmdclass={
        "build_py": BuildPyWithFrontend,
    },
)
