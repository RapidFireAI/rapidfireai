"""Utility functions for OS information."""

import subprocess
import re
import os
from pathlib import Path

def mkdir_p(path: str, parents: bool = True, exist_ok: bool = True, notify: bool = True):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
            Path(path).mkdir(parents=parents, exist_ok=exist_ok)
            if notify:
                print(f"Created directory: {path}")
            return
    if not os.path.isdir(path):
        raise OSError(f"Path exist and is not a directory: {path}")
    return


def get_os_package_installed(package_pattern: str):
    """Get list of installed packages matching a pattern."""
    import distro
    dist_id = distro.id()
    
    try:
        if dist_id in ['ubuntu', 'debian']:
            # Use dpkg-query for Debian-based
            result = subprocess.run(
                ['dpkg-query', '-W', '-f=${Package}\n', package_pattern],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
            return []
            
        elif dist_id in ['rhel', 'centos', 'fedora', 'rocky', 'almalinux']:
            # Use rpm for Red Hat-based
            result = subprocess.run(
                ['rpm', '-qa', package_pattern],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
            return []
            
        elif dist_id in ['arch', 'manjaro']:
            # Use pacman for Arch-based
            result = subprocess.run(
                ['pacman', '-Qq'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                all_packages = result.stdout.strip().split('\n')
                # Convert shell glob pattern to regex
                pattern_regex = package_pattern.replace('*', '.*')
                return [pkg for pkg in all_packages if re.match(pattern_regex, pkg)]
            return []
            
        elif dist_id in ['opensuse', 'sles']:
            # Use rpm for openSUSE
            result = subprocess.run(
                ['rpm', '-qa', package_pattern],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
            return []
            
        else:
            # Fallback: try dpkg first, then rpm
            for cmd in [['dpkg-query', '-W', '-f=${Package}\n', package_pattern],
                       ['rpm', '-qa', package_pattern]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if result.returncode == 0 and result.stdout.strip():
                        return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
                except FileNotFoundError:
                    continue
            return []
            
    except Exception as e:
        print(f"Error checking packages: {e}")
        return []
