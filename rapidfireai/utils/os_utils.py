"""Utility functions for OS information."""

import platform
import re
import shutil
import subprocess
import os
from pathlib import Path


# Canonical (Debian/Ubuntu) OS package name -> per-package-manager native name.
# Used to install / verify OS dependencies required by ``unstructured[all-docs]``
# (PDF / OCR / XML toolchain) for the ``--multimodal`` init option.
MULTIMODAL_OS_PACKAGES = {
    "libmagic1":    {"apt": "libmagic1",    "dnf": "file-libs",     "yum": "file-libs",     "pacman": "file",     "zypper": "file-magic",   "brew": "libmagic"},
    "poppler-utils":{"apt": "poppler-utils","dnf": "poppler-utils", "yum": "poppler-utils", "pacman": "poppler",  "zypper": "poppler-tools","brew": "poppler"},
    "tesseract-ocr":{"apt": "tesseract-ocr","dnf": "tesseract",     "yum": "tesseract",     "pacman": "tesseract","zypper": "tesseract-ocr","brew": "tesseract"},
    "libxml2":      {"apt": "libxml2",      "dnf": "libxml2",       "yum": "libxml2",       "pacman": "libxml2",  "zypper": "libxml2-2",    "brew": "libxml2"},
    "libxslt1-dev": {"apt": "libxslt1-dev", "dnf": "libxslt-devel", "yum": "libxslt-devel", "pacman": "libxslt",  "zypper": "libxslt-devel","brew": "libxslt"},
    "wget":         {"apt": "wget",         "dnf": "wget",          "yum": "wget",          "pacman": "wget",     "zypper": "wget",         "brew": "wget"},
    "unrar":        {"apt": "unrar",        "dnf": "unrar",         "yum": "unrar",         "pacman": "unrar",    "zypper": "unrar",        "brew": "unrar"},
}

# Python packages that pair with the multimodal OS packages above.
MULTIMODAL_PYTHON_PACKAGES = ["unstructured", "nltk"]


def detect_pkg_manager():
    """Detect the OS package manager.

    Returns (manager, distro_id). ``manager`` is one of
    ``apt``/``dnf``/``yum``/``pacman``/``zypper``/``brew`` or ``None`` if
    nothing supported is available.
    """
    system = platform.system()
    if system == "Darwin":
        return ("brew" if shutil.which("brew") else None, "macos")
    if system != "Linux":
        return (None, system.lower())

    try:
        import distro  # type: ignore
        dist_id = (distro.id() or "").lower()
    except Exception:
        dist_id = ""

    debian_like = {"debian", "ubuntu", "linuxmint", "pop", "kali", "raspbian", "elementary"}
    rhel_like = {"rhel", "centos", "fedora", "rocky", "almalinux", "amzn", "ol"}
    arch_like = {"arch", "manjaro", "endeavouros"}
    suse_like = {"opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles"}

    if dist_id in debian_like and shutil.which("apt-get"):
        return ("apt", dist_id)
    if dist_id in rhel_like:
        if shutil.which("dnf"):
            return ("dnf", dist_id)
        if shutil.which("yum"):
            return ("yum", dist_id)
    if dist_id in arch_like and shutil.which("pacman"):
        return ("pacman", dist_id)
    if dist_id in suse_like and shutil.which("zypper"):
        return ("zypper", dist_id)

    for mgr, cmd in (("apt", "apt-get"), ("dnf", "dnf"), ("yum", "yum"),
                      ("pacman", "pacman"), ("zypper", "zypper")):
        if shutil.which(cmd):
            return (mgr, dist_id)

    return (None, dist_id)


def is_os_pkg_installed(manager: str, name: str) -> bool:
    """Return True if an OS package is installed via the given package manager."""
    try:
        if manager == "apt":
            r = subprocess.run(["dpkg", "-s", name], capture_output=True, text=True, check=False)
            return r.returncode == 0 and "Status: install ok installed" in r.stdout
        if manager in ("dnf", "yum", "zypper"):
            r = subprocess.run(["rpm", "-q", name], capture_output=True, text=True, check=False)
            return r.returncode == 0
        if manager == "pacman":
            r = subprocess.run(["pacman", "-Q", name], capture_output=True, text=True, check=False)
            return r.returncode == 0
        if manager == "brew":
            r = subprocess.run(["brew", "list", "--formula", name], capture_output=True, text=True, check=False)
            return r.returncode == 0
    except FileNotFoundError:
        return False
    return False


def build_install_cmd(manager: str, names: list[str]) -> list[str] | None:
    """Build the install command for the given OS package manager."""
    if manager == "apt":
        return ["sudo", "apt-get", "install", "-y"] + names
    if manager == "dnf":
        return ["sudo", "dnf", "install", "-y"] + names
    if manager == "yum":
        return ["sudo", "yum", "install", "-y"] + names
    if manager == "pacman":
        return ["sudo", "pacman", "-S", "--noconfirm", "--needed"] + names
    if manager == "zypper":
        return ["sudo", "zypper", "--non-interactive", "install"] + names
    if manager == "brew":
        # Homebrew explicitly refuses to run under sudo.
        return ["brew", "install"] + names
    return None


def check_multimodal_os_packages() -> dict:
    """Check the multimodal OS dependencies.

    Returns a dict::

        {
            "manager": str | None,
            "distro_id": str,
            "packages": [
                {"canonical": str, "native": str, "installed": bool | None},
                ...,
            ],
            "missing": [native names],
            "all_installed": bool,
        }

    When no supported package manager is detected, ``manager`` is ``None``,
    ``installed`` is ``None`` for each package, ``missing`` is the full list
    of canonical names, and ``all_installed`` is ``False``.
    """
    manager, dist_id = detect_pkg_manager()
    packages: list[dict] = []
    missing: list[str] = []

    if manager is None:
        for canonical in MULTIMODAL_OS_PACKAGES:
            packages.append({"canonical": canonical, "native": canonical, "installed": None})
            missing.append(canonical)
        return {
            "manager": None,
            "distro_id": dist_id,
            "packages": packages,
            "missing": missing,
            "all_installed": False,
        }

    for canonical, native_map in MULTIMODAL_OS_PACKAGES.items():
        native = native_map.get(manager, canonical)
        installed = is_os_pkg_installed(manager, native)
        packages.append({"canonical": canonical, "native": native, "installed": installed})
        if not installed:
            missing.append(native)

    return {
        "manager": manager,
        "distro_id": dist_id,
        "packages": packages,
        "missing": missing,
        "all_installed": len(missing) == 0,
    }


def get_python_package_version(name: str) -> str | None:
    """Return the installed version of a Python distribution, or ``None`` if missing."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def check_multimodal_python_packages() -> dict:
    """Check the multimodal Python packages (``unstructured`` and ``nltk``).

    Returns::

        {
            "packages": [{"name": str, "version": str | None, "installed": bool}, ...],
            "missing": [names],
            "all_installed": bool,
        }
    """
    packages: list[dict] = []
    missing: list[str] = []
    for name in MULTIMODAL_PYTHON_PACKAGES:
        ver = get_python_package_version(name)
        installed = ver is not None
        packages.append({"name": name, "version": ver, "installed": installed})
        if not installed:
            missing.append(name)
    return {
        "packages": packages,
        "missing": missing,
        "all_installed": len(missing) == 0,
    }

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
