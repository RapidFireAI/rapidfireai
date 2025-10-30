#!/usr/bin/env python3
"""
Test script to verify the rapidfireai package structure
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that the package can be imported correctly."""
    print("Testing package imports...")
    
    try:
        import rapidfireai
        print(f"✓ Successfully imported rapidfireai version {rapidfireai.__version__}")
        
        from rapidfireai import Experiment
        print("✓ Successfully imported Experiment class")
        
        from rapidfireai import coming_soon
        print("✓ Successfully imported coming_soon function")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import rapidfireai: {e}")
        return False

def test_cli_module():
    """Test that the CLI module can be imported."""
    print("\nTesting CLI module...")
    
    try:
        from rapidfireai import cli
        print("✓ Successfully imported CLI module")
        
        # Test that the main function exists
        if hasattr(cli, 'main'):
            print("✓ CLI main function exists")
        else:
            print("✗ CLI main function not found")
            return False
            
        return True
    except ImportError as e:
        print(f"✗ Failed to import CLI module: {e}")
        return False

def test_package_structure():
    """Test that the package has the expected structure."""
    print("\nTesting package structure...")
    
    package_dir = Path(__file__).parent / "rapidfireai"
    
    expected_files = [
        "__init__.py",
        "cli.py",
        "start.sh",
        "experiment.py",
    ]
    
    expected_dirs = [
        "dispatcher",
        "frontend", 
        "ml",
        "automl",
        "backend",
        "db",
        "utils",
    ]
    
    all_good = True
    
    # Check files
    for file_name in expected_files:
        file_path = package_dir / file_name
        if file_path.exists():
            print(f"✓ Found {file_name}")
        else:
            print(f"✗ Missing {file_name}")
            all_good = False
    
    # Check directories
    for dir_name in expected_dirs:
        dir_path = package_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ Found directory {dir_name}")
        else:
            print(f"✗ Missing directory {dir_name}")
            all_good = False
    
    return all_good

def test_start_script():
    """Test that the start.sh script exists and is executable."""
    print("\nTesting start.sh script...")
    
    script_path = Path(__file__).parent / "rapidfireai" / "start.sh"
    
    if script_path.exists():
        print("✓ start.sh script exists")
        
        # Check if it's executable
        if os.access(script_path, os.X_OK):
            print("✓ start.sh script is executable")
        else:
            print("⚠ start.sh script exists but is not executable")
            print("  You may need to run: chmod +x rapidfireai/start.sh")
            
        return True
    else:
        print("✗ start.sh script not found")
        return False

def main():
    """Run all tests."""
    print("RapidFire AI Package Test")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_cli_module,
        test_package_structure,
        test_start_script,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The package is ready for installation.")
        print("\nTo install the package, run:")
        print("  pip install -e .")
        print("\nOr build and install:")
        print("  python -m build")
        print("  pip install dist/*.whl")
    else:
        print("❌ Some tests failed. Please fix the issues before installing.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 