#!/usr/bin/env python3
"""
Fix notebook cell structure for GitHub rendering.

This script automatically fixes common notebook formatting issues:
- Removes invalid execution_count/outputs from markdown cells
- Adds missing execution_count/outputs to code cells

Usage:
    python tests/fix_notebooks.py [notebook_path]

    If no path is provided, fixes all notebooks in tutorial_notebooks/
"""

import json
import sys
from pathlib import Path


def fix_notebook(notebook_path):
    """Fix notebook cells to have correct schema.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        int: Number of cells fixed
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    fixed_count = 0
    for i, cell in enumerate(notebook.get('cells', [])):
        cell_type = cell.get('cell_type')

        if cell_type == 'markdown':
            # Markdown cells should NOT have execution_count or outputs
            changed = False
            if 'execution_count' in cell:
                del cell['execution_count']
                changed = True
            if 'outputs' in cell:
                del cell['outputs']
                changed = True

            if changed:
                fixed_count += 1
                print(f"  Fixed markdown cell {i}: removed execution_count/outputs")

        elif cell_type == 'code':
            # Code cells MUST have execution_count (can be None) and outputs (can be [])
            changed = False
            if 'execution_count' not in cell:
                cell['execution_count'] = None
                changed = True
                print(f"  Fixed code cell {i}: added execution_count")

            if 'outputs' not in cell:
                cell['outputs'] = []
                changed = True
                print(f"  Fixed code cell {i}: added outputs")

            if changed:
                fixed_count += 1

    if fixed_count > 0:
        # Write back to file
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✅ Fixed {fixed_count} cells in {notebook_path.name}\n")
    else:
        print(f"✅ No issues found in {notebook_path.name}\n")

    return fixed_count


def main():
    """Main entry point."""
    # Get notebook path(s)
    if len(sys.argv) > 1:
        # Fix specific notebook
        notebook_paths = [Path(sys.argv[1])]
    else:
        # Fix all notebooks in tutorial_notebooks/
        base_dir = Path(__file__).parent.parent / "tutorial_notebooks"
        notebook_paths = list(base_dir.glob("*.ipynb"))

    if not notebook_paths:
        print("No notebooks found to fix.")
        return 1

    print(f"Fixing {len(notebook_paths)} notebook(s)...\n")

    total_fixed = 0
    for notebook_path in notebook_paths:
        print(f"Checking {notebook_path.name}...")
        fixed = fix_notebook(notebook_path)
        total_fixed += fixed

    if total_fixed > 0:
        print(f"\n🎉 Fixed {total_fixed} total cells across {len(notebook_paths)} notebook(s)")
        print("\nRun tests to verify:")
        print("  pytest tests/test_notebooks.py")
    else:
        print(f"\n✨ All {len(notebook_paths)} notebook(s) are already correctly formatted!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
