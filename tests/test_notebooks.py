"""Test suite for notebook formatting and validation."""

import json
from pathlib import Path

import pytest


class TestNotebookFormatting:
    """Test suite for ensuring notebooks have correct cell structure."""

    @pytest.fixture
    def notebook_dir(self):
        """Get the tutorial_notebooks directory."""
        # Assumes tests/ is at same level as tutorial_notebooks/
        return Path(__file__).parent.parent / "tutorial_notebooks"

    @pytest.fixture
    def all_notebooks(self, notebook_dir):
        """Get all notebook files."""
        return list(notebook_dir.glob("*.ipynb"))

    def test_notebooks_exist(self, all_notebooks):
        """Test that we have notebooks to validate."""
        assert len(all_notebooks) > 0, "No notebooks found in tutorial_notebooks/"

    def test_notebook_cell_structure(self, all_notebooks):
        """Test that all notebooks have correct cell structure for GitHub rendering."""
        errors = []

        for notebook_path in all_notebooks:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            for i, cell in enumerate(notebook.get('cells', [])):
                cell_type = cell.get('cell_type')
                cell_id = f"{notebook_path.name}:cell-{i}"

                if cell_type == 'markdown':
                    # Markdown cells should NOT have execution_count or outputs
                    if 'execution_count' in cell:
                        errors.append(
                            f"{cell_id} (markdown): has invalid 'execution_count' field"
                        )
                    if 'outputs' in cell:
                        errors.append(
                            f"{cell_id} (markdown): has invalid 'outputs' field"
                        )

                elif cell_type == 'code':
                    # Code cells MUST have execution_count (can be None) and outputs (can be [])
                    if 'execution_count' not in cell:
                        errors.append(
                            f"{cell_id} (code): missing 'execution_count' field"
                        )
                    if 'outputs' not in cell:
                        errors.append(
                            f"{cell_id} (code): missing 'outputs' field"
                        )

        if errors:
            error_msg = "\n".join([
                "Notebook cell structure validation failed:",
                "",
                *errors,
                "",
                "To fix these issues, run:",
                "  python tests/fix_notebooks.py"
            ])
            pytest.fail(error_msg)

    def test_notebook_has_required_fields(self, all_notebooks):
        """Test that notebooks have required top-level fields."""
        required_fields = ['cells', 'metadata', 'nbformat', 'nbformat_minor']

        for notebook_path in all_notebooks:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            missing = [field for field in required_fields if field not in notebook]
            assert not missing, (
                f"{notebook_path.name} missing required fields: {missing}"
            )

    def test_notebook_cells_have_required_fields(self, all_notebooks):
        """Test that all cells have required fields."""
        required_cell_fields = ['cell_type', 'metadata', 'source']

        for notebook_path in all_notebooks:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            for i, cell in enumerate(notebook.get('cells', [])):
                missing = [field for field in required_cell_fields if field not in cell]
                assert not missing, (
                    f"{notebook_path.name}:cell-{i} missing required fields: {missing}"
                )

    def test_notebook_validates_with_nbformat(self, all_notebooks):
        """Test that notebooks validate with nbformat library."""
        try:
            import nbformat
        except ImportError:
            pytest.skip("nbformat not installed")

        for notebook_path in all_notebooks:
            try:
                nb = nbformat.read(str(notebook_path), as_version=4)
                nbformat.validate(nb)
            except nbformat.ValidationError as e:
                pytest.fail(
                    f"{notebook_path.name} failed nbformat validation: {e}"
                )
