"""Tests that the IDE-context Claude skill (under ide-context/) matches the
public rapidfireai API. These guard against drift between what the skill teaches
end users and what main actually exposes.

Static checks (no GPU, no HF token) plus one E2E test for the install command.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
IDE_CONTEXT = REPO_ROOT / "ide-context"

SKILL_FILES = {
    "CLAUDE.md": IDE_CONTEXT / "CLAUDE.md",
    "rapidfireai-api.md": IDE_CONTEXT / ".claude" / "rules" / "rapidfireai-api.md",
    "rapidfireai.mdc": IDE_CONTEXT / ".cursor" / "rules" / "rapidfireai.mdc",
}


def _read(name: str) -> str:
    return SKILL_FILES[name].read_text(encoding="utf-8")


def _all_text() -> str:
    return "\n\n".join(_read(n) for n in SKILL_FILES)


def _extract_imports(text: str) -> list[tuple[str, list[str]]]:
    """Pull every `from rapidfireai... import a, b, c` (handles parenthesized multi-line)."""
    pat = re.compile(
        r"from\s+(rapidfireai[\w.]*)\s+import\s+(?:\(\s*([^)]+?)\s*\)|([^\n#]+))",
        re.DOTALL,
    )
    out: list[tuple[str, list[str]]] = []
    for m in pat.finditer(text):
        names_str = m.group(2) or m.group(3) or ""
        names = [n.strip() for n in names_str.split(",") if n.strip()]
        out.append((m.group(1).strip(), names))
    return out


def _extract_python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def _looks_like_signature_display(block: str) -> bool:
    """Detect pseudo-signature blocks like `name: Type = default` in positional
    form, which are doc-only and not valid Python. Quoted dict keys won't match
    because the regex requires an unquoted identifier.
    """
    return bool(re.search(r"^\s+[a-z_][a-z_0-9]*\s*:\s*[A-Za-z]", block, re.MULTILINE))


def test_imports_resolve():
    """Every `from rapidfireai... import X` claim in the skill must resolve."""
    found_any = False
    for module, names in _extract_imports(_all_text()):
        try:
            mod = __import__(module, fromlist=["*"])
        except ImportError as e:  # pragma: no cover - test env should have it
            pytest.fail(f"Skill imports `{module}` but it cannot be imported: {e}")
        for name in names:
            obj = getattr(mod, name, None)
            assert obj is not None, (
                f"Skill claims `from {module} import {name}`, but {name} is None or missing. "
                f"This usually means the optional dep gate in {module}/__init__.py "
                f"failed to register {name}."
            )
            found_any = True
    assert found_any, "Regex did not find any `from rapidfireai...` imports — extractor is broken."


def test_experiment_signature_matches_skill():
    """Experiment.__init__, run_fit, run_evals must include the params the skill documents."""
    from rapidfireai import Experiment

    init_params = inspect.signature(Experiment.__init__).parameters
    for required in ("experiment_name", "mode", "experiment_path", "num_cpus", "num_gpus"):
        assert required in init_params, (
            f"Experiment.__init__ missing `{required}` (skill in rapidfireai-api.md documents it)."
        )

    run_fit_params = inspect.signature(Experiment.run_fit).parameters
    for required in (
        "param_config", "create_model_fn", "train_dataset", "eval_dataset",
        "num_chunks", "seed", "num_gpus", "monte_carlo_simulations",
    ):
        assert required in run_fit_params, f"Experiment.run_fit missing `{required}`."

    run_evals_params = inspect.signature(Experiment.run_evals).parameters
    for required in (
        "config_group", "dataset", "num_shards", "seed",
        "num_actors", "gpus_per_actor", "cpus_per_actor",
    ):
        assert required in run_evals_params, f"Experiment.run_evals missing `{required}`."

    for method in ("end", "get_runs_info", "get_results"):
        assert hasattr(Experiment, method), f"Experiment.{method} is missing."


def test_automl_classes_exported():
    """Every name imported from rapidfireai.automl in the skill must be non-None on main.

    The package uses conditional `try/except` exports that set names to `None` when
    optional deps are missing. A `None` here means the test environment is missing
    a dependency the skill assumes is installed.
    """
    import rapidfireai.automl as automl

    checked = []
    for module, names in _extract_imports(_all_text()):
        if module != "rapidfireai.automl":
            continue
        for name in names:
            obj = getattr(automl, name, None)
            assert obj is not None, (
                f"`rapidfireai.automl.{name}` is None — skill claims it's importable but "
                f"the conditional export gate failed (likely missing optional dep)."
            )
            checked.append(name)
    assert checked, "No `from rapidfireai.automl import ...` lines found in skill — extractor broken."


def test_rag_spec_uses_cfg_kwargs():
    """RFLangChainRagSpec must use the new `*_cfg` shape, not the pre-0.15 flat-kwargs shape."""
    from rapidfireai.automl import RFLangChainRagSpec

    sig = inspect.signature(RFLangChainRagSpec.__init__).parameters
    for required in ("embedding_cfg", "vector_store_cfg", "search_cfg", "reranker_cfg"):
        assert required in sig, (
            f"RFLangChainRagSpec.__init__ missing `{required}` (new API shape — skill documents it)."
        )
    for removed in (
        "vector_store", "search_type", "search_kwargs",
        "reranker_cls", "reranker_kwargs", "embedding_cls", "embedding_kwargs",
    ):
        assert removed not in sig, (
            f"RFLangChainRagSpec.__init__ has `{removed}` — that's the OLD pre-0.15 shape. "
            f"Skill must not document removed kwargs."
        )


def test_python_code_blocks_parse():
    """Every fenced ```python``` block in the skill must parse, except doc-only signature displays."""
    failures: list[str] = []
    parsed = 0
    skipped_sig = 0

    for name, path in SKILL_FILES.items():
        for i, block in enumerate(_extract_python_blocks(path.read_text(encoding="utf-8"))):
            try:
                ast.parse(block)
                parsed += 1
            except SyntaxError as e:
                if _looks_like_signature_display(block):
                    skipped_sig += 1
                else:
                    failures.append(
                        f"{name} block #{i}: SyntaxError at line {e.lineno}: {e.msg}\n"
                        f"--- block ---\n{block}--- end ---"
                    )

    assert not failures, "Skill code blocks failed to parse:\n\n" + "\n\n".join(failures)
    assert parsed > 0, "No parseable python blocks found — extractor regex may be broken."


def test_ports_match_constants():
    """Every port the skill mentions must map to a real Config class on main, and the
    dashboard URL must use 127.0.0.1 (not 0.0.0.0)."""
    from rapidfireai.utils import constants

    expected_ports = {8850, 8851, 8852, 8853, 8855}
    actual_ports: set[int] = set()
    for cls_name in ("DispatcherConfig", "FrontendConfig", "MLflowConfig", "JupyterConfig", "RayConfig"):
        cls = getattr(constants, cls_name, None)
        if cls is not None and hasattr(cls, "PORT"):
            actual_ports.add(int(cls.PORT))

    missing = expected_ports - actual_ports
    assert not missing, (
        f"Skill mentions ports {sorted(missing)} but no Config class on main has them. "
        f"Found: {sorted(actual_ports)}"
    )

    api_md = _read("rapidfireai-api.md")
    assert "http://127.0.0.1:8853" in api_md, (
        "Skill must use 127.0.0.1 for the dashboard URL — that's the host default in constants.py."
    )
    assert "http://0.0.0.0:8853" not in api_md, (
        "Skill still references obsolete 0.0.0.0:8853 dashboard URL."
    )


def test_prompt_manager_uses_embedding_cfg():
    """RFPromptManager must accept embedding_cfg, not the removed embedding_cls/embedding_kwargs."""
    from rapidfireai.automl import RFPromptManager

    sig = inspect.signature(RFPromptManager.__init__).parameters
    assert "embedding_cfg" in sig, "RFPromptManager.__init__ missing `embedding_cfg`."
    for removed in ("embedding_cls", "embedding_kwargs"):
        assert removed not in sig, (
            f"RFPromptManager.__init__ still has `{removed}` (removed in 0.15.x)."
        )


def test_install_ide_context_command(tmp_path, monkeypatch):
    """`install_ide_context()` must drop the three skill files into cwd."""
    monkeypatch.chdir(tmp_path)

    from rapidfireai.cli import install_ide_context

    rc = install_ide_context()
    assert rc == 0, f"install_ide_context returned non-zero exit code {rc}"

    expected = {
        tmp_path / "CLAUDE.md": "RapidFire AI Project",
        tmp_path / ".claude" / "rules" / "rapidfireai-api.md": "## Experiment Class",
        tmp_path / ".cursor" / "rules" / "rapidfireai.mdc": "RapidFire AI",
    }
    for path, sentinel in expected.items():
        assert path.exists(), f"install-ide-context did not create {path}"
        assert path.stat().st_size > 0, f"{path} is empty"
        assert sentinel in path.read_text(encoding="utf-8"), (
            f"{path} missing sentinel `{sentinel}` — skill content may be truncated."
        )
