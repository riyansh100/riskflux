"""Guards on dvc.yaml: a stage must declare every input it actually uses.

If a stage imports a riskflux module that isn't listed in its deps, editing that
module wouldn't trigger `dvc repro`, and the stage's outputs would silently go stale.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src"
DVC = yaml.safe_load((ROOT / "dvc.yaml").read_text())
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def module_file(dotted: str) -> Path | None:
    """'riskflux.data.clean' -> src/riskflux/data/clean.py (None for packages/non-files)."""
    path = SRC.joinpath(*dotted.split("."))
    return path.with_suffix(".py") if path.with_suffix(".py").exists() else None


def riskflux_imports(file: Path) -> set[Path]:
    """Every riskflux module file `file` imports, directly or transitively.
    Package __init__ files are skipped: ours contain only docstrings."""
    found: set[Path] = set()
    for node in ast.walk(ast.parse(file.read_text())):
        candidates = []
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            # `from riskflux import columns` imports a submodule; `from riskflux.x import y`
            # imports from module x. Try both interpretations.
            candidates = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        for dotted in candidates:
            target = module_file(dotted) if dotted.startswith("riskflux") else None
            if target and target not in found:
                found |= {target} | riskflux_imports(target)
    return found


def stage_module(cmd: str) -> Path:
    match = re.fullmatch(r"python -m ([\w.]+)", cmd.strip())
    assert match, f"unexpected stage command: {cmd!r}"
    return module_file(match.group(1))


@pytest.mark.parametrize("stage", DVC["stages"])
def test_stage_declares_all_code_it_imports(stage):
    spec = DVC["stages"][stage]
    entry = stage_module(spec["cmd"])
    needed = {entry, *riskflux_imports(entry)}
    declared = {ROOT / dep for dep in spec["deps"]}
    missing = sorted(str(p.relative_to(ROOT)) for p in needed - declared)
    assert not missing, f"stage '{stage}' imports modules missing from its deps: {missing}"


@pytest.mark.parametrize("stage", DVC["stages"])
def test_stage_params_exist(stage):
    for key in DVC["stages"][stage].get("params", []):
        assert key in PARAMS, f"stage '{stage}' depends on unknown param '{key}'"


def paths(entries: list) -> set[str]:
    """dvc.yaml entries are either 'path' or {'path': {options}}."""
    return {next(iter(e)) if isinstance(e, dict) else e for e in entries}


def test_each_stage_consumes_the_previous_stage_output():
    stages = DVC["stages"]
    produced = set().union(*(paths(spec.get("outs", [])) for spec in stages.values()))
    for name, spec in stages.items():
        generated = [
            d for d in spec["deps"] if d.startswith(("data/interim/", "data/processed/", "models/"))
        ]
        for dep in generated:
            assert dep in produced, f"stage '{name}' reads {dep}, which no stage produces"
