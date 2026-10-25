import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    importlib.util.find_spec("hdbscan") is None, reason="needs the phase2 extra"
)
def test_mining_and_analysis_workflow_with_stand_in_embedder(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_workflow",
            "--database",
            str(tmp_path / "verify.db"),
            "--stand-in-embedder",
        ],
        check=False,
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow completed" in result.stdout
