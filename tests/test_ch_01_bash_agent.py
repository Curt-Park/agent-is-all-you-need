import os
import shutil
from pathlib import Path

import pytest

from ch_01_bash_agent import run_agent


@pytest.fixture
def workspace():
    ws = Path(f"test_workspace_{os.getpid()}").resolve()
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir()

    # Use environment variables if needed to point to the workspace
    # or just chdir temporarily
    original_cwd = os.getcwd()
    os.chdir(ws)

    yield ws

    os.chdir(original_cwd)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)


def test_run_agent_write(workspace):
    run_agent("Write 'hello' to test_write.txt", max_steps=2, enable_hitl=False)
    assert (workspace / "test_write.txt").exists()
    assert (workspace / "test_write.txt").read_text().strip() == "hello"


def test_run_agent_edit(workspace):
    (workspace / "test_edit.txt").write_text("line 1\n")
    run_agent("Append 'line 2' to test_edit.txt", max_steps=2, enable_hitl=False)
    # The file content should now be line 1\nline 2
    assert (workspace / "test_edit.txt").read_text().strip() == "line 1\nline 2"


def test_run_agent_calculation(workspace):
    run_agent("Calculate 2 + 3 and save the result in calculation.txt", max_steps=2, enable_hitl=False)
    assert (workspace / "calculation.txt").exists()
    assert (workspace / "calculation.txt").read_text().strip() == "5"
