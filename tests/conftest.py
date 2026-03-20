"""Shared pytest fixtures for all tests."""

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def workspace():
    """Create a temporary workspace directory for isolated testing."""
    ws = Path(f"test_workspace_{os.getpid()}").resolve()
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir()

    original_cwd = os.getcwd()
    os.chdir(ws)
    yield ws
    os.chdir(original_cwd)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
