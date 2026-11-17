"""Runs the reference script once and turns each PASS/FAIL line into a pytest case (24 PASS expected)."""
import contextlib
import io
import runpy
from functools import lru_cache
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("mathkit_check.py")


@lru_cache(maxsize=1)
def _output() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    return buf.getvalue()


_CHECKS = [ln for ln in _output().splitlines() if " ref " in ln and ln.rstrip().endswith(("PASS", "FAIL"))]


@pytest.mark.parametrize("line", _CHECKS, ids=[c[:50].strip() for c in _CHECKS])
def test_check_line(line):
    assert line.rstrip().endswith("PASS"), line


def test_totals():
    assert "24 PASS 0 FAIL" in _output()
    assert len(_CHECKS) == 24
