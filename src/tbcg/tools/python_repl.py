"""Guarded Python execution for the Finance agent's quantitative work.

Wraps LangChain's ``PythonREPL``. The agent is expected to ``print`` its
results; we capture stdout and return it. Execution is local and best-effort —
errors are returned as strings rather than raised.
"""

from __future__ import annotations

from functools import lru_cache

from langsmith import traceable


@lru_cache(maxsize=1)
def _repl():
    from langchain_experimental.utilities import PythonREPL

    return PythonREPL()


@traceable(run_type="tool", name="python_repl")
def run_python(code: str) -> str:
    """Execute ``code`` and return captured stdout (or an error string)."""
    try:
        return _repl().run(code)
    except Exception as exc:  # noqa: BLE001
        return f"PythonREPL error: {exc}"
