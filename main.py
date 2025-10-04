"""Compatibility shim for the cointegration CLI."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def _load_cli_module():
    try:
        return import_module("cointegration_analysis.cli")
    except ModuleNotFoundError:  # pragma: no cover - fallback for local execution
        project_root = Path(__file__).resolve().parent
        src_dir = project_root / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        return import_module("cointegration_analysis.cli")


def main() -> int:
    """Entry point delegating to the packaged CLI."""

    return _CLI_MODULE.main()


_CLI_MODULE = _load_cli_module()
load_pair_data = _CLI_MODULE.load_pair_data


if __name__ == "__main__":
    sys.exit(main())
