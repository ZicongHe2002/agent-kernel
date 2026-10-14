#!/usr/bin/env python3
"""Validate portable memory input and generate the shared result-only layout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_memory.services.memory_input import main


if __name__ == "__main__":
    raise SystemExit(main())
