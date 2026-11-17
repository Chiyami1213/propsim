"""Launch PropSim:  python run.py [case.psim.json]

Works from a fresh checkout without `pip install -e .` (adds this folder to sys.path).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from propsim.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
