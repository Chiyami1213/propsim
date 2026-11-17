"""Launch PropSim:  python run.py [case.psim.json]

Works from a fresh checkout without `pip install -e .` (adds this folder to sys.path).
Imports stay inside the main guard so that worker processes (which re-import this module) do not load the GUI.
"""
import multiprocessing
import sys
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from propsim.app import main

    sys.exit(main())
