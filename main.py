"""Compatibility entry point.

This file keeps a simple `python main.py` command available. The actual pipeline
implementation lives in run_pipeline.py.
"""

from run_pipeline import main


if __name__ == "__main__":
    main()
