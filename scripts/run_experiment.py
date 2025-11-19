#!/usr/bin/env python
"""
Command-line script to run QuantML experiments.

This is a convenience script that calls the main CLI module.
"""

import sys
from pathlib import Path

# Add quantml to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantml.cli.run_experiment import main

if __name__ == "__main__":
    main()

