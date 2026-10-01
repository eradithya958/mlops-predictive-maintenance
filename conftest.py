"""
Pytest configuration — adds project root to sys.path so `src.*` imports resolve
without requiring a full `pip install -e .`.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))
