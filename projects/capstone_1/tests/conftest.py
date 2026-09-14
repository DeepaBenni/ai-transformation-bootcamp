"""Puts the project root on the import path so ``import app`` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
