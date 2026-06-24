import sys
from pathlib import Path

# Make the ``src`` layout importable in tests.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
