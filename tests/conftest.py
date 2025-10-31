import sys
from pathlib import Path

# Ensure repository root is on sys.path so tests can import local modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optionally print the path for debugging when verbose
def pytest_configure(config):
    # config can be used to enable debug logging if needed
    pass
