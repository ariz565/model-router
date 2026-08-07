"""Put src/ on sys.path so `import modelrouter` resolves during tests without
requiring an editable install first. pyproject.toml's [tool.pytest.ini_options]
pythonpath already does this on modern pytest; this conftest is the belt-and-
suspenders fallback for environments where that setting isn't honored."""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
