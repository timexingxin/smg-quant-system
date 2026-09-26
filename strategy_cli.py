"""
Shim entrypoint for backward-compatibility with `python3 strategy_cli.py`.
The canonical CLI module is packaged under `src/strategy_cli.py`.
"""
import sys
from pathlib import Path

# Ensure src/ is on sys.path
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Delegate execution to src/strategy_cli.py without circular import
_cli_file = Path(_src) / "strategy_cli.py"
with open(_cli_file, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), str(_cli_file), "exec"), globals())
