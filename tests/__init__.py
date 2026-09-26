import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct test discovery without requiring manual PYTHONPATH=src
_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
