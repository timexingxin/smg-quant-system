import argparse
import json
from pathlib import Path
from typing import Union

from smg_strategy.models import Snapshot
from smg_strategy.report import build_proposals, render_review_packet


def generate_report(
    source: Union[str, Path],
    output: Union[str, Path],
    market_weak: bool = False,
) -> Path:
    source_path = Path(source)
    output_path = Path(output)
    snapshot = Snapshot.from_dict(json.loads(source_path.read_text(encoding="utf-8")))
    report = render_review_packet(snapshot, build_proposals(snapshot, market_weak))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an SMG trade review packet")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market-weak", action="store_true")
    args = parser.parse_args()
    generate_report(args.snapshot, args.output, args.market_weak)


if __name__ == "__main__":
    main()
