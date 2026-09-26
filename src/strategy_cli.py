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
    parser = argparse.ArgumentParser(description="SMG Quantitative Strategy CLI & Backtesting Engine")
    parser.add_argument("snapshot", nargs="?", type=Path, help="Path to snapshot JSON for review report")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown report path")
    parser.add_argument("--market-weak", action="store_true", help="Flag indicating weak market conditions")
    
    # Backtesting arguments
    parser.add_argument("--backtest", action="store_true", help="Run historical point-in-time backtest")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA,AMZN,GOOGL", help="Comma-separated tickers for backtesting")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-06-30", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=100_000.0, help="Initial portfolio capital")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker (default: SPY)")
    parser.add_argument("--json-output", type=Path, default=None, help="Export backtest metrics as JSON file")

    # Interactive Dashboard arguments
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive Web UI dashboard on localhost")
    parser.add_argument("--port", type=int, default=8088, help="Port for interactive dashboard (default: 8088)")
    parser.add_argument("--export-dashboard", type=Path, default=None, help="Export standalone HTML dashboard to path")

    args = parser.parse_args()

    if args.dashboard:
        from smg_strategy.dashboard import serve_dashboard
        serve_dashboard(port=args.port)
    elif args.export_dashboard:
        from smg_strategy.dashboard import export_dashboard
        out = export_dashboard(args.export_dashboard)
        print(f"📊 交互式量化 Web 仪表盘已导出至: {out.resolve()}")
    elif args.backtest:
        from smg_strategy.backtest import BacktestConfig, BacktestEngine
        ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        cfg = BacktestConfig(
            tickers=ticker_list,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.initial_capital,
            benchmark=args.benchmark.upper()
        )
        print(f"🚀 启动 SMG 量化历史回测 [{args.start} -> {args.end}]...")
        engine = BacktestEngine(cfg)
        result = engine.run()
        print(result.print_summary())

        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"💾 回测详细指标已保存至: {args.json_output}")
    else:
        if not args.snapshot or not args.output:
            parser.error("snapshot and --output are required when not running in --backtest or --dashboard mode")
        generate_report(args.snapshot, args.output, args.market_weak)


if __name__ == "__main__":
    main()
