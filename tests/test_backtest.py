import unittest
import pandas as pd
import numpy as np

from smg_strategy.backtest import BacktestConfig, BacktestEngine, BacktestResult


class BacktestEngineTests(unittest.TestCase):
    def _create_mock_df(self, prices, start_date="2024-01-01"):
        n = len(prices)
        dates = pd.date_range(start=start_date, periods=n, freq="B")
        closes = np.array(prices, dtype=float)
        highs = closes * 1.01
        lows = closes * 0.99
        opens = closes * 1.00
        volumes = np.full(n, 1_000_000.0)

        df = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            },
            index=dates,
        )
        return df

    def test_backtest_runs_and_computes_core_metrics(self):
        """测试回测引擎完整运行与量化指标计算"""
        n = 50
        # SPY 平稳微涨
        spy_prices = [400.0 * (1.0 + 0.001 * i) for i in range(n)]
        # AAPL 在第 20 天后强势上涨
        aapl_prices = [150.0 * (1.0 + (0.015 * (i - 20) if i >= 20 else 0.0)) for i in range(n)]

        price_data = {
            "SPY": self._create_mock_df(spy_prices),
            "AAPL": self._create_mock_df(aapl_prices),
        }

        cfg = BacktestConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-03-15",
            initial_capital=100_000.0,
            benchmark="SPY",
            min_buy_score=60,  # 允许买入触发
        )

        engine = BacktestEngine(cfg, price_data=price_data)
        result = engine.run()

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.initial_capital, 100_000.0)
        self.assertGreater(result.final_equity, 100_000.0)
        self.assertGreater(result.total_return, 0.0)
        self.assertIn("AAPL", result.config["tickers"])
        self.assertGreater(len(result.daily_history), 20)
        self.assertTrue(result.max_drawdown <= 0.0)
        self.assertIsNotNone(result.sharpe_ratio)
        self.assertGreater(len(result.trades), 0)

    def test_backtest_triggers_stop_loss(self):
        """测试当标的跌幅超过 -6% 时触发硬止损平仓"""
        n = 40
        spy_prices = [400.0 for _ in range(n)]
        # AAPL 前 25 天横盘/微涨，触发买入后暴跌 10%
        aapl_prices = [100.0 + (i * 0.2 if i < 25 else (25 * 0.2 - (i - 25) * 2.5)) for i in range(n)]

        price_data = {
            "SPY": self._create_mock_df(spy_prices),
            "AAPL": self._create_mock_df(aapl_prices),
        }

        cfg = BacktestConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-02-28",
            initial_capital=100_000.0,
            benchmark="SPY",
            min_buy_score=50,
            stop_loss=-0.06,
        )

        engine = BacktestEngine(cfg, price_data=price_data)
        result = engine.run()

        stop_loss_trades = [t for t in result.trades if t["exit_reason"] == "STOP_LOSS"]
        self.assertGreater(len(stop_loss_trades), 0, "必须记录硬止损平仓记录")

    def test_backtest_triggers_trailing_stop(self):
        """测试标的自历史最高点回撤 >= 5% 时触发移动止损平仓"""
        n = 50
        spy_prices = [400.0 for _ in range(n)]
        # AAPL 在第 25 天到 35 天大涨到 130，随后回调到 120 (回撤 > 5%)
        aapl_prices = []
        for i in range(n):
            if i < 25:
                p = 100.0 + i * 0.2
            elif i < 35:
                p = 105.0 + (i - 25) * 2.5  # 冲高到 130
            else:
                p = 130.0 - (i - 35) * 1.5  # 回落
            aapl_prices.append(p)

        price_data = {
            "SPY": self._create_mock_df(spy_prices),
            "AAPL": self._create_mock_df(aapl_prices),
        }

        cfg = BacktestConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-03-15",
            initial_capital=100_000.0,
            benchmark="SPY",
            min_buy_score=50,
            trailing_stop=0.05,
        )

        engine = BacktestEngine(cfg, price_data=price_data)
        result = engine.run()

        trailing_trades = [t for t in result.trades if t["exit_reason"] == "TRAILING_STOP"]
        self.assertGreater(len(trailing_trades), 0, "必须成功触发移动止损")

    def test_ascii_chart_rendering(self):
        """测试终端 ASCII 走势图格式渲染输出"""
        n = 30
        spy_prices = [400.0 * (1.0 + 0.002 * i) for i in range(n)]
        aapl_prices = [150.0 * (1.0 + 0.003 * i) for i in range(n)]
        price_data = {
            "SPY": self._create_mock_df(spy_prices),
            "AAPL": self._create_mock_df(aapl_prices),
        }
        cfg = BacktestConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-02-15",
            initial_capital=100_000.0,
            min_buy_score=50,
        )
        engine = BacktestEngine(cfg, price_data=price_data)
        result = engine.run()
        chart = result.render_ascii_chart(width=40, height=8)

        self.assertIn("策略净值走势图", chart)
        self.assertIn("★", chart)
        self.assertIn("·", chart)


if __name__ == "__main__":
    unittest.main()
