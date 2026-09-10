import unittest
import pandas as pd
import numpy as np

from smg_strategy.backtest import BacktestConfig, BacktestEngine, BacktestResult, run_walk_forward


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
            trailing_stop=0.50,  # 隔离测试硬止损，避免移动止损先触发
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

    def test_next_day_open_execution(self):
        """测试无前视偏差：信号在当日收盘产生，成交必须在次日开盘价 (t+1 Open)"""
        dates = pd.date_range("2024-01-01", periods=25, freq="B")
        spy_df = pd.DataFrame({"Close": [400.0]*25, "Open": [400.0]*25, "High": [401.0]*25, "Low": [399.0]*25, "Volume": [1e6]*25}, index=dates)

        # Day 0-20 横盘，Day 20 收盘打分触发买入；Day 21 Open 价格为 105.0
        opens = [100.0]*21 + [105.0] + [110.0]*3
        closes = [100.0]*21 + [110.0] + [110.0]*3
        highs = [100.0]*21 + [112.0] + [110.0]*3
        lows = [100.0]*21 + [104.0] + [110.0]*3

        aapl_df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1e6]*25}, index=dates)

        cfg = BacktestConfig(tickers=["AAPL"], start_date="2024-01-01", end_date="2024-02-05", min_buy_score=40)
        eng = BacktestEngine(cfg, price_data={"SPY": spy_df, "AAPL": aapl_df})
        res = eng.run()

        day_21_str = dates[21].strftime("%Y-%m-%d")
        trade = res.trades[0]
        self.assertEqual(trade["entry_date"], day_21_str, "入场日期必须是次日 (t+1)")
        # 买入价为次日开盘价 + 滑点
        expected_entry = 105.0 * (1.0 + cfg.slippage_bps / 10000.0)
        self.assertAlmostEqual(trade["entry_price"], expected_entry, places=3)

    def test_gap_down_stop_loss_penalty(self):
        """测试跳空低开跌破止损线时，按开盘价止损而非按理想止损线乐观止损"""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        spy_df = pd.DataFrame({"Close": [400.0]*30, "Open": [400.0]*30, "High": [401.0]*30, "Low": [399.0]*30, "Volume": [1e6]*30}, index=dates)

        # 前21天100元微涨，第22天开盘买入(约104.2)，第23天跳空低开到90元 (低开超过13%)
        closes = [100.0 + 0.2*i for i in range(22)] + [90.0]*8
        opens = [100.0 + 0.2*i for i in range(22)] + [90.0]*8
        highs = [c * 1.01 for c in closes]
        lows = [c * 0.99 for c in closes]
        aapl_df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1e6]*30}, index=dates)

        cfg = BacktestConfig(tickers=["AAPL"], start_date="2024-01-01", end_date="2024-02-15", min_buy_score=50, stop_loss=-0.06, trailing_stop=0.5)
        eng = BacktestEngine(cfg, price_data={"SPY": spy_df, "AAPL": aapl_df})
        res = eng.run()

        trade = res.trades[0]
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        expected_exit = 90.0 * (1.0 - cfg.slippage_bps / 10000.0)
        self.assertAlmostEqual(trade["exit_price"], expected_exit, places=2)

    def test_calendar_alignment_robustness(self):
        """测试多标的在交易日不完全对齐时能安全 reindex 到基准日历"""
        spy_dates = pd.date_range("2024-01-01", periods=30, freq="B")
        spy_df = pd.DataFrame({"Close": [400.0]*30, "Open": [400.0]*30, "High": [401.0]*30, "Low": [399.0]*30, "Volume": [1e6]*30}, index=spy_dates)

        # 标的 AAPL 缺失第 10 到第 15 个交易日
        aapl_dates = spy_dates[:10].append(spy_dates[15:])
        aapl_closes = [100.0]*len(aapl_dates)
        aapl_df = pd.DataFrame({"Close": aapl_closes, "Open": aapl_closes, "High": [c*1.01 for c in aapl_closes], "Low": [c*0.99 for c in aapl_closes], "Volume": [1e6]*len(aapl_dates)}, index=aapl_dates)

        cfg = BacktestConfig(tickers=["AAPL"], start_date="2024-01-01", end_date="2024-02-15", min_buy_score=80)
        eng = BacktestEngine(cfg, price_data={"SPY": spy_df, "AAPL": aapl_df})
        res = eng.run()
        self.assertEqual(len(res.daily_history), 30)

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

    def test_walk_forward_execution(self):
        """测试滚动 Walk-Forward 样本内外切分与抗过拟合指标生成"""
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        spy_df = pd.DataFrame({"Close": [400.0 + i*0.1 for i in range(90)], "Open": [400.0 + i*0.1 for i in range(90)], "High": [401.0 + i*0.1 for i in range(90)], "Low": [399.0 + i*0.1 for i in range(90)], "Volume": [1e6]*90}, index=dates)
        aapl_df = pd.DataFrame({"Close": [100.0 + i*0.3 for i in range(90)], "Open": [100.0 + i*0.3 for i in range(90)], "High": [101.0 + i*0.3 for i in range(90)], "Low": [99.0 + i*0.3 for i in range(90)], "Volume": [1e6]*90}, index=dates)

        wf_results = run_walk_forward(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-05-15",
            n_splits=2,
            price_data={"SPY": spy_df, "AAPL": aapl_df}
        )
        self.assertIsInstance(wf_results, list)
        self.assertGreater(len(wf_results), 0)
        self.assertIn("train_range", wf_results[0])
        self.assertIn("test_range", wf_results[0])
        self.assertIn("oos_return", wf_results[0])
        self.assertIn("oos_excess_return", wf_results[0])


if __name__ == "__main__":
    unittest.main()
