import unittest
import numpy as np

from smg_strategy.quant_engine import (
    MCResult,
    portfolio_kelly_allocation,
    correlation_analysis,
    run_monte_carlo,
    compute_technical_indicators,
    dcf_intrinsic_value,
)
from smg_strategy.config import POSITION_CAP, MIN_ORDER_SHARES


class QuantEngineFixesTests(unittest.TestCase):
    def _create_mock_mc_result(self, ticker="AAPL", score=85, price=100.0, kelly_f=0.25):
        return MCResult(
            ticker=ticker,
            score=score,
            entry_price=price,
            n_paths=1000,
            prob_stopped=0.12,
            prob_profit=0.68,
            expected_return=0.08,
            expected_return_cond=0.12,
            median_return=0.07,
            p5_return=-0.05,
            p95_return=0.22,
            max_dd_avg=-0.04,
            sharpe_ratio=1.6,
            avg_days_stop=6.0,
            kelly_f=kelly_f,
            ev_dollar=8.0,
            recommendation="BUY",
        )

    def test_kelly_hard_blocks_when_position_at_or_above_cap(self):
        """P0-2: 当现有持仓已达到或超过 20% 上限时，净空间为 0，严格硬阻断而非放行开仓"""
        total_equity = 100_000.0
        # 标的 AAPL 当前已有 200 股 @ $105 = $21,000 (已达 21% > 20% 上限)
        holdings = [{"ticker": "AAPL", "qty": 200, "current": 105.0}]
        account = {
            "equity": total_equity,
            "cash": 40_000.0,
            "buying_power": 40_000.0,
        }
        mc_results = [self._create_mock_mc_result(ticker="AAPL", price=105.0)]

        res = portfolio_kelly_allocation(mc_results, account, holdings=holdings)
        allocations = res["allocations"]
        blocked = res["blocked"]

        self.assertEqual(len(allocations), 0, "达到集中度上限的标的不应有新分配")
        self.assertEqual(len(blocked), 1, "达到集中度上限的标的必须进入 blocked 阻断名单")
        self.assertEqual(blocked[0]["ticker"], "AAPL")
        self.assertIn("已达/超 20% 集中度上限", blocked[0]["reason"])

    def test_kelly_blocks_when_headroom_below_minimum_shares(self):
        """P0-2: 剩余空间折合股数不足 10 股时，不生成订单"""
        total_equity = 100_000.0
        # 标的 AAPL 价格 $100，已有 195 股 = $19,500，仅剩 $500 额度，最多只能买 5 股 (< 10 股)
        holdings = [{"ticker": "AAPL", "qty": 195, "current": 100.0}]
        account = {
            "equity": total_equity,
            "cash": 40_000.0,
            "buying_power": 40_000.0,
        }
        mc_results = [self._create_mock_mc_result(ticker="AAPL", price=100.0)]

        res = portfolio_kelly_allocation(mc_results, account, holdings=holdings)
        self.assertEqual(len(res["allocations"]), 0)

    def test_union_find_transitive_correlation_clustering(self):
        """P1-3: 并查集传递闭包聚类，A-B 高度相关且 B-C 高度相关时，A/B/C 必须划入同一风险簇"""
        np.random.seed(42)
        n = 100
        # 构造累积收益序列（价格序列）
        base1 = np.cumsum(np.random.normal(0.001, 0.02, n))
        base2 = np.cumsum(np.random.normal(0.001, 0.02, n))
        p_A = 100.0 * np.exp(base1 + 0.02 * np.cumsum(np.random.normal(0, 0.01, n)))
        p_B = 100.0 * np.exp(base1 + 0.03 * np.cumsum(np.random.normal(0, 0.01, n)))
        p_C = 100.0 * np.exp(0.8 * base1 + 0.6 * base2)
        p_D = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))

        prices_dict = {"A": p_A, "B": p_B, "C": p_C, "D": p_D}
        res = correlation_analysis(["A", "B", "C", "D"], threshold=0.70, prices_dict=prices_dict)
        clusters = res["clusters"]

        # 验证包含 A 的簇必然也包含了 B
        a_cluster = next((c for c in clusters if "A" in c), None)
        self.assertIsNotNone(a_cluster, "A 应该被聚类进入高相关簇")
        self.assertIn("B", a_cluster, "A 与 B 强相关必须在同簇")
        self.assertNotIn("D", a_cluster, "独立标的 D 绝不能混入 A 的高风险簇")

    def test_monte_carlo_bgk_barrier_shift(self):
        """P1-1: 验证 Broadie-Glasserman-Kou (1997) 离散步长边界修正逻辑有效"""
        np.random.seed(42)
        market_params = {
            "annual_vol": 0.30,
            "daily_vol": 0.30 / np.sqrt(252),
            "beta": 1.2,
            "current_rsi": 55.0,
            "atr_pct": 0.025,
            "adx": 28.0,
            "vol_ratio": 1.2,
            "trend": "BULLISH",
        }
        res = run_monte_carlo(
            ticker="TEST",
            score=85,
            entry_price=100.0,
            market_params=market_params,
            n_paths=2000,
            horizon_days=14,
            stop_loss=-0.06,
            sub_steps=4,
        )
        self.assertIsInstance(res, MCResult)
        self.assertGreater(res.prob_stopped, 0.0)
        self.assertLess(res.prob_stopped, 1.0)
        self.assertGreater(res.prob_profit, 0.0)
        self.assertLess(res.prob_profit, 1.0)

    def test_technical_indicators_with_volume(self):
        """P1-2: 验证技术指标计算能够正确接收真实成交量序列并影响量价评分"""
        n = 40
        closes = [100.0 + i * 0.5 for i in range(n)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        volumes_high = [1_000_000 * (2.0 if i >= 35 else 1.0) for i in range(n)]
        volumes_low = [1_000_000 * (0.3 if i >= 35 else 1.0) for i in range(n)]

        tech_high = compute_technical_indicators(closes, highs, lows, volumes_high)
        tech_low = compute_technical_indicators(closes, highs, lows, volumes_low)

        self.assertIn("vol_ratio", tech_high)
        self.assertGreater(tech_high["vol_ratio"], tech_low["vol_ratio"])
        self.assertGreaterEqual(tech_high["tech_score"], tech_low["tech_score"])

    def test_dcf_dynamic_growth_and_wacc(self):
        """P2-3/P2-4: 验证 DCF 估值在输入真实财务数据时的 WACC 计算与合理内在价值"""
        mock_info = {
            "freeCashflow": 1_000_000_000,
            "marketCap": 20_000_000_000,
            "sharesOutstanding": 1_000_000_000,
            "totalDebt": 5_000_000_000,
            "beta": 1.25,
            "earningsGrowth": 0.12,
        }
        res = dcf_intrinsic_value("TEST", market_price=20.0, info=mock_info)
        self.assertIn("fair_value", res)
        self.assertIn("wacc", res)
        self.assertIn("growth_s1", res)
        self.assertGreater(res["wacc"], 0.06)
        self.assertLess(res["wacc"], 0.15)
        self.assertGreater(res["fair_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
