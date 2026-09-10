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
        """P1-3: 并查集传递闭包聚类，构建真传递链 A-B>0.7, B-C>0.7 但 A-C<0.7，验证 A/B/C 闭包同簇"""
        np.random.seed(42)
        n = 2000
        x1 = np.random.normal(0, 0.01, n)
        x2 = np.random.normal(0, 0.01, n)
        x3 = np.random.normal(0, 0.01, n)
        x4 = np.random.normal(0, 0.01, n)

        r_A = x1
        r_B = 0.82 * x1 + np.sqrt(1 - 0.82**2) * x2
        r_C = 0.82 * r_B + np.sqrt(1 - 0.82**2) * x3
        r_D = x4

        p_A = 100.0 * np.exp(np.cumsum(r_A))
        p_B = 100.0 * np.exp(np.cumsum(r_B))
        p_C = 100.0 * np.exp(np.cumsum(r_C))
        p_D = 100.0 * np.exp(np.cumsum(r_D))

        prices_dict = {"A": p_A, "B": p_B, "C": p_C, "D": p_D}
        res = correlation_analysis(["A", "B", "C", "D"], threshold=0.70, prices_dict=prices_dict)
        mat = res["corr_matrix"]
        corr_ab = mat[0][1]
        corr_bc = mat[1][2]
        corr_ac = mat[0][2]

        self.assertGreater(corr_ab, 0.70, "A与B必须强相关")
        self.assertGreater(corr_bc, 0.70, "B与C必须强相关")
        self.assertLess(corr_ac, 0.70, "A与C相关系数必须低于阈值，以检验真传递性")

        clusters = res["clusters"]
        a_cluster = next((c for c in clusters if "A" in c), None)
        self.assertIsNotNone(a_cluster, "A 必须存在于风险簇中")
        self.assertIn("B", a_cluster, "B 必须与 A 同簇")
        self.assertIn("C", a_cluster, "C 必须通过 B 的传递桥梁与 A 聚入同簇")
        self.assertNotIn("D", a_cluster, "独立标的 D 绝不能混入该簇")

    def test_monte_carlo_bgk_barrier_shift(self):
        """P1-1: 验证 Broadie-Glasserman-Kou (1997) 修正后模拟结果与连续布朗运动首达时解析解误差 < 1%"""
        import math
        np.random.seed(42)
        mu_d = 0.0003
        sigma_d = 0.30 / np.sqrt(252)
        horizon_days = 14
        stop_loss = -0.06

        # 连续布朗运动首达时理论解析解 (Analytical Hitting Time)
        nu = mu_d - 0.5 * (sigma_d ** 2)
        B = math.log(1.0 + stop_loss)
        T = horizon_days
        denom = sigma_d * math.sqrt(T)

        d1 = (B - nu * T) / denom
        d2 = (B + nu * T) / denom

        def phi(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        p_analytical = phi(d1) + math.exp((2.0 * nu * B) / (sigma_d ** 2)) * phi(d2)

        market_params = {
            "mu_daily": mu_d,
            "sigma_daily": sigma_d,
            "daily_vol": sigma_d,
            "annual_vol": 0.30,
        }
        res = run_monte_carlo(
            ticker="TEST",
            score=85,
            entry_price=100.0,
            market_params=market_params,
            n_paths=30000,
            horizon_days=horizon_days,
            stop_loss=stop_loss,
            sub_steps=4,
        )
        self.assertIsInstance(res, MCResult)
        abs_err = abs(res.prob_stopped - p_analytical)
        self.assertLess(abs_err, 0.01, f"BGK 修正模拟值 {res.prob_stopped:.4f} 与解析解 {p_analytical:.4f} 绝对误差超出 1pp")

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
        self.assertLess(res["wacc"], 0.20)
        self.assertGreater(res["fair_value"], 0.0)

    def test_dcf_mos_bounding_and_applicability(self):
        """P2-1/P2-2: 验证 DCF 安全边际截断 [-1.0, 1.0] 以及负现金流成长股 is_applicable 标记"""
        # 负现金流成长股
        info_neg = {"freeCashflow": -500_000_000, "marketCap": 20_000_000_000, "sharesOutstanding": 1_000_000_000, "totalDebt": 0}
        res_neg = dcf_intrinsic_value("GROWTH", market_price=50.0, info=info_neg)
        self.assertFalse(res_neg["is_applicable"], "负自由现金流标的必须标记 is_applicable=False")
        self.assertEqual(res_neg["dcf_signal"], "NOT_APPLICABLE")
        self.assertEqual(res_neg["mos"], -1.0)

        # 极端估值偏差截断检验
        info_extreme = {"freeCashflow": 1000, "marketCap": 1_000_000_000, "sharesOutstanding": 1_000_000, "totalDebt": 0}
        res_extreme = dcf_intrinsic_value("EXTREME", market_price=1000.0, info=info_extreme)
        self.assertGreaterEqual(res_extreme["mos"], -1.0)
        self.assertLessEqual(res_extreme["mos"], 1.0)

    def test_record_daily_buy_quota(self):
        """P0-3: 验证每日开仓限额熔断计数器及 can_buy_today 拦截"""
        import smg_strategy.risk_patrol as rp
        rp.DAILY_BUY_COUNT = 0
        rp.MAX_DAILY_BUYS = 2

        self.assertTrue(rp.can_buy_today())
        rp.record_daily_buy("AAPL")
        self.assertEqual(rp.DAILY_BUY_COUNT, 1)
        self.assertTrue(rp.can_buy_today())

        rp.record_daily_buy("NVDA")
        self.assertEqual(rp.DAILY_BUY_COUNT, 2)
        self.assertFalse(rp.can_buy_today(), "达到单日最大买入次数后必须禁止继续开仓")

    def test_pre_market_override_governance_blocks_when_core_reconciliation_fails(self):
        """P0-4: 验证越权模式 (OVERRIDE_MODE) 绝不能绕过核心对账 (sa_passed, un_passed, ae_passed)"""
        import os
        from unittest.mock import patch, mock_open
        import json
        import smg_strategy.pre_market_gate as pmg

        # 模拟核心持仓对账失败 (sa_passed=False)
        with patch.dict(os.environ, {
            "SMG_OVERRIDE_MODE": "SUPREME_EXECUTOR",
            "SMG_OVERRIDE_REASON": "Valid override test reason exceeding ten characters"
        }):
            with patch("smg_strategy.pre_market_gate.load_baseline", return_value={}), \
                 patch("smg_strategy.pre_market_gate.load_ledger", return_value=[]), \
                 patch("smg_strategy.pre_market_gate.fetch_holdings", return_value=({}, {})), \
                 patch("smg_strategy.pre_market_gate.check_self_audit", return_value=(False, 100, {}, {})), \
                 patch("smg_strategy.pre_market_gate.check_unresolved_orders", return_value=(True, 0, [])), \
                 patch("smg_strategy.pre_market_gate.check_account_equation", return_value=(True, {})), \
                 patch("smg_strategy.pre_market_gate.check_tx_history_fresh", return_value=(False, {})), \
                 patch("builtins.open", mock_open()) as m_open:
                pmg.main()
                written = "".join(call.args[0] for call in m_open().write.call_args_list)
                if written:
                    gate_doc = json.loads(written)
                    self.assertFalse(gate_doc["gate_open"], "核心对账失败时，越权模式绝不能开闸！")


if __name__ == "__main__":
    unittest.main()
