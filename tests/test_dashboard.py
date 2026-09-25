import unittest
from pathlib import Path
import tempfile

from smg_strategy.dashboard import (
    generate_dashboard_html,
    export_dashboard,
    get_default_quant_data
)


class DashboardTests(unittest.TestCase):
    def test_default_quant_data_structure(self):
        data = get_default_quant_data()
        self.assertIn("account", data)
        self.assertIn("holdings", data)
        self.assertIn("dcf_models", data)
        self.assertIn("monte_carlo", data)
        self.assertIn("correlation_clusters", data)
        self.assertIn("kelly_allocations", data)
        self.assertIn("backtest_summary", data)
        self.assertEqual(data["account"]["equity"], 100515.78)

    def test_generate_dashboard_html_contains_critical_components(self):
        html = generate_dashboard_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("SMG QUANT SYSTEM", html)
        self.assertIn("0.5s RISK PATROL ACTIVE", html)
        self.assertIn("PIT VERIFIED (0 LOOKAHEAD)", html)
        self.assertIn("多阶段 DCF 估值与安全边际", html)
        self.assertIn("BGK 屏障校正 100k 路径蒙特卡洛", html)
        self.assertIn("并查集传递相关性聚类", html)
        self.assertIn("Half-Kelly 仓位优化与 20% 规则硬阻断", html)
        self.assertIn("Point-in-Time 历史回测绩效看板", html)
        self.assertIn("NVDA", html)
        self.assertIn("AMD", html)

    def test_export_dashboard_creates_valid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_dash.html"
            result_path = export_dashboard(out_file)
            self.assertTrue(result_path.exists())
            content = result_path.read_text(encoding="utf-8")
            self.assertIn("SMG QUANT SYSTEM", content)
            self.assertGreater(len(content), 1000)


if __name__ == "__main__":
    unittest.main()
