import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src/ is on sys.path regardless of execution mode or runner
_src = str(Path(__file__).resolve().parent.parent / "src")
_root = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from smg_strategy.quant_engine import run_monte_carlo
from smg_strategy.hourly_quant_decision import mc_stop_loss_risk


class TestEnvironmentIsolation(unittest.TestCase):
    """
    Regression test suite enforcing:
    1. Zero state pollution under user HOME or .gemini/.antigravity
    2. No hard-coded personal usernames or scratch paths in source files
    3. Deterministic Monte Carlo simulation repeatability with fixed seeds
    """

    def test_no_hardcoded_personal_paths_in_src(self):
        """Source files must not contain personal developer paths or usernames."""
        src_dir = Path(__file__).resolve().parent.parent / "src"
        forbidden_patterns = [
            "~/.gemini/antigravity/scratch",
            "/Users/timexingxin",
            "Kings-MacBook-Air",
        ]
        
        violations = []
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                if pattern in content:
                    violations.append(f"{py_file.name}: contains forbidden pattern '{pattern}'")
        
        self.assertEqual(violations, [], f"Found hardcoded personal paths in source:\n" + "\n".join(violations))

    def test_test_suite_does_not_pollute_home(self):
        """Running isolated functions under custom SMG_STATE_DIR must not touch HOME."""
        with tempfile.TemporaryDirectory() as fake_home:
            with tempfile.TemporaryDirectory() as fake_state:
                orig_env = os.environ.copy()
                try:
                    os.environ["HOME"] = fake_home
                    os.environ["SMG_STATE_DIR"] = fake_state
                    os.environ["SMG_DISABLE_FILE_LOG"] = "1"

                    # Import and exercise ticker locks with custom state dir
                    import smg_strategy.ticker_locks as tl
                    tl.SMG_DIR = fake_state
                    tl.LOCKS_PATH = os.path.join(fake_state, "ticker_locks.json")
                    tl.record("TEST_SYMBOL", "TEST_ERROR")
                    
                    # Verify lock file was written inside fake_state, not fake_home
                    self.assertTrue(os.path.exists(tl.LOCKS_PATH))
                    self.assertFalse(os.path.exists(os.path.join(fake_home, ".gemini")))
                    self.assertFalse(os.path.exists(os.path.join(fake_home, ".antigravity")))
                finally:
                    os.environ.clear()
                    os.environ.update(orig_env)

    def test_monte_carlo_deterministic_seed(self):
        """Monte Carlo simulations must produce identical outputs given the same seed."""
        # Fix seed and run two simulations with identical parameters
        params = {"daily_drift": 0.0003, "daily_vol": 0.015}
        res1 = run_monte_carlo(
            ticker="TEST",
            score=80,
            entry_price=100.0,
            market_params=params,
            n_paths=1000,
            horizon_days=30,
            seed=42,
        )
        res2 = run_monte_carlo(
            ticker="TEST",
            score=80,
            entry_price=100.0,
            market_params=params,
            n_paths=1000,
            horizon_days=30,
            seed=42,
        )

        self.assertEqual(res1.prob_stopped, res2.prob_stopped)
        self.assertAlmostEqual(res1.expected_return, res2.expected_return, places=6)
        self.assertAlmostEqual(res1.median_return, res2.median_return, places=6)
        self.assertAlmostEqual(res1.prob_profit, res2.prob_profit, places=6)
        self.assertAlmostEqual(res1.sharpe_ratio, res2.sharpe_ratio, places=6)

    def test_monte_carlo_identical_seed_repeatability(self):
        """Monte Carlo simulations must produce identical outputs given the same seed across all fields."""
        params = {"daily_drift": 0.0003, "daily_vol": 0.015}
        res1 = run_monte_carlo(
            ticker="TEST", score=80, entry_price=100.0,
            market_params=params, n_paths=1000, horizon_days=30, seed=42,
        )
        res2 = run_monte_carlo(
            ticker="TEST", score=80, entry_price=100.0,
            market_params=params, n_paths=1000, horizon_days=30, seed=42,
        )
        self.assertEqual(res1.prob_stopped, res2.prob_stopped)
        self.assertEqual(res1.expected_return, res2.expected_return)
        self.assertEqual(res1.median_return, res2.median_return)
        self.assertEqual(res1.prob_profit, res2.prob_profit)
        self.assertEqual(res1.sharpe_ratio, res2.sharpe_ratio)
        self.assertEqual(res1.p5_return, res2.p5_return)
        self.assertEqual(res1.p95_return, res2.p95_return)
        self.assertEqual(res1.max_dd_avg, res2.max_dd_avg)
        self.assertEqual(res1, res2)

        # Verify hourly_quant_decision mc_stop_loss_risk repeatability
        mc1 = mc_stop_loss_risk(
            cost_per_share=100.0, current_price=100.0,
            sigma_daily=0.015, mu_daily=0.0003, horizon_days=30, n_paths=1000, seed=42,
        )
        mc2 = mc_stop_loss_risk(
            cost_per_share=100.0, current_price=100.0,
            sigma_daily=0.015, mu_daily=0.0003, horizon_days=30, n_paths=1000, seed=42,
        )
        self.assertEqual(mc1, mc2)

    def test_monte_carlo_seed_variance(self):
        """Monte Carlo simulations with different seeds must produce divergent results."""
        params = {"daily_drift": 0.0003, "daily_vol": 0.015}
        res1 = run_monte_carlo(
            ticker="TEST", score=80, entry_price=100.0,
            market_params=params, n_paths=1000, horizon_days=30, seed=42,
        )
        res2 = run_monte_carlo(
            ticker="TEST", score=80, entry_price=100.0,
            market_params=params, n_paths=1000, horizon_days=30, seed=2026,
        )
        self.assertNotEqual(res1.expected_return, res2.expected_return)

        mc1 = mc_stop_loss_risk(
            cost_per_share=100.0, current_price=100.0,
            sigma_daily=0.015, mu_daily=0.0003, horizon_days=30, n_paths=1000, seed=42,
        )
        mc2 = mc_stop_loss_risk(
            cost_per_share=100.0, current_price=100.0,
            sigma_daily=0.015, mu_daily=0.0003, horizon_days=30, n_paths=1000, seed=2026,
        )
        self.assertNotEqual(mc1["expected_return"], mc2["expected_return"])

    def test_monte_carlo_global_rng_state_immutability(self):
        """Calling Monte Carlo must never alter or pollute global np.random state."""
        np.random.seed(987654321)
        state_before = np.random.get_state()
        reference_draws = [np.random.rand() for _ in range(5)]
        np.random.set_state(state_before)

        params = {"daily_drift": 0.0003, "daily_vol": 0.015}
        _ = run_monte_carlo(
            ticker="TEST", score=80, entry_price=100.0,
            market_params=params, n_paths=500, horizon_days=20, seed=42,
        )
        _ = run_monte_carlo(
            ticker="TEST", score=80, entry_price=100.0,
            market_params=params, n_paths=500, horizon_days=20, seed=None,
        )
        _ = mc_stop_loss_risk(
            cost_per_share=100.0, current_price=100.0,
            sigma_daily=0.015, mu_daily=0.0003, horizon_days=20, n_paths=500, seed=99,
        )
        _ = mc_stop_loss_risk(
            cost_per_share=100.0, current_price=100.0,
            sigma_daily=0.015, mu_daily=0.0003, horizon_days=20, n_paths=500, seed=None,
        )

        state_after = np.random.get_state()
        self.assertEqual(state_before[0], state_after[0])
        self.assertTrue(np.array_equal(state_before[1], state_after[1]))
        self.assertEqual(state_before[2:], state_after[2:])

        actual_draws = [np.random.rand() for _ in range(5)]
        self.assertEqual(reference_draws, actual_draws)


if __name__ == "__main__":
    unittest.main()
