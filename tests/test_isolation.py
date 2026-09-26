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

from smg_strategy.quant_engine import run_monte_carlo


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


if __name__ == "__main__":
    unittest.main()
