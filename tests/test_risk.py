import unittest

from smg_strategy.risk import risk_state, size_position


class RiskTests(unittest.TestCase):
    def test_drawdown_states(self):
        self.assertEqual("NORMAL", risk_state(100000, 96000))
        self.assertEqual("CAUTION", risk_state(100000, 94000))
        self.assertEqual("DEFENSE", risk_state(100000, 92000))
        self.assertEqual("STOP", risk_state(100000, 91000))

    def test_position_obeys_one_percent_risk_and_initial_cap(self):
        result = size_position(
            equity=100000,
            buying_power=150000,
            price=100,
            stop_percent=0.05,
            existing_symbol_value=0,
        )
        self.assertEqual(result.shares, 120)
        self.assertEqual(result.market_value, 12000)

    def test_position_obeys_absolute_symbol_and_sector_caps(self):
        symbol_limited = size_position(100000, 150000, 100, 0.05, 19000)
        self.assertEqual(symbol_limited.shares, 10)

        sector_limited = size_position(
            100000, 150000, 100, 0.05, 0, sector_exposure=34000
        )
        self.assertEqual(sector_limited.shares, 10)

    def test_leaves_fifteen_percent_buying_power_unused(self):
        result = size_position(100000, 10000, 100, 0.05, 0)
        self.assertEqual(result.shares, 85)

    def test_defense_reduces_risk_and_stop_blocks_new_positions(self):
        defense = size_position(100000, 150000, 100, 0.05, 0, state="DEFENSE")
        stopped = size_position(100000, 150000, 100, 0.05, 0, state="STOP")
        self.assertEqual(defense.shares, 100)
        self.assertEqual(stopped.shares, 0)
        self.assertIn("drawdown_stop", stopped.reasons)

    def test_rejects_opening_order_below_ten_shares(self):
        result = size_position(100000, 150000, 1000, 0.05, 19500)
        self.assertEqual(result.shares, 0)
        self.assertIn("minimum_shares", result.reasons)


if __name__ == "__main__":
    unittest.main()
