import unittest

from smg_strategy.rules import validate_candidate


class RuleTests(unittest.TestCase):
    def test_rejects_unsupported_exchange(self):
        reasons = validate_candidate("long", "OTC", 10, 10, 1e9, 100)
        self.assertIn("exchange", reasons)

    def test_rejects_low_current_or_previous_price(self):
        for price, previous_close in ((2.99, 3.10), (3.10, 2.99)):
            with self.subTest(price=price, previous_close=previous_close):
                reasons = validate_candidate("long", "NASDAQ", price, previous_close, 1e9, 100)
                self.assertIn("price", reasons)

    def test_rejects_small_market_cap_and_opening_order(self):
        self.assertIn("market_cap", validate_candidate("long", "NYSE", 20, 20, 24_999_999, 10))
        self.assertIn("shares", validate_candidate("short", "NYSE", 20, 20, 1e9, 9))

    def test_allows_stock_short_but_not_mutual_fund_or_bond_short(self):
        self.assertEqual([], validate_candidate("short", "NYSE", 20, 20, 1e9, 10, "stock"))
        for asset_type in ("mutual_fund", "bond"):
            with self.subTest(asset_type=asset_type):
                reasons = validate_candidate("short", "NYSE", 20, 20, 1e9, 10, asset_type)
                self.assertIn("asset_type", reasons)


if __name__ == "__main__":
    unittest.main()
