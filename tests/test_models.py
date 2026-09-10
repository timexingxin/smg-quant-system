import unittest

from smg_strategy.models import Snapshot


def valid_snapshot():
    return {
        "as_of": "2026-06-30T15:45:00-07:00",
        "account": {
            "equity": 100002.08,
            "buying_power": 150003.12,
            "cash": 100002.08,
            "peak_equity": 100002.08,
        },
        "positions": [],
        "candidates": [
            {
                "symbol": "TEST",
                "side": "long",
                "price": 100,
                "previous_close": 99,
                "market_cap": 1_000_000_000,
                "exchange": "NASDAQ",
                "asset_type": "stock",
                "average_daily_dollar_volume": 50_000_000,
                "relative_strength": 80,
                "trend_quality": 75,
                "volume_confirmation": 70,
                "catalyst_quality": 60,
                "earnings_revision": 50,
                "liquidity": 90,
                "risk_penalty": 5,
                "stop_percent": 0.06,
                "sector": "Technology",
                "catalyst_source": "https://example.com/source",
                "invalidation": "Closes below support",
            }
        ],
    }


class SnapshotTests(unittest.TestCase):
    def test_parses_account_and_candidate(self):
        snapshot = Snapshot.from_dict(valid_snapshot())
        self.assertEqual(snapshot.account.equity, 100002.08)
        self.assertEqual(snapshot.account.successful_long_trades, 0)
        self.assertEqual(snapshot.candidates[0].symbol, "TEST")

    def test_rejects_missing_timestamp(self):
        value = valid_snapshot()
        del value["as_of"]
        with self.assertRaisesRegex(ValueError, "as_of"):
            Snapshot.from_dict(value)

    def test_rejects_invalid_account_values(self):
        for field, value in (("equity", 0), ("buying_power", -1)):
            data = valid_snapshot()
            data["account"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    Snapshot.from_dict(data)

    def test_rejects_invalid_side_and_stop(self):
        data = valid_snapshot()
        data["candidates"][0]["side"] = "option"
        with self.assertRaisesRegex(ValueError, "side"):
            Snapshot.from_dict(data)

        data = valid_snapshot()
        data["candidates"][0]["stop_percent"] = 0.30
        with self.assertRaisesRegex(ValueError, "stop_percent"):
            Snapshot.from_dict(data)


if __name__ == "__main__":
    unittest.main()
