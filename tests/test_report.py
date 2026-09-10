import json
import tempfile
import unittest
from pathlib import Path

from smg_strategy.models import Snapshot
from smg_strategy.report import build_proposals, render_review_packet
from strategy_cli import generate_report


def strong_snapshot():
    return {
        "as_of": "2026-06-30T15:45:00-07:00",
        "account": {
            "equity": 100002.08,
            "buying_power": 150003.12,
            "cash": 100002.08,
            "peak_equity": 100002.08,
            "successful_long_trades": 0,
        },
        "positions": [],
        "candidates": [{
            "symbol": "TEST", "side": "long", "price": 100,
            "previous_close": 99, "market_cap": 1_000_000_000,
            "exchange": "NASDAQ", "asset_type": "stock",
            "average_daily_dollar_volume": 50_000_000,
            "relative_strength": 95, "trend_quality": 90,
            "volume_confirmation": 90, "catalyst_quality": 85,
            "earnings_revision": 80, "liquidity": 95,
            "risk_penalty": 2, "stop_percent": 0.06,
            "sector": "Technology",
            "catalyst_source": "https://example.com/source",
            "invalidation": "Closes below support",
        }],
    }


class ReportTests(unittest.TestCase):
    def test_builds_approved_review_proposal(self):
        snapshot = Snapshot.from_dict(strong_snapshot())
        proposals = build_proposals(snapshot, market_weak=False)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].decision, "APPROVE_FOR_REVIEW")
        self.assertGreaterEqual(proposals[0].shares, 10)
        self.assertEqual(proposals[0].rule_reasons, ())

    def test_packet_contains_audit_fields_and_no_password_field(self):
        snapshot = Snapshot.from_dict(strong_snapshot())
        text = render_review_packet(snapshot, build_proposals(snapshot))
        for label in (
            "Data timestamp", "Account equity", "Successful long trades",
            "Rule check", "Entry", "Stop", "Shares", "Invalidation",
            "Claude verdict", "Antigravity verdict",
        ):
            self.assertIn(label, text)
        self.assertNotIn("password", text.lower())

    def test_conflicting_market_data_forces_hold(self):
        data = strong_snapshot()
        data["candidates"][0]["data_conflict"] = True
        proposal = build_proposals(Snapshot.from_dict(data))[0]
        self.assertEqual(proposal.decision, "HOLD")
        self.assertIn("data_conflict", proposal.rule_reasons)

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "snapshot.json"
            output = root / "report.md"
            source.write_text(json.dumps(strong_snapshot()), encoding="utf-8")
            generate_report(source, output, market_weak=False)
            self.assertTrue(output.exists())
            self.assertIn("TEST", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
