import unittest

from smg_strategy.scoring import qualifies_long, qualifies_short, score_long, score_short


class ScoringTests(unittest.TestCase):
    def test_weighted_long_score(self):
        score = score_long(80, 75, 70, 60, 50, 90, 5)
        self.assertAlmostEqual(score, 66.0)

    def test_scores_are_clamped(self):
        self.assertEqual(score_long(100, 100, 100, 100, 100, 100, -20), 100)
        self.assertEqual(score_long(0, 0, 0, 0, 0, 0, 20), 0)

    def test_long_threshold_is_72(self):
        self.assertFalse(qualifies_long(71.99))
        self.assertTrue(qualifies_long(72))

    def test_short_inverts_bullish_inputs_and_requires_weak_market(self):
        bullish_score = score_short(80, 80, 80, 80, 80, 80, 0)
        bearish_score = score_short(10, 10, 10, 10, 10, 90, 0)
        self.assertLess(bullish_score, 78)
        self.assertGreaterEqual(bearish_score, 78)
        self.assertFalse(qualifies_short(bearish_score, market_weak=False))
        self.assertTrue(qualifies_short(bearish_score, market_weak=True))


if __name__ == "__main__":
    unittest.main()
