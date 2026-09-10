def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def score_long(
    relative_strength: float,
    trend_quality: float,
    volume_confirmation: float,
    catalyst_quality: float,
    earnings_revision: float,
    liquidity: float,
    risk_penalty: float,
) -> float:
    score = (
        0.30 * relative_strength
        + 0.20 * trend_quality
        + 0.15 * volume_confirmation
        + 0.20 * catalyst_quality
        + 0.10 * earnings_revision
        + 0.05 * liquidity
        - risk_penalty
    )
    return _clamp(score)


def score_short(
    relative_strength: float,
    trend_quality: float,
    volume_confirmation: float,
    catalyst_quality: float,
    earnings_revision: float,
    liquidity: float,
    risk_penalty: float,
) -> float:
    return score_long(
        100 - relative_strength,
        100 - trend_quality,
        100 - volume_confirmation,
        100 - catalyst_quality,
        100 - earnings_revision,
        liquidity,
        risk_penalty,
    )


def qualifies_long(score: float) -> bool:
    return score >= 72


def qualifies_short(score: float, market_weak: bool) -> bool:
    return market_weak and score >= 78
