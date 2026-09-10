from typing import List


def validate_candidate(
    side: str,
    exchange: str,
    price: float,
    previous_close: float,
    market_cap: float,
    shares: int,
    asset_type: str = "stock",
    opening: bool = True,
) -> List[str]:
    """Return stable SMG rule violation codes for a proposed opening trade."""
    reasons = []
    side = side.lower()
    exchange = exchange.upper()
    asset_type = asset_type.lower()

    if side not in {"long", "short"}:
        reasons.append("side")
    if exchange not in {"NYSE", "NASDAQ"}:
        reasons.append("exchange")
    if opening and (price < 3 or previous_close < 3):
        reasons.append("price")
    if market_cap < 25_000_000:
        reasons.append("market_cap")
    if opening and shares < 10:
        reasons.append("shares")
    if side == "short" and asset_type in {"mutual_fund", "bond"}:
        reasons.append("asset_type")
    return reasons
