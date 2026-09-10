from typing import List

try:
    from .config import (
        MIN_STOCK_PRICE, MIN_MARKET_CAP, MIN_ORDER_SHARES
    )
except ImportError:
    from smg_strategy.config import (
        MIN_STOCK_PRICE, MIN_MARKET_CAP, MIN_ORDER_SHARES
    )


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
    if opening and (price < MIN_STOCK_PRICE or previous_close < MIN_STOCK_PRICE):
        reasons.append("price")
    if market_cap < MIN_MARKET_CAP:
        reasons.append("market_cap")
    if opening and shares < MIN_ORDER_SHARES:
        reasons.append("shares")
    if side == "short" and asset_type in {"mutual_fund", "bond"}:
        reasons.append("asset_type")
    return reasons
