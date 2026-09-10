from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PositionSize:
    shares: int
    market_value: float
    risk_dollars: float
    reasons: Tuple[str, ...] = ()


def risk_state(peak_equity: float, current_equity: float) -> str:
    if peak_equity <= 0 or current_equity < 0:
        raise ValueError("equity values must be valid")
    drawdown = (peak_equity - current_equity) / peak_equity
    if drawdown >= 0.09:
        return "STOP"
    if drawdown >= 0.075:
        return "DEFENSE"
    if drawdown >= 0.05:
        return "CAUTION"
    return "NORMAL"


def size_position(
    equity: float,
    buying_power: float,
    price: float,
    stop_percent: float,
    existing_symbol_value: float,
    sector_exposure: float = 0,
    current_gross_exposure: float = 0,
    state: str = "NORMAL",
) -> PositionSize:
    if equity <= 0 or price <= 0 or not 0 < stop_percent <= 0.25:
        raise ValueError("invalid sizing inputs")
    if state == "STOP":
        return PositionSize(0, 0, 0, ("drawdown_stop",))

    state_limits = {
        "NORMAL": (0.01, 1.25),
        "CAUTION": (0.0075, 0.90),
        "DEFENSE": (0.005, 0.50),
    }
    if state not in state_limits:
        raise ValueError("unknown risk state")
    risk_fraction, gross_multiple = state_limits[state]

    risk_budget = equity * risk_fraction
    risk_cap = risk_budget / stop_percent
    initial_cap = equity * 0.12
    symbol_cap = max(0.0, equity * 0.20 - existing_symbol_value)
    sector_cap = max(0.0, equity * 0.35 - sector_exposure)
    buying_power_cap = max(0.0, buying_power * 0.85)
    gross_cap = max(0.0, equity * gross_multiple - current_gross_exposure)
    market_value = min(
        risk_cap,
        initial_cap,
        symbol_cap,
        sector_cap,
        buying_power_cap,
        gross_cap,
    )
    shares = int(market_value // price)
    if shares < 10:
        return PositionSize(0, 0, 0, ("minimum_shares",))
    market_value = shares * price
    return PositionSize(
        shares=shares,
        market_value=market_value,
        risk_dollars=market_value * stop_percent,
    )
