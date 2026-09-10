from dataclasses import dataclass
from typing import Any, Dict, Tuple


def _required(data: Dict[str, Any], key: str) -> Any:
    if key not in data or data[key] in (None, ""):
        raise ValueError(f"missing {key}")
    return data[key]


@dataclass(frozen=True)
class Account:
    equity: float
    buying_power: float
    cash: float
    peak_equity: float
    successful_long_trades: int = 0


@dataclass(frozen=True)
class Position:
    symbol: str
    side: str
    shares: int
    market_value: float
    sector: str = "Unknown"


@dataclass(frozen=True)
class Candidate:
    symbol: str
    side: str
    price: float
    previous_close: float
    market_cap: float
    exchange: str
    asset_type: str
    average_daily_dollar_volume: float
    relative_strength: float
    trend_quality: float
    volume_confirmation: float
    catalyst_quality: float
    earnings_revision: float
    liquidity: float
    risk_penalty: float
    stop_percent: float
    sector: str
    catalyst_source: str
    invalidation: str
    data_conflict: bool = False


@dataclass(frozen=True)
class Snapshot:
    as_of: str
    account: Account
    positions: Tuple[Position, ...]
    candidates: Tuple[Candidate, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        as_of = str(_required(data, "as_of"))
        account_data = _required(data, "account")
        equity = float(_required(account_data, "equity"))
        buying_power = float(_required(account_data, "buying_power"))
        cash = float(_required(account_data, "cash"))
        peak_equity = float(_required(account_data, "peak_equity"))
        if equity <= 0:
            raise ValueError("equity must be positive")
        if buying_power < 0:
            raise ValueError("buying_power cannot be negative")
        if peak_equity <= 0:
            raise ValueError("peak_equity must be positive")
        successful_long_trades = int(account_data.get("successful_long_trades", 0))
        if successful_long_trades < 0:
            raise ValueError("successful_long_trades cannot be negative")
        account = Account(equity, buying_power, cash, peak_equity, successful_long_trades)

        positions = []
        for item in data.get("positions", []):
            positions.append(Position(
                symbol=str(_required(item, "symbol")).upper(),
                side=str(_required(item, "side")).lower(),
                shares=int(_required(item, "shares")),
                market_value=float(_required(item, "market_value")),
                sector=str(item.get("sector", "Unknown")),
            ))

        candidates = []
        for item in data.get("candidates", []):
            side = str(_required(item, "side")).lower()
            if side not in {"long", "short"}:
                raise ValueError("side must be long or short")
            stop_percent = float(_required(item, "stop_percent"))
            if not 0 < stop_percent <= 0.25:
                raise ValueError("stop_percent must be in (0, 0.25]")
            candidates.append(Candidate(
                symbol=str(_required(item, "symbol")).upper(),
                side=side,
                price=float(_required(item, "price")),
                previous_close=float(_required(item, "previous_close")),
                market_cap=float(_required(item, "market_cap")),
                exchange=str(_required(item, "exchange")).upper(),
                asset_type=str(item.get("asset_type", "stock")).lower(),
                average_daily_dollar_volume=float(_required(item, "average_daily_dollar_volume")),
                relative_strength=float(_required(item, "relative_strength")),
                trend_quality=float(_required(item, "trend_quality")),
                volume_confirmation=float(_required(item, "volume_confirmation")),
                catalyst_quality=float(_required(item, "catalyst_quality")),
                earnings_revision=float(_required(item, "earnings_revision")),
                liquidity=float(_required(item, "liquidity")),
                risk_penalty=float(item.get("risk_penalty", 0)),
                stop_percent=stop_percent,
                sector=str(item.get("sector", "Unknown")),
                catalyst_source=str(item.get("catalyst_source", "")),
                invalidation=str(item.get("invalidation", "")),
                data_conflict=bool(item.get("data_conflict", False)),
            ))
        return cls(as_of, account, tuple(positions), tuple(candidates))
