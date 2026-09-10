from dataclasses import dataclass
from typing import List, Tuple

from .models import Snapshot
from .risk import risk_state, size_position
from .rules import validate_candidate
from .scoring import qualifies_long, qualifies_short, score_long, score_short


@dataclass(frozen=True)
class Proposal:
    symbol: str
    side: str
    decision: str
    score: float
    shares: int
    entry: float
    stop: float
    market_value: float
    risk_dollars: float
    exposure_after: float
    rule_reasons: Tuple[str, ...]
    catalyst_source: str
    invalidation: str


def build_proposals(snapshot: Snapshot, market_weak: bool = False) -> List[Proposal]:
    state = risk_state(snapshot.account.peak_equity, snapshot.account.equity)
    gross_exposure = sum(abs(position.market_value) for position in snapshot.positions)
    proposals = []

    for candidate in snapshot.candidates:
        existing_symbol_value = sum(
            abs(position.market_value)
            for position in snapshot.positions
            if position.symbol == candidate.symbol
        )
        sector_exposure = sum(
            abs(position.market_value)
            for position in snapshot.positions
            if position.sector == candidate.sector
        )
        sizing = size_position(
            equity=snapshot.account.equity,
            buying_power=snapshot.account.buying_power,
            price=candidate.price,
            stop_percent=candidate.stop_percent,
            existing_symbol_value=existing_symbol_value,
            sector_exposure=sector_exposure,
            current_gross_exposure=gross_exposure,
            state=state,
        )
        components = (
            candidate.relative_strength,
            candidate.trend_quality,
            candidate.volume_confirmation,
            candidate.catalyst_quality,
            candidate.earnings_revision,
            candidate.liquidity,
            candidate.risk_penalty,
        )
        if candidate.side == "long":
            score = score_long(*components)
            score_passes = qualifies_long(score)
            stop = candidate.price * (1 - candidate.stop_percent)
        else:
            score = score_short(*components)
            score_passes = qualifies_short(score, market_weak)
            stop = candidate.price * (1 + candidate.stop_percent)

        rule_reasons = validate_candidate(
            candidate.side,
            candidate.exchange,
            candidate.price,
            candidate.previous_close,
            candidate.market_cap,
            sizing.shares,
            candidate.asset_type,
        )
        reasons = list(rule_reasons) + list(sizing.reasons)
        if candidate.data_conflict:
            reasons.append("data_conflict")
        if not score_passes:
            reasons.append("score_threshold")
        unique_reasons = tuple(dict.fromkeys(reasons))
        decision = "APPROVE_FOR_REVIEW" if not unique_reasons else "HOLD"
        proposals.append(Proposal(
            symbol=candidate.symbol,
            side=candidate.side,
            decision=decision,
            score=round(score, 2),
            shares=sizing.shares,
            entry=candidate.price,
            stop=round(stop, 4),
            market_value=sizing.market_value,
            risk_dollars=sizing.risk_dollars,
            exposure_after=gross_exposure + sizing.market_value,
            rule_reasons=unique_reasons,
            catalyst_source=candidate.catalyst_source,
            invalidation=candidate.invalidation,
        ))
    return proposals


def render_review_packet(snapshot: Snapshot, proposals: List[Proposal]) -> str:
    state = risk_state(snapshot.account.peak_equity, snapshot.account.equity)
    drawdown = (snapshot.account.peak_equity - snapshot.account.equity) / snapshot.account.peak_equity
    lines = [
        "# SMG Trade Review Packet",
        "",
        f"- Data timestamp: {snapshot.as_of}",
        f"- Account equity: ${snapshot.account.equity:,.2f}",
        f"- Buying power: ${snapshot.account.buying_power:,.2f}",
        f"- Drawdown state: {state} ({drawdown:.2%})",
        f"- Successful long trades: {snapshot.account.successful_long_trades}/3 minimum",
        "",
    ]
    if not proposals:
        lines.extend(["## No qualifying candidates", "", "Decision: HOLD", ""])
    for proposal in proposals:
        rules = "PASS" if not proposal.rule_reasons else ", ".join(proposal.rule_reasons)
        lines.extend([
            f"## {proposal.symbol} — {proposal.side.upper()}",
            "",
            f"- Decision: {proposal.decision}",
            f"- Score: {proposal.score:.2f}",
            f"- Rule check: {rules}",
            f"- Entry: ${proposal.entry:,.4f}",
            f"- Stop: ${proposal.stop:,.4f}",
            f"- Shares: {proposal.shares}",
            f"- Market value: ${proposal.market_value:,.2f}",
            f"- Risk dollars: ${proposal.risk_dollars:,.2f}",
            f"- Gross exposure after trade: ${proposal.exposure_after:,.2f}",
            f"- Catalyst source: {proposal.catalyst_source or 'Missing'}",
            f"- Invalidation: {proposal.invalidation or 'Missing'}",
            "- Claude verdict: PENDING",
            "- Antigravity verdict: PENDING",
            "",
        ])
    return "\n".join(lines)
