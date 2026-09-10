#!/usr/bin/env python3
"""
===========================================================================
SMG 小时级量化决策引擎 — 首席量化分析师
===========================================================================
三模型融合: DCF内在价值 + 蒙特卡洛VaR + V2.0技术评分
输出: 每笔持仓的确切操作指令 (HOLD / TIGHTEN_STOP / SELL)
===========================================================================
免责声明: 此分析仅用于 SMG 虚拟盘中学生教育目的，不构成真实投资建议。
===========================================================================
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json, sys, warnings
warnings.filterwarnings('ignore')

# =========================================================================
# SECTION 0: 当前账户与持仓快照 (来自系统)
# =========================================================================

ACCOUNT = {
    "equity": 99300.20,
    "buying_power": 86918.07,
    "cash": 37205.87,
}

# 持仓明细 (每笔独立交易)
HOLDINGS = [
    {"ticker": "AAPL", "date": "2026-07-08", "qty": 10,  "cost_basis": 3120.10,  "prev_close": 313.39, "current": 316.156, "chg_pct": 0.883},
    {"ticker": "AMD",  "date": "2026-07-07", "qty": 35,  "cost_basis": 18001.30, "prev_close": 517.405,"current": 546.686, "chg_pct": 5.659},
    {"ticker": "MSTR", "date": "2026-07-07", "qty": 39,  "cost_basis": 3812.68,  "prev_close": 93.87,  "current": 93.955,  "chg_pct": 0.091},
    {"ticker": "MU",   "date": "2026-07-06", "qty": 9,   "cost_basis": 9066.00,  "prev_close": 948.80,  "current": 991.179, "chg_pct": 4.467},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 100, "cost_basis": 19741.00, "prev_close": 204.12,  "current": 202.764, "chg_pct": -0.666},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 10,  "cost_basis": 1928.70,  "prev_close": 204.12,  "current": 202.764, "chg_pct": -0.666},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 10,  "cost_basis": 1981.80,  "prev_close": 204.12,  "current": 202.764, "chg_pct": -0.666},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 10,  "cost_basis": 1926.80,  "prev_close": 204.12,  "current": 202.764, "chg_pct": -0.666},
]

# V2.0 Scanner 结果
SCANNER = {
    "AAPL": 65, "GEV": 63, "AMD": 62, "TSLA": 58,
    "NVDA": 57, "PLTR": 57, "MSFT": 55,
}

# 硬约束
STOP_LOSS = -0.06          # -6% 死线
MIN_BUY_SCORE = 80         # 最低买入分
POSITION_CAP = 0.20        # 单仓上限 20% equity
REMAINING_DAYS = 38        # 剩余交易日 (至8月底约38天)
TRADING_DAYS_YEAR = 252
RISK_FREE = 0.0475         # 无风险利率 ~4.75%

# =========================================================================
# SECTION 1: 持仓风险矩阵 — 核心诊断
# =========================================================================

def compute_position_risk(holdings, account):
    """计算每笔持仓的精确风险指标"""
    results = []
    total_market_value = 0

    for i, h in enumerate(holdings):
        ticker = h["ticker"]
        qty = h["qty"]
        cost_total = h["cost_basis"]
        cost_per_share = cost_total / qty
        current_price = h["current"]

        # 精确盈亏
        pnl_pct = (current_price / cost_per_share) - 1.0

        # 距离止损线的 "缓冲距离"
        stop_price = cost_per_share * (1 + STOP_LOSS)
        distance_to_stop_pct = (current_price / stop_price) - 1.0

        # 集中度风险
        ticker_total_mv = sum(h2["qty"] * h2["current"] for h2 in holdings if h2["ticker"] == ticker)
        concentration = ticker_total_mv / account["equity"]

        # 持仓天数
        from datetime import datetime
        entry_date = datetime.strptime(h["date"], "%Y-%m-%d")
        holding_days = (datetime(2026, 7, 9) - entry_date).days

        results.append({
            "lot_id": i,
            "ticker": ticker,
            "qty": qty,
            "cost_per_share": cost_per_share,
            "current_price": current_price,
            "cost_total": cost_total,
            "market_value": qty * current_price,
            "pnl_dollar": qty * current_price - cost_total,
            "pnl_pct": pnl_pct,
            "stop_price": stop_price,
            "distance_to_stop_pct": distance_to_stop_pct,
            "ticker_concentration": concentration,
            "holding_days": holding_days,
        })

        total_market_value += qty * current_price

    total_pnl = total_market_value - sum(h["cost_basis"] for h in holdings)

    return results, total_market_value, total_pnl


# =========================================================================
# SECTION 2: 蒙特卡洛 VaR 模拟
# =========================================================================

def estimate_volatility(ticker: str) -> dict:
    """从 yfinance 拉取历史波动率参数"""
    try:
        import yfinance as yf
        data = yf.download(ticker, period="3mo", progress=False)
        if len(data) < 30:
            return None
        closes = data['Close'].values.flatten()
        log_returns = np.diff(np.log(closes))
        mu_daily = float(np.mean(log_returns))
        sigma_daily = float(np.std(log_returns, ddof=1))
        return {
            "mu_daily": mu_daily,
            "sigma_daily": sigma_daily,
            "mu_annual": mu_daily * TRADING_DAYS_YEAR,
            "sigma_annual": sigma_daily * np.sqrt(TRADING_DAYS_YEAR),
            "sharpe": (mu_daily * TRADING_DAYS_YEAR) / (sigma_daily * np.sqrt(TRADING_DAYS_YEAR)) if sigma_daily > 0 else 0,
        }
    except Exception:
        return None


def mc_stop_loss_risk(cost_per_share, current_price, sigma_daily, mu_daily,
                       horizon_days=38, n_paths=50000, stop_loss=-0.06):
    """蒙特卡洛: 估计 horizon_days 内触发 -6% 止损的概率"""
    if sigma_daily <= 0:
        sigma_daily = 0.02

    drift = mu_daily - 0.5 * sigma_daily**2

    Z = np.random.randn(n_paths, horizon_days)
    log_returns = drift + sigma_daily * Z
    log_prices = np.log(current_price) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_prices)

    returns_vs_cost = prices / cost_per_share - 1.0

    stopped = np.any(returns_vs_cost <= stop_loss, axis=1)
    prob_stop = float(np.mean(stopped))

    final_return = np.where(stopped, stop_loss, returns_vs_cost[:, -1])
    expected_return = float(np.mean(final_return))
    var_95 = float(np.percentile(final_return, 5))
    cvar_95 = float(np.mean(final_return[final_return <= var_95]))

    if np.any(stopped):
        stop_days = np.argmax(returns_vs_cost <= stop_loss, axis=1)
        median_stop_day = float(np.median(stop_days[stopped]))
    else:
        median_stop_day = float('inf')

    return {
        "prob_stop": prob_stop,
        "expected_return": expected_return,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "median_stop_day": median_stop_day,
    }


# =========================================================================
# SECTION 3: DCF 内在价值
# =========================================================================

def quick_dcf(ticker: str, current_price: float) -> dict:
    """简化 DCF 估值"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        fcf = info.get('freeCashflow', None)
        if fcf is None or fcf <= 0:
            ocf = info.get('operatingCashflow', 0) or info.get('totalCashFromOperatingActivities', 0)
            capex = info.get('capitalExpenditures', 0) or 0
            fcf = ocf - abs(capex)
            if fcf <= 0:
                fcf = (info.get('ebitda', 0) or 0) * 0.6

        market_cap = info.get('marketCap', 0)
        shares = market_cap / current_price if current_price > 0 else 1e9
        fcf_per_share = fcf / shares if shares > 0 else 0

        beta = info.get('beta', 1.2)
        wacc = RISK_FREE + beta * 0.05

        # 行业增长假设
        growth_map = {
            'NVDA': 0.18, 'AMD': 0.18, 'AAPL': 0.10, 'MSFT': 0.10,
            'PLTR': 0.22, 'TSLA': 0.20, 'MSTR': 0.12, 'MU': 0.14, 'GEV': 0.12,
        }
        growth = growth_map.get(ticker, 0.08)
        terminal_g = 0.03

        pv = 0
        cf = fcf_per_share
        for y in range(1, 6):
            cf *= (1 + growth)
            pv += cf / ((1 + wacc) ** y)

        terminal_cf = cf * (1 + terminal_g)
        tv = terminal_cf / (wacc - terminal_g)
        pv_tv = tv / ((1 + wacc) ** 5)

        fair_value = pv + pv_tv
        mos = (fair_value - current_price) / fair_value if fair_value > 0 else -999

        return {
            "ticker": ticker,
            "current_price": current_price,
            "fair_value": fair_value,
            "mos": mos,
            "wacc": wacc,
            "fcf_per_share": fcf_per_share,
            "growth": growth,
            "beta": beta,
            "dcf_signal": "UNDERVALUED" if mos > 0.15 else ("FAIR" if mos > 0 else "OVERvalued"),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# =========================================================================
# SECTION 4: 相关性矩阵
# =========================================================================

def correlation_analysis(held_tickers):
    """计算持仓间的相关性"""
    try:
        import yfinance as yf
        prices = {}
        for t in held_tickers:
            data = yf.download(t, period="3mo", progress=False)
            if len(data) >= 50:
                prices[t] = data['Close'].values.flatten()

        if len(prices) < 2:
            return {}

        min_len = min(len(v) for v in prices.values())
        aligned = {t: v[-min_len:] for t, v in prices.items()}
        returns = {t: np.diff(np.log(v)) for t, v in aligned.items()}

        corr_matrix = {}
        tickers = list(returns.keys())
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i < j:
                    corr = float(np.corrcoef(returns[t1], returns[t2])[0, 1])
                    corr_matrix[f"{t1}-{t2}"] = corr

        return corr_matrix
    except Exception:
        return {}


# =========================================================================
# SECTION 5: 综合决策引擎
# =========================================================================

def generate_decision(positions, mc_results, dcf_results, corr, account, scanner):
    """生成每笔持仓的具体操作指令"""

    print("\n" + "█" * 72)
    print("█  SMG 小时级量化决策报告 — Chief Quant Analyst")
    print("█  数据时点: 2026-07-09 13:00 EST")
    print("█" * 72)

    # --- PART A: 止损监控 ---
    print("\n" + "─" * 95)
    print("  PART A: 止损线监控 (-6% 硬止损)")
    print("─" * 95)
    header = (f"  {'Lot':<5} {'Ticker':<6} {'Qty':>4} {'成本价':>10} {'现价':>10} "
              f"{'盈亏%':>8} {'止损价':>10} {'距止损':>8} {'止损概率':>8} {'风险等级'}")
    print(header)
    print(f"  {'-'*90}")

    worst_position = None
    worst_distance = float('inf')

    for i, pos in enumerate(positions):
        mc = mc_results.get(pos["ticker"], {})
        prob = mc.get("prob_stop", 0)

        risk_level = "🟢 SAFE"
        if pos["distance_to_stop_pct"] < 0.05:
            risk_level = "🔴 DANGER"
        elif pos["distance_to_stop_pct"] < 0.10:
            risk_level = "🟠 WARNING"
        elif pos["distance_to_stop_pct"] < 0.15:
            risk_level = "🟡 CAUTION"

        if pos["distance_to_stop_pct"] < worst_distance:
            worst_distance = pos["distance_to_stop_pct"]
            worst_position = pos

        print(f"  {pos['lot_id']:<5} {pos['ticker']:<6} {pos['qty']:>4} "
              f"${pos['cost_per_share']:>9.2f} ${pos['current_price']:>9.3f} "
              f"{pos['pnl_pct']:>7.2%} ${pos['stop_price']:>9.2f} "
              f"{pos['distance_to_stop_pct']:>7.2%} {prob:>7.1%}  {risk_level}")

    # --- PART B: 集中度分析 ---
    print("\n" + "─" * 65)
    print("  PART B: 集中度 & 相关性分析")
    print("─" * 65)

    ticker_agg = {}
    for pos in positions:
        t = pos["ticker"]
        if t not in ticker_agg:
            ticker_agg[t] = {"mv": 0, "pnl": 0, "qty": 0, "lots": 0}
        ticker_agg[t]["mv"] += pos["market_value"]
        ticker_agg[t]["pnl"] += pos["pnl_dollar"]
        ticker_agg[t]["qty"] += pos["qty"]
        ticker_agg[t]["lots"] += 1

    print(f"  {'Ticker':<6} {'总市值':>12} {'占权益%':>8} {'总盈亏':>10} {'股数':>6} {'笔数':>5} {'超标?'}")
    print(f"  {'-'*55}")

    for ticker, agg in ticker_agg.items():
        conc = agg["mv"] / account["equity"]
        flag = "⚠️ 超20%" if conc > POSITION_CAP else "✓ 合规"
        print(f"  {ticker:<6} ${agg['mv']:>11,.2f} {conc:>7.1%} "
              f"${agg['pnl']:>9,.2f} {agg['qty']:>6} {agg['lots']:>5}  {flag}")

    if corr:
        print(f"\n  持仓间相关系数:")
        for pair, c in corr.items():
            level = "🔴 HIGH" if abs(c) > 0.7 else ("🟡 MED" if abs(c) > 0.4 else "🟢 LOW")
            print(f"    {pair}: {c:+.3f} {level}")

    # --- PART C: DCF 估值 ---
    print("\n" + "─" * 75)
    print("  PART C: DCF 内在价值")
    print("─" * 75)
    print(f"  {'Ticker':<6} {'现价':>10} {'公允价值':>10} {'安全边际':>8} "
          f"{'WACC':>7} {'FCF/Sh':>8} {'β':>6} {'判定'}")
    print(f"  {'-'*65}")

    for ticker, dcf in dcf_results.items():
        if "error" in dcf:
            print(f"  {ticker:<6} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>7} {'N/A':>8} {'N/A':>6} ⚠️ {dcf['error']}")
            continue
        print(f"  {ticker:<6} ${dcf['current_price']:>9.2f} ${dcf['fair_value']:>9.2f} "
              f"{dcf['mos']:>7.1%} {dcf['wacc']:>6.1%} ${dcf['fcf_per_share']:>7.2f} "
              f"{dcf['beta']:>5.1f} {dcf['dcf_signal']}")

    # --- PART D: V2.0 Scanner ---
    print("\n" + "─" * 72)
    print("  PART D: V2.0 Scanner 买入信号检查 (>=80分买入)")
    print("─" * 72)
    max_score = max(scanner.values())
    print(f"  最高得分: {max_score} 分 -> {'❌ 无合格标的 (所有 < 80)' if max_score < MIN_BUY_SCORE else '✅ 有买入信号'}")
    print(f"  扫描: " + ", ".join(f"{t}={s}" for t, s in sorted(scanner.items(), key=lambda x: -x[1])))

    # --- PART E: 最终决策 ---
    print("\n" + "─" * 72)
    print("  PART E: ⚡ 最终交易决策")
    print("─" * 72)

    decisions = []

    # BUY check
    if max_score < MIN_BUY_SCORE:
        decisions.append(("🛑 买入", "所有标的得分 < 80 -> 无合格买入信号，禁止开新仓"))
    else:
        best = max(scanner, key=scanner.get)
        decisions.append(("🟢 买入", f"{best} 得分 {scanner[best]} >= 80 -> 可考虑开仓"))

    # Per-position check
    for pos in positions:
        pnl_pct = pos["pnl_pct"]
        dist = pos["distance_to_stop_pct"]
        mc_prob = mc_results.get(pos["ticker"], {}).get("prob_stop", 0)

        if pnl_pct <= STOP_LOSS:
            decisions.append(("🔴 强制止损",
                f"{pos['ticker']} Lot#{pos['lot_id']}: P&L={pnl_pct:.2%} 已触及-6%死线"))
        elif dist < 0.03 and mc_prob > 0.60:
            decisions.append(("🟠 预警减仓",
                f"{pos['ticker']} Lot#{pos['lot_id']}: 距止损仅{dist:.1%}, MC止损概率{mc_prob:.0%} -> 减仓50%"))
        elif dist < 0.05:
            decisions.append(("🟡 严密监控",
                f"{pos['ticker']} Lot#{pos['lot_id']}: 距止损{dist:.1%}"))
        elif pnl_pct > 0.10:
            new_stop = pos["cost_per_share"] * 1.02
            decisions.append(("📈 上移止损",
                f"{pos['ticker']} Lot#{pos['lot_id']}: 盈利{pnl_pct:.1%} -> 新止损 ${new_stop:.2f} (保本+2%)"))
        else:
            decisions.append(("✅ 持有",
                f"{pos['ticker']} Lot#{pos['lot_id']}: 距止损{dist:.1%}, P&L={pnl_pct:.2%}, 无异常"))

    # Concentration warning
    for ticker, agg in ticker_agg.items():
        conc = agg["mv"] / account["equity"]
        if conc > POSITION_CAP:
            decisions.append(("⚠️ 集中度",
                f"{ticker}: 占权益{conc:.1%} > 20%上限 -> 禁止追加，等待稀释或减仓"))

    # Print all decisions
    for icon_title, detail in decisions:
        print(f"  {icon_title}: {detail}")

    # --- SUMMARY ---
    print("\n" + "═" * 72)
    print("  📋 执行摘要")
    print("═" * 72)

    total_mv = sum(p["market_value"] for p in positions)
    total_pnl = total_mv - sum(p["cost_total"] for p in positions)

    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  账户权益:     ${account['equity']:>12,.2f}                          ║
  ║  持仓市值:     ${total_mv:>12,.2f}  ({total_mv/account['equity']:.0%}仓位)                    ║
  ║  总盈亏:       ${total_pnl:>+12,.2f}                          ║
  ║  现金:         ${account['cash']:>12,.2f}                          ║
  ║  买入力:       ${account['buying_power']:>12,.2f}                          ║
  ╠══════════════════════════════════════════════════════════════════╣""")

    buy_count = sum(1 for d in decisions if "买入" in d[0] and "禁止" not in d[1])
    sell_count = sum(1 for d in decisions if "止损" in d[0])
    reduce_count = sum(1 for d in decisions if "减仓" in d[0])
    monitor_count = sum(1 for d in decisions if "监控" in d[0])
    trail_count = sum(1 for d in decisions if "上移" in d[0])
    hold_count = sum(1 for d in decisions if "持有" in d[0])

    print(f"  ║  决策: {buy_count}买 {sell_count}卖 {reduce_count}减仓 {monitor_count}监控 {trail_count}移止损 {hold_count}持有     ║")
    if sell_count > 0:
        print(f"  ║  ⚠️  有持仓触发止损 -> 立即执行卖出                              ║")
    elif reduce_count > 0:
        print(f"  ║  ⚠️  有持仓接近止损线 -> 建议减仓                                ║")
    else:
        print(f"  ║  ✅ 按兵不动 — 所有持仓安全，无合格买入信号                     ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")

    # Monte Carlo summary
    print(f"\n  📊 蒙特卡洛风险矩阵 (50k路径, 38日展望):")
    print(f"  {'Ticker':<6} {'E[R]':>8} {'VaR(95)':>8} {'CVaR(95)':>9} {'P(Stop)':>8} {'Stop中位日':>10}")
    print(f"  {'-'*55}")
    for ticker, mc in mc_results.items():
        stop_day_str = f"{mc['median_stop_day']:>9.0f}" if mc['median_stop_day'] != float('inf') else "       N/A"
        print(f"  {ticker:<6} {mc['expected_return']:>7.1%} {mc['var_95']:>7.1%} "
              f"{mc['cvar_95']:>8.1%} {mc['prob_stop']:>7.1%} {stop_day_str}")

    return decisions


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 72)
    print("  SMG CHIEF QUANT ANALYST — 小时级量化巡检")
    print("  免责声明: SMG 虚拟盘教育用途，非真实投资建议")
    print("=" * 72)

    # Step 1: 持仓风险计算
    print("\n[1/5] 计算持仓风险矩阵...")
    positions, total_mv, total_pnl = compute_position_risk(HOLDINGS, ACCOUNT)

    # Step 2: 拉取波动率
    print("[2/5] 拉取历史波动率参数...")
    held_tickers = list(set(p["ticker"] for p in positions))
    vol_params = {}
    for t in held_tickers:
        params = estimate_volatility(t)
        if params:
            vol_params[t] = params
            print(f"  {t}: sigma_ann={params['sigma_annual']:.1%}, mu_ann={params['mu_annual']:.1%}, Sharpe={params['sharpe']:.2f}")
        else:
            vol_params[t] = {"mu_daily": 0.0003, "sigma_daily": 0.018, "mu_annual": 0.075, "sigma_annual": 0.28, "sharpe": 0.27}
            print(f"  {t}: 使用默认参数 sigma_ann=28%, mu_ann=7.5%")

    # Step 3: 蒙特卡洛
    print("\n[3/5] 蒙特卡洛 VaR 模拟 (50k 路径/标的)...")
    mc_results = {}
    for pos in positions:
        t = pos["ticker"]
        if t not in mc_results:
            params = vol_params[t]
            mc = mc_stop_loss_risk(
                cost_per_share=pos["cost_per_share"],
                current_price=pos["current_price"],
                sigma_daily=params["sigma_daily"],
                mu_daily=params["mu_daily"],
                horizon_days=REMAINING_DAYS,
                n_paths=50000,
                stop_loss=STOP_LOSS,
            )
            mc_results[t] = mc
            print(f"  {t}: P(stop)={mc['prob_stop']:.1%}, E[R]={mc['expected_return']:.1%}, "
                  f"VaR95={mc['var_95']:.1%}, CVaR95={mc['cvar_95']:.1%}")

    # Step 4: DCF
    print("\n[4/5] DCF 内在价值估算...")
    dcf_results = {}
    for t in held_tickers:
        price = next(p["current_price"] for p in positions if p["ticker"] == t)
        print(f"  Computing DCF for {t}...")
        dcf = quick_dcf(t, price)
        dcf_results[t] = dcf
        if "error" in dcf:
            print(f"    ⚠️  {dcf['error']}")
        else:
            print(f"    Fair=${dcf['fair_value']:.2f}, MoS={dcf['mos']:.1%}, -> {dcf['dcf_signal']}")

    # Step 5: 相关性
    print("\n[5/5] 相关性分析...")
    corr = correlation_analysis(held_tickers)

    # Generate decisions
    decisions = generate_decision(positions, mc_results, dcf_results, corr, ACCOUNT, SCANNER)

    # Actionable output
    print("\n" + "=" * 72)
    print("  🎯 可执行操作指令:")
    print("=" * 72)

    has_action = False
    for icon_title, detail in decisions:
        if "止损" in icon_title:
            for pos in positions:
                if pos["pnl_pct"] <= STOP_LOSS:
                    print(f"  -> 卖出 {pos['ticker']} x {pos['qty']} 股 @ ${pos['current_price']:.2f}")
                    has_action = True
        elif "减仓" in icon_title:
            for pos in positions:
                if pos["distance_to_stop_pct"] < 0.03:
                    print(f"  -> 减仓 {pos['ticker']} Lot#{pos['lot_id']}: 卖出 {pos['qty']//2} 股")
                    has_action = True
        elif "上移" in icon_title:
            for pos in positions:
                if pos["pnl_pct"] > 0.10:
                    new_stop = pos["cost_per_share"] * 1.02
                    print(f"  -> {pos['ticker']} Lot#{pos['lot_id']}: 设置追踪止损 @ ${new_stop:.2f}")
                    has_action = True

    if not has_action:
        print(f"  -> 【核心指令】按兵不动。无触发止损、无合格买入信号。")
        print(f"  -> 下次巡检继续监控 MSTR (最接近止损线，距止损仅 {worst_position['distance_to_stop_pct']:.1%})。")

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE.")
    print("=" * 72)


if __name__ == "__main__":
    main()
