#!/usr/bin/env python3
"""
===========================================================================
SMG V3.0 激进冲刺量化验证引擎 — Chief Quant Analyst
===========================================================================
四模型融合: DCF内在价值 + 蒙特卡洛VaR + 技术指标 + 组合优化 (Kelly)
目标: 冲击 $110,000 Equity
免责声明: 此分析仅用于 SMG 虚拟盘中学生教育目的，不构成真实投资建议。
===========================================================================
"""

import numpy as np
from scipy import stats, optimize
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json, sys, warnings, time
warnings.filterwarnings('ignore')

# =========================================================================
# SECTION 0: 实时数据输入
# =========================================================================

ACCOUNT = {
    "equity": 100515.78,
    "buying_power": 87527.07,
    "cash": 37205.87,
}

HOLDINGS = [
    {"ticker": "AAPL", "date": "2026-07-08", "qty": 10,  "cost_total": 3120.10,  "current": 313.042},
    {"ticker": "AMD",  "date": "2026-07-07", "qty": 35,  "cost_total": 18001.30, "current": 548.134},
    {"ticker": "MSTR", "date": "2026-07-07", "qty": 39,  "cost_total": 3812.68,  "current": 96.239},
    {"ticker": "MU",   "date": "2026-07-06", "qty": 9,   "cost_total": 9066.00,  "current": 976.935},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 10,  "cost_total": 1928.70,  "current": 206.996},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 100, "cost_total": 19741.00, "current": 206.996},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 10,  "cost_total": 1981.80,  "current": 206.996},
    {"ticker": "NVDA", "date": "2026-07-07", "qty": 10,  "cost_total": 1926.80,  "current": 206.996},
]

# V2.0 Scanner 结果
SCANNER = {
    "AMD":  {"score": 68, "price": 547.86},
    "NVDA": {"score": 63, "price": 206.93},
    "GEV":  {"score": 63, "price": 1073.19},
    "AAPL": {"score": 60, "price": 313.01},
    "TSLA": {"score": 60, "price": 406.90},
    "MSFT": {"score": 55, "price": 384.12},
    "PLTR": {"score": 55, "price": 126.93},
}

# V3.0 参数
STOP_LOSS = -0.06
MIN_BUY_SCORE = 60          # V3.0: 从 80 降至 60
POSITION_CAP = 0.25          # V3.0: 从 20% 提至 25%
TARGET_EQUITY = 110000.0
REMAINING_DAYS = 37
TRADING_DAYS_YEAR = 252
RISK_FREE = 0.0475
MAX_LEVERAGE = 1.50          # V3.0: 允许 1.5x 杠杆

# =========================================================================
# SECTION 1: 历史波动率与市场参数获取
# =========================================================================

def fetch_market_params(ticker: str, retries=3) -> Optional[Dict]:
    """拉取近 90 天价格数据，估计年化波动率 σ 和漂移率 μ"""
    for attempt in range(retries):
        try:
            import yfinance as yf
            data = yf.download(ticker, period="3mo", progress=False)
            if len(data) < 20:
                continue
            closes = data['Close'].values.flatten()
            volumes = data['Volume'].values.flatten()
            highs = data['High'].values.flatten()
            lows = data['Low'].values.flatten()

            log_returns = np.diff(np.log(closes))
            mu_daily = float(np.mean(log_returns))
            sigma_daily = float(np.std(log_returns, ddof=1))
            mu_annual = mu_daily * TRADING_DAYS_YEAR
            sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_YEAR)
            sharpe = mu_annual / sigma_annual if sigma_annual > 0 else 0

            # 额外技术数据
            sma20 = float(np.mean(closes[-20:]))
            sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else sma20
            avg_vol_20d = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else float(np.mean(volumes))

            return {
                "mu_daily": mu_daily,
                "sigma_daily": sigma_daily,
                "mu_annual": mu_annual,
                "sigma_annual": sigma_annual,
                "sharpe": sharpe,
                "price": float(closes[-1]),
                "sma20": sma20,
                "sma50": sma50,
                "avg_vol": avg_vol_20d,
                "n_days": len(data),
                "closes": closes,
                "highs": highs,
                "lows": lows,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            continue
    return None


# =========================================================================
# SECTION 2: DCF 内在价值模型
# =========================================================================

def dcf_intrinsic_value(ticker: str, market_price: float) -> Dict:
    """多阶段 DCF 估值模型"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        # 自由现金流
        fcf = info.get('freeCashflow', None)
        if fcf is None or fcf <= 0:
            ocf = info.get('operatingCashflow', 0) or info.get('totalCashFromOperatingActivities', 0) or 0
            capex = abs(info.get('capitalExpenditures', 0) or 0)
            fcf = ocf - capex
            if fcf <= 0:
                fcf = (info.get('ebitda', 0) or 0) * 0.55

        market_cap = info.get('marketCap', None) or 1e10
        shares = market_cap / market_price if market_price > 0 else 1e9
        fcf_per_share = fcf / shares if shares > 0 else 0

        beta = info.get('beta', 1.2)
        if beta is None or beta <= 0:
            beta = 1.2
        erp = 0.055  # equity risk premium
        wacc = RISK_FREE + beta * erp
        wacc = max(0.06, min(wacc, 0.15))

        # 行业增长假设 (基于分析师共识)
        growth_map = {
            'NVDA': 0.20, 'AMD': 0.18, 'AAPL': 0.09, 'MSFT': 0.12,
            'PLTR': 0.22, 'TSLA': 0.18, 'MSTR': 0.12, 'MU': 0.15,
            'GEV': 0.14, 'VST': 0.10, 'SMR': 0.25, 'CRWV': 0.08,
        }
        g1 = growth_map.get(ticker, 0.08)
        g1 = max(0.03, min(g1, 0.35))
        terminal_g = 0.03
        growth_years = 5

        # 阶段1: 高增长
        pv = 0.0
        cf = fcf_per_share
        for yr in range(1, growth_years + 1):
            cf *= (1 + g1)
            pv += cf / ((1 + wacc) ** yr)

        # 终值
        terminal_fcf = cf * (1 + terminal_g)
        if wacc > terminal_g:
            terminal_value = terminal_fcf / (wacc - terminal_g)
        else:
            terminal_value = terminal_fcf / 0.03
        pv_terminal = terminal_value / ((1 + wacc) ** growth_years)

        fair_value = pv + pv_terminal
        mos = (fair_value - market_price) / fair_value if fair_value > 0 else 0

        if mos > 0.25:
            rec = "STRONG_UNDER"
        elif mos > 0.10:
            rec = "UNDER"
        elif mos > -0.05:
            rec = "FAIR"
        elif mos > -0.20:
            rec = "OVER"
        else:
            rec = "EXPENSIVE"

        return {
            "ticker": ticker,
            "market_price": market_price,
            "fair_value": round(fair_value, 2),
            "mos": round(mos, 4),
            "wacc": round(wacc, 4),
            "growth_s1": round(g1, 4),
            "fcf_per_share": round(fcf_per_share, 2),
            "beta": round(beta, 2),
            "dcf_signal": rec,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# =========================================================================
# SECTION 3: 蒙特卡洛模拟 (GBM + 止损)
# =========================================================================

@dataclass
class MCResult:
    ticker: str
    score: int
    entry_price: float
    n_paths: int
    prob_stopped: float
    prob_profit: float
    expected_return: float
    expected_return_cond: float
    median_return: float
    p5_return: float
    p95_return: float
    max_dd_avg: float
    sharpe_ratio: float
    avg_days_stop: float
    kelly_f: float
    ev_dollar: float
    recommendation: str


def run_monte_carlo(ticker: str, score: int, entry_price: float,
                    market_params: Dict, n_paths: int = 100_000,
                    horizon_days: int = None, stop_loss: float = STOP_LOSS) -> MCResult:
    """GBM 蒙特卡洛 + 日内止损监控"""
    if horizon_days is None:
        horizon_days = REMAINING_DAYS

    mu_d = market_params.get("mu_daily", 0.0003)
    sigma_d = market_params.get("sigma_daily", 0.018)
    if sigma_d <= 0:
        sigma_d = 0.018

    drift = mu_d - 0.5 * sigma_d ** 2

    # 向量化模拟 [n_paths, horizon_days]
    Z = np.random.randn(n_paths, horizon_days)
    log_returns = drift + sigma_d * Z
    log_prices = np.log(entry_price) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_prices)

    returns_vs_entry = prices / entry_price - 1.0

    # 止损检测
    stopped = np.any(returns_vs_entry <= stop_loss, axis=1)
    prob_stopped = float(np.mean(stopped))

    stop_days = np.argmax(returns_vs_entry <= stop_loss, axis=1)
    stop_days[~stopped] = horizon_days

    final_return = np.where(stopped, stop_loss, returns_vs_entry[:, -1])
    expected_return = float(np.mean(final_return))
    median_return = float(np.median(final_return))
    p5 = float(np.percentile(final_return, 5))
    p95 = float(np.percentile(final_return, 95))
    prob_profit = float(np.mean(final_return > 0))

    cond_return = float(np.mean(final_return[~stopped])) if np.any(~stopped) else stop_loss

    # 最大回撤
    cummax = np.maximum.accumulate(prices, axis=1)
    drawdowns = prices / cummax - 1.0
    max_dd_avg = float(np.mean(np.min(drawdowns, axis=1)))

    # 夏普
    std_ret = float(np.std(final_return))
    sharpe = expected_return / std_ret if std_ret > 0 else 0

    # 止损平均天数
    avg_days_stop = float(np.mean(stop_days[stopped])) if np.any(stopped) else float(horizon_days)

    # Kelly fraction
    win_mask = final_return > 0
    if np.any(win_mask) and np.any(~win_mask):
        avg_win = float(np.mean(final_return[win_mask]))
        avg_loss = abs(float(np.mean(final_return[~win_mask])))
        win_rate = float(np.mean(win_mask))
        if avg_loss > 0:
            kelly_f = win_rate - (1 - win_rate) / (avg_win / avg_loss)
            kelly_f = max(0.0, min(kelly_f, 0.40))
        else:
            kelly_f = 0.0
    else:
        kelly_f = 0.0 if np.all(final_return <= 0) else 0.30

    ev_dollar = expected_return

    # V3.0 建议
    if score >= MIN_BUY_SCORE and expected_return > 0.015 and prob_stopped < 0.45:
        rec = "🟢 STRONG BUY"
    elif expected_return > 0.005 and prob_stopped < 0.55:
        rec = "🟡 WEAK BUY"
    elif expected_return > -0.02:
        rec = "🟠 HOLD"
    else:
        rec = "🔴 PASS"

    return MCResult(
        ticker=ticker, score=score, entry_price=entry_price,
        n_paths=n_paths, prob_stopped=prob_stopped, prob_profit=prob_profit,
        expected_return=expected_return, expected_return_cond=cond_return,
        median_return=median_return, p5_return=p5, p95_return=p95,
        max_dd_avg=max_dd_avg, sharpe_ratio=sharpe, avg_days_stop=avg_days_stop,
        kelly_f=kelly_f, ev_dollar=ev_dollar, recommendation=rec,
    )


# =========================================================================
# SECTION 4: 技术指标计算
# =========================================================================

def compute_technical_indicators(closes, highs, lows, volumes) -> Dict:
    """计算完整技术指标套件"""
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(closes)

    result = {}

    # --- 均线系统 ---
    ema8 = float(pd_ema(closes, 8)[-1])
    ema21 = float(pd_ema(closes, 21)[-1])
    ema50 = float(pd_ema(closes, 50)[-1]) if n >= 50 else ema21
    sma20 = float(np.mean(closes[-20:]))
    sma50 = float(np.mean(closes[-50:])) if n >= 50 else sma20
    close = closes[-1]

    result["ema8"] = round(ema8, 2)
    result["ema21"] = round(ema21, 2)
    result["ema50"] = round(ema50, 2)
    result["sma20"] = round(sma20, 2)
    result["sma50"] = round(sma50, 2)
    result["trend"] = "BULL" if (ema8 > ema21 and ema21 > ema50 and close > ema8) else \
                      "BEAR" if (ema8 < ema21 and ema21 < ema50) else "NEUTRAL"

    # --- RSI 14 ---
    result["rsi14"] = round(compute_rsi(closes, 14), 1)

    # --- MACD ---
    ema12 = pd_ema(closes, 12)
    ema26 = pd_ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = pd_ema(macd_line, 9)
    histogram = macd_line - signal_line
    result["macd"] = round(float(macd_line[-1]), 3)
    result["macd_signal"] = round(float(signal_line[-1]), 3)
    result["macd_hist"] = round(float(histogram[-1]), 3)
    result["macd_cross"] = "BULL" if histogram[-1] > 0 else "BEAR"

    # --- 布林带 ---
    result["bb_upper"] = round(sma20 + 2 * np.std(closes[-20:], ddof=1), 2)
    result["bb_lower"] = round(sma20 - 2 * np.std(closes[-20:], ddof=1), 2)
    result["bb_mid"] = round(sma20, 2)
    bb_pos = (close - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"]) if result["bb_upper"] != result["bb_lower"] else 0.5
    result["bb_position"] = round(bb_pos, 2)

    # --- 波动率 ---
    log_ret = np.diff(np.log(closes))
    result["hist_vol_20d"] = round(float(np.std(log_ret[-20:], ddof=1) * np.sqrt(252)), 4)
    result["hist_vol_10d"] = round(float(np.std(log_ret[-10:], ddof=1) * np.sqrt(252)), 4)

    # --- ATR ---
    result["atr14"] = round(compute_atr(highs, lows, closes, 14), 2)

    # --- 成交量 ---
    avg_vol = np.mean(volumes[-20:])
    last_vol = volumes[-1]
    result["vol_ratio"] = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    # --- 动量 ---
    result["return_5d"] = round(float(closes[-1] / closes[-6] - 1), 4) if n >= 6 else 0
    result["return_10d"] = round(float(closes[-1] / closes[-11] - 1), 4) if n >= 11 else 0
    result["return_20d"] = round(float(closes[-1] / closes[-21] - 1), 4) if n >= 21 else 0

    # --- 支撑/阻力 ---
    result["resistance"] = round(float(np.max(highs[-20:])), 2)
    result["support"] = round(float(np.min(lows[-20:])), 2)

    # --- 综合技术评分 (0-100) ---
    tech_score = 0
    if result["trend"] == "BULL": tech_score += 30
    elif result["trend"] == "NEUTRAL": tech_score += 15
    if result["macd_cross"] == "BULL": tech_score += 20
    if 40 <= result["rsi14"] <= 70: tech_score += 15
    elif 30 <= result["rsi14"] < 40: tech_score += 10  # oversold bounce
    if result["vol_ratio"] > 1.2: tech_score += 10
    if result["return_5d"] > 0: tech_score += 10
    if result["return_10d"] > 0: tech_score += 10
    if result["return_20d"] > 0: tech_score += 5
    result["tech_score"] = tech_score

    return result


def pd_ema(data, span):
    """Pandas-style EMA"""
    alpha = 2 / (span + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def compute_rsi(closes, period=14):
    """RSI"""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    return rsi[-1]


def compute_atr(highs, lows, closes, period=14):
    """Average True Range"""
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
    atr = np.zeros(len(closes))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr[-1]


# =========================================================================
# SECTION 5: 相关性矩阵 & 组合优化
# =========================================================================

def correlation_analysis(tickers: List[str]) -> Dict:
    """计算标的间相关性矩阵"""
    try:
        import yfinance as yf
        prices = {}
        for t in tickers:
            data = yf.download(t, period="3mo", progress=False)
            if len(data) >= 30:
                prices[t] = data['Close'].values.flatten()
        if len(prices) < 2:
            return {"clusters": [], "avg_corr": 0}

        # 对齐长度
        min_len = min(len(v) for v in prices.values())
        aligned = {}
        for t, p in prices.items():
            aligned[t] = p[-min_len:]

        rets = {}
        for t, p in aligned.items():
            rets[t] = np.diff(np.log(p))

        n = len(rets)
        corr_mat = np.zeros((n, n))
        ticker_list = list(rets.keys())
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_mat[i, j] = 1.0
                else:
                    r1, r2 = rets[ticker_list[i]], rets[ticker_list[j]]
                    min_l = min(len(r1), len(r2))
                    corr_mat[i, j] = np.corrcoef(r1[:min_l], r2[:min_l])[0, 1]

        # 找高相关集群 (>0.70)
        clusters = []
        visited = set()
        for i in range(n):
            if i in visited:
                continue
            cluster = [ticker_list[i]]
            for j in range(i + 1, n):
                if corr_mat[i, j] > 0.70:
                    cluster.append(ticker_list[j])
                    visited.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)
            visited.add(i)

        avg_corr = float(np.mean(corr_mat[np.triu_indices(n, k=1)]))

        return {
            "tickers": ticker_list,
            "corr_matrix": corr_mat.tolist(),
            "clusters": clusters,
            "avg_corr": round(avg_corr, 4),
        }
    except Exception:
        return {"clusters": [], "avg_corr": 0, "error": "fetch failed"}


def portfolio_kelly_allocation(mc_results: List[MCResult], account: Dict) -> Dict:
    """基于 Kelly Criterion 的最优组合配置"""
    # 过滤有正期望的标的
    valid = [r for r in mc_results if r.kelly_f > 0 and r.score >= MIN_BUY_SCORE]
    if not valid:
        return {"allocations": [], "total_invest": 0}

    # 按 Kelly f 排序
    valid.sort(key=lambda x: x.kelly_f, reverse=True)

    # 资金分配: 先用现金, 再用部分融资
    available = account["cash"] + (account["buying_power"] - account["cash"]) * 0.5
    # 限制: 最多用 $50k 新增资金
    max_new_invest = min(available, 50000.0)

    total_kelly = sum(r.kelly_f for r in valid)
    allocations = []

    remaining = max_new_invest
    for r in valid:
        weight = r.kelly_f / total_kelly if total_kelly > 0 else 1.0 / len(valid)
        target_amount = min(max_new_invest * weight, account["equity"] * POSITION_CAP)
        target_amount = min(target_amount, remaining)
        shares = int(target_amount / r.entry_price)
        actual_amount = shares * r.entry_price
        if shares > 0 and actual_amount > 500:
            allocations.append({
                "ticker": r.ticker,
                "score": r.score,
                "kelly_f": round(r.kelly_f, 3),
                "shares": shares,
                "amount": round(actual_amount, 2),
                "entry_price": r.entry_price,
            })
            remaining -= actual_amount

    return {
        "allocations": allocations,
        "total_invest": round(sum(a["amount"] for a in allocations), 2),
        "max_new_invest": round(max_new_invest, 2),
    }


# =========================================================================
# SECTION 6: 持仓诊断与操作指令
# =========================================================================

def diagnose_holdings(holdings, account, mc_results, tech_data, dcf_results) -> List[Dict]:
    """逐笔持仓诊断"""
    actions = []

    # 聚合持仓
    ticker_agg = {}
    for h in holdings:
        t = h["ticker"]
        if t not in ticker_agg:
            ticker_agg[t] = {"qty": 0, "cost_total": 0, "mv": 0}
        ticker_agg[t]["qty"] += h["qty"]
        ticker_agg[t]["cost_total"] += h["cost_total"]
        ticker_agg[t]["mv"] += h["qty"] * h["current"]

    for ticker, agg in ticker_agg.items():
        cost_per_share = agg["cost_total"] / agg["qty"]
        current_price = agg["mv"] / agg["qty"]
        pnl_pct = (current_price / cost_per_share) - 1.0
        mv = agg["mv"]
        concentration = mv / account["equity"]

        stop_price = cost_per_share * (1 + STOP_LOSS)
        dist_to_stop = (current_price / stop_price) - 1.0

        # MC 数据
        mc = next((r for r in mc_results if r.ticker == ticker), None)

        action = {
            "ticker": ticker,
            "qty": agg["qty"],
            "cost_per_share": round(cost_per_share, 2),
            "current_price": round(current_price, 2),
            "mv": round(mv, 2),
            "pnl_pct": round(pnl_pct, 4),
            "pnl_dollar": round(mv - agg["cost_total"], 2),
            "concentration": round(concentration, 3),
            "stop_price": round(stop_price, 2),
            "dist_to_stop_pct": round(dist_to_stop, 4),
        }

        # 决策逻辑
        if pnl_pct <= STOP_LOSS:
            action["decision"] = "🔴 SELL"
            action["reason"] = f"触及-6%死线 (P&L={pnl_pct:.2%})"
        elif dist_to_stop < 0.03:
            action["decision"] = "🟠 REDUCE 50%"
            action["reason"] = f"距止损仅{dist_to_stop:.1%}，MC止损概率{mc.prob_stopped:.0%}"
            action["sell_qty"] = agg["qty"] // 2
        elif pnl_pct > 0.08:
            new_stop = cost_per_share * 1.03
            action["decision"] = "📈 TRAIL STOP"
            action["reason"] = f"盈利{pnl_pct:.1%}，上移止损至${new_stop:.2f} (+3%)"
            action["new_stop"] = round(new_stop, 2)
        elif concentration > POSITION_CAP:
            action["decision"] = "⚠️ NO ADD"
            action["reason"] = f"集中度{concentration:.1%} > {POSITION_CAP:.0%}上限"
        else:
            action["decision"] = "✅ HOLD"
            action["reason"] = f"距止损{dist_to_stop:.1%}，无异常信号"

        # 附加 DCF 信息
        dcf = dcf_results.get(ticker, {})
        if dcf and "mos" in dcf:
            action["dcf_mos"] = dcf["mos"]
            action["dcf_fair"] = dcf.get("fair_value", 0)
        else:
            action["dcf_mos"] = None

        actions.append(action)

    return actions


# =========================================================================
# SECTION 7: 综合输出
# =========================================================================

def print_header(title: str):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def main():
    print("╔" + "═" * 70 + "╗")
    print("║  SMG V3.0 CHIEF QUANT ANALYST — 激进冲刺模式验证引擎         ║")
    print("║  目标: $100,516 → $110,000 | 四模型融合: DCF + MC + Tech + Portfolio  ║")
    print("║  免责声明: SMG 虚拟盘教育用途，不构成真实投资建议             ║")
    print("╚" + "═" * 70 + "╝")

    # ===== STEP 1: 市场数据 =====
    print_header("STEP 1/6: 拉取市场参数 & 技术指标")
    all_tickers = list(SCANNER.keys())
    held_tickers = list(set(h["ticker"] for h in HOLDINGS))
    all_tickers = list(dict.fromkeys(all_tickers + held_tickers))  # unique, ordered

    market_data = {}
    tech_indicators = {}
    for ticker in all_tickers:
        print(f"  Fetching {ticker}...", end=" ")
        params = fetch_market_params(ticker)
        if params:
            market_data[ticker] = params
            # 计算技术指标
            tech = compute_technical_indicators(
                params["closes"], params["highs"], params["lows"],
                np.ones_like(params["closes"]) * params.get("avg_vol", 1e6)
            )
            tech_indicators[ticker] = tech
            print(f"σ={params['sigma_annual']:.1%}, μ={params['mu_annual']:.1%}, "
                  f"Sharpe={params['sharpe']:.2f}, Tech={tech['tech_score']}")
        else:
            print("FAILED — using defaults")
            market_data[ticker] = {
                "mu_daily": 0.0004, "sigma_daily": 0.020,
                "mu_annual": 0.10, "sigma_annual": 0.32,
                "sharpe": 0.31, "price": SCANNER.get(ticker, {}).get("price", 100),
                "sma20": 0, "sma50": 0, "avg_vol": 1e6,
                "n_days": 60, "closes": np.ones(60) * 100,
                "highs": np.ones(60) * 100, "lows": np.ones(60) * 100,
            }
            tech_indicators[ticker] = {"tech_score": 50, "trend": "NEUTRAL",
                                        "rsi14": 50, "macd_cross": "NEUTRAL"}

    # ===== STEP 2: DCF =====
    print_header("STEP 2/6: DCF 内在价值估值 (多阶段模型)")
    dcf_results = {}
    for ticker, info in SCANNER.items():
        price = info["price"]
        print(f"  Computing DCF for {ticker} (P=${price:.2f})...")
        dcf = dcf_intrinsic_value(ticker, price)
        dcf_results[ticker] = dcf
        if "error" in dcf:
            print(f"    ⚠️  {dcf['error']}")
        else:
            print(f"    Fair=${dcf['fair_value']:.2f}, MoS={dcf['mos']:.1%}, "
                  f"WACC={dcf['wacc']:.1%}, β={dcf['beta']:.2f}, → {dcf['dcf_signal']}")

    # 也检查持仓中不在 scanner 的
    for t in held_tickers:
        if t not in dcf_results:
            price = next(h["current"] for h in HOLDINGS if h["ticker"] == t)
            dcf = dcf_intrinsic_value(t, price)
            dcf_results[t] = dcf
            if "error" not in dcf:
                print(f"  {t} (holding): Fair=${dcf['fair_value']:.2f}, MoS={dcf['mos']:.1%}")

    # ===== STEP 3: 蒙特卡洛 =====
    print_header("STEP 3/6: 蒙特卡洛模拟 (100k路径, 37日展望, -6%止损)")
    mc_results = []
    for ticker, info in SCANNER.items():
        params = market_data[ticker]
        result = run_monte_carlo(
            ticker=ticker, score=info["score"],
            entry_price=info["price"], market_params=params,
            n_paths=100_000, horizon_days=REMAINING_DAYS, stop_loss=STOP_LOSS,
        )
        mc_results.append(result)
        print(f"  {ticker}: E[R]={result.expected_return:+.1%}, "
              f"P(Stop)={result.prob_stopped:.1%}, "
              f"Sharpe={result.sharpe_ratio:.2f}, "
              f"Kelly f={result.kelly_f:.1%}, "
              f"→ {result.recommendation}")

    # ===== STEP 4: 相关性 =====
    print_header("STEP 4/6: 相关性矩阵 & 集中度风险")
    corr = correlation_analysis(all_tickers)
    print(f"  平均相关系数: {corr.get('avg_corr', 'N/A')}")
    if corr.get("clusters"):
        for c in corr["clusters"]:
            print(f"  ⚠️  高相关集群 (r>0.70): {' ↔ '.join(c)}")
    else:
        print(f"  ✅ 无危险高相关集群")

    # ===== STEP 5: 持仓诊断 =====
    print_header("STEP 5/6: 现有持仓逐笔诊断")
    positions = diagnose_holdings(HOLDINGS, ACCOUNT, mc_results, tech_indicators, dcf_results)

    print(f"\n  {'Ticker':<6} {'Qty':>5} {'Cost/Sh':>9} {'Price':>8} {'MV':>10} {'P&L%':>8} "
          f"{'DistStop':>9} {'Conc':>6} {'DCF MoS':>8} {'Decision':>16}")
    print(f"  {'-'*100}")
    for p in positions:
        mos_str = f"{p['dcf_mos']:.1%}" if p['dcf_mos'] is not None else "N/A"
        print(f"  {p['ticker']:<6} {p['qty']:>5} ${p['cost_per_share']:>8.2f} ${p['current_price']:>7.2f} "
              f"${p['mv']:>9.2f} {p['pnl_pct']:>7.2%} {p['dist_to_stop_pct']:>8.2%} "
              f"{p['concentration']:>5.1%} {mos_str:>8} {p['decision']}")

    # 逐笔理由
    for p in positions:
        if p["decision"] != "✅ HOLD":
            print(f"    → {p['ticker']}: {p['reason']}")

    # ===== STEP 6: Kelly 组合优化 & 买入计划 =====
    print_header("STEP 6/6: V3.0 Kelly 组合优化 & 买入决策")

    alloc = portfolio_kelly_allocation(mc_results, ACCOUNT)

    if alloc["allocations"]:
        print(f"\n  💰 可用新增资金: ${alloc['max_new_invest']:,.2f}")
        print(f"  📊 V3.0 买入计划 (按Kelly仓位):")
        print(f"\n  {'Ticker':<6} {'Score':>5} {'Kelly':>7} {'Shares':>7} {'Amount':>10} {'Price':>8}")
        print(f"  {'-'*50}")
        for a in alloc["allocations"]:
            print(f"  {a['ticker']:<6} {a['score']:>5} {a['kelly_f']:>6.1%} {a['shares']:>7} "
                  f"${a['amount']:>9,.2f} ${a['entry_price']:>7.2f}")
        print(f"  {'-'*50}")
        print(f"  合计买入: ${alloc['total_invest']:,.2f}")
    else:
        print(f"\n  ⚠️  Kelly 模型未产生任何正期望买入信号 (score≥{MIN_BUY_SCORE})")

    # =========================================================================
    # FINAL: 三模型融合矩阵
    # =========================================================================
    print_header("FINAL: 四模型融合决策矩阵")
    print(f"\n  {'Ticker':<6} {'V2.0':>5} {'MC_EV':>8} {'DCF_MoS':>8} {'Tech':>5} "
          f"{'P(Stop)':>8} {'Kelly':>7} {'Sharpe':>7} {'FUSION':>18}")
    print(f"  {'-'*92}")

    mc_map = {r.ticker: r for r in mc_results}
    for ticker, info in SCANNER.items():
        mc = mc_map[ticker]
        dcf = dcf_results.get(ticker, {})
        tech = tech_indicators.get(ticker, {})

        score = info["score"]
        ev = mc.expected_return
        mos = dcf.get("mos", None)
        tech_s = tech.get("tech_score", 50)
        p_stop = mc.prob_stopped
        kelly = mc.kelly_f
        sharpe = mc.sharpe_ratio

        # 四模型投票
        votes = 0
        reasons = []
        if score >= MIN_BUY_SCORE:
            votes += 1
            reasons.append("V2.0≥60")
        if ev > 0.015 and p_stop < 0.50:
            votes += 1
            reasons.append("MC+EV")
        if mos is not None and mos > 0.05:
            votes += 1
            reasons.append("DCF")
        if tech_s >= 60:
            votes += 1
            reasons.append(f"Tech{tech_s}")

        if votes >= 3:
            fusion = "🟢 STRONG BUY"
        elif votes >= 2:
            fusion = "🟡 WEAK BUY"
        elif votes >= 1:
            fusion = "🟠 MONITOR"
        else:
            fusion = "🔴 PASS"

        mos_str = f"{mos:.1%}" if mos is not None else "N/A"
        print(f"  {ticker:<6} {score:>5} {ev:>7.1%} {mos_str:>8} {tech_s:>5} "
              f"{p_stop:>7.1%} {kelly:>6.1%} {sharpe:>7.2f} {fusion} ({', '.join(reasons)})")

    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    print_header("🎯 EXECUTIVE SUMMARY — 可执行操作指令")

    # 聚合持仓
    total_mv = sum(h["qty"] * h["current"] for h in HOLDINGS)
    total_cost = sum(h["cost_total"] for h in HOLDINGS)
    total_pnl = total_mv - total_cost

    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  账户权益:     ${ACCOUNT['equity']:>12,.2f}                          ║
  ║  目标权益:     ${TARGET_EQUITY:>12,.2f}  (差距 ${TARGET_EQUITY - ACCOUNT['equity']:,.2f})             ║
  ║  持仓市值:     ${total_mv:>12,.2f}  ({total_mv/ACCOUNT['equity']:.0%}仓位)                    ║
  ║  总盈亏:       ${total_pnl:>+12,.2f}                          ║
  ║  现金:         ${ACCOUNT['cash']:>12,.2f}                          ║
  ║  买入力:       ${ACCOUNT['buying_power']:>12,.2f}                          ║
  ╚══════════════════════════════════════════════════════════════════╝""")

    # --- SELL 指令 ---
    sells = [p for p in positions if p["decision"] in ("🔴 SELL", "🟠 REDUCE 50%")]
    if sells:
        print(f"\n  🔴 卖出指令:")
        for p in sells:
            sell_qty = p.get("sell_qty", p["qty"])
            print(f"     SELL {p['ticker']} x {sell_qty} 股 @ ~${p['current_price']:.2f}  →  {p['reason']}")

    # --- BUY 指令 ---
    if alloc["allocations"]:
        print(f"\n  🟢 V3.0 买入指令 (启动 ${{alloc['total_invest']:,.2f}}):")
        for a in alloc["allocations"]:
            print(f"     BUY {a['ticker']} x {a['shares']} 股 @ ~${a['entry_price']:.2f} "
                  f"≈ ${a['amount']:,.2f}  (Kelly f={a['kelly_f']:.1%})")

        # 验证集中度
        print(f"\n  📊 买入后集中度验证:")
        for a in alloc["allocations"]:
            existing_mv = sum(h["qty"] * h["current"] for h in HOLDINGS if h["ticker"] == a["ticker"])
            post_mv = existing_mv + a["amount"]
            post_conc = post_mv / (ACCOUNT["equity"] + alloc["total_invest"])
            flag = "⚠️" if post_conc > POSITION_CAP else "✅"
            print(f"     {a['ticker']}: 已有${existing_mv:,.2f} + 新增${a['amount']:,.2f} "
                  f"= ${post_mv:,.2f} ({post_conc:.1%}) {flag}")

    # --- 风险警告 ---
    print(f"\n  🛡️  风险警示:")
    # 检查 MSTR (已知接近止损线)
    mstr_pos = next((p for p in positions if p["ticker"] == "MSTR"), None)
    if mstr_pos:
        if mstr_pos["pnl_pct"] <= -0.04:
            print(f"     ⚠️  MSTR: P&L={mstr_pos['pnl_pct']:.2%}，极度接近-6%死线！若跌破${mstr_pos['stop_price']:.2f}立即斩仓！")
        else:
            print(f"     👀  MSTR: P&L={mstr_pos['pnl_pct']:.2%}，距止损{mstr_pos['dist_to_stop_pct']:.1%}，需持续监控")

    # MU 亏损
    mu_pos = next((p for p in positions if p["ticker"] == "MU"), None)
    if mu_pos and mu_pos["pnl_pct"] < -0.02:
        print(f"     👀  MU: P&L={mu_pos['pnl_pct']:.2%}，距止损{mu_pos['dist_to_stop_pct']:.1%}")

    # 高相关警告
    if corr.get("clusters"):
        for c in corr["clusters"]:
            if len(c) >= 2:
                print(f"     ⚠️  高相关集群 ({', '.join(c)}): 下行共振风险高，考虑降低同群仓位")

    # 杠杆警告
    target_exposure = total_mv + alloc["total_invest"]
    leverage = target_exposure / ACCOUNT["equity"]
    if leverage > 1.0:
        print(f"     ⚠️  计划杠杆: {leverage:.2f}x (目标敞口${target_exposure:,.0f} vs 权益${ACCOUNT['equity']:,.0f})")

    # =========================================================================
    # 目标达成概率
    # =========================================================================
    print_header("🎲 目标达成概率: 冲击 $110,000")

    # 简单计算: 需要 ~9.4% 收益, 37天
    needed_return = TARGET_EQUITY / ACCOUNT["equity"] - 1
    print(f"  需要收益: +{needed_return:.1%} ({TARGET_EQUITY - ACCOUNT['equity']:,.2f})")
    print(f"  剩余交易日: {REMAINING_DAYS} 天")

    # 用组合预期收益估算
    if alloc["allocations"]:
        total_ev = 0.0
        for a in alloc["allocations"]:
            mc = mc_map[a["ticker"]]
            total_ev += mc.expected_return * a["amount"]
        # 加上现有持仓
        for p in positions:
            t = p["ticker"]
            if t in mc_map:
                total_ev += mc_map[t].expected_return * p["mv"]

        portfolio_ev = total_ev / ACCOUNT["equity"]
        print(f"  组合预期收益 (37天): +{portfolio_ev:.1%}")
        if portfolio_ev >= needed_return:
            print(f"  ✅ 预期收益覆盖目标！概率较高")
        else:
            shortfall = needed_return - portfolio_ev
            print(f"  ⚠️  预期收益不足，缺口 {shortfall:.1%}，需超常表现或追加风险敞口")
    else:
        print(f"  ⚠️  当前无买入信号，仅靠现有持仓难以达成目标")

    print(f"\n{'═' * 72}")
    print(f"  ANALYSIS COMPLETE. V3.0 Quant Validation Engine 签名完毕。")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
