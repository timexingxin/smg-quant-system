"""
SMG Quantitative Trading System - Point-in-Time Backtesting Engine (v1.0)
完整历史回测框架：
1. 严格 Point-in-Time 决策（无未来函数/Lookahead Bias）
2. 标普500基准 (SPY) 同期对比与 Alpha/Beta 测算
3. 核心量化指标体系：Total Return, CAGR, Sharpe Ratio, Max Drawdown, Calmar Ratio, Win Rate, Profit Factor
4. SMG 规则深度集成：20% 单票持仓上限、10股最小订单、-6% 破位硬止损、5% 动态追踪止损、5 bps 滑点
5. 终端 ASCII 净值曲线图与结构化回测报告导出
"""
import math
import json
import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    from .config import (
        POSITION_CAP, MIN_ORDER_SHARES, STOP_LOSS, TRAILING_STOP,
        RISK_FREE_RATE, MIN_BUY_SCORE, TRADING_DAYS_YEAR
    )
    from .scoring import score_long, qualifies_long
except ImportError:
    from smg_strategy.config import (
        POSITION_CAP, MIN_ORDER_SHARES, STOP_LOSS, TRAILING_STOP,
        RISK_FREE_RATE, MIN_BUY_SCORE, TRADING_DAYS_YEAR
    )
    from smg_strategy.scoring import score_long, qualifies_long


@dataclass
class BacktestConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100_000.0
    benchmark: str = "SPY"
    slippage_bps: float = 5.0              # 5 bps 滑点 (0.0005)
    position_cap: float = POSITION_CAP     # 20% 单标的持仓上限
    min_shares: int = MIN_ORDER_SHARES     # 最少 10 股
    stop_loss: float = STOP_LOSS           # -0.06
    trailing_stop: float = TRAILING_STOP   # 0.05
    min_buy_score: int = MIN_BUY_SCORE     # 72
    risk_free_rate: float = RISK_FREE_RATE # 0.0475


@dataclass
class TradeRecord:
    ticker: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    config: Dict[str, Any]
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    benchmark_total_return: float
    benchmark_cagr: float
    excess_return: float
    alpha: float
    beta: float
    daily_history: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render_ascii_chart(self, width: int = 60, height: int = 15) -> str:
        """渲染高保真 ASCII 净值对比走势图"""
        if not self.daily_history or len(self.daily_history) < 2:
            return "No enough daily history for chart rendering."

        dates = [d["date"] for d in self.daily_history]
        strat_values = [d["equity"] for d in self.daily_history]
        bench_values = [d["benchmark_equity"] for d in self.daily_history]

        min_val = min(min(strat_values), min(bench_values))
        max_val = max(max(strat_values), max(bench_values))
        if max_val == min_val:
            max_val += 1.0

        n = len(dates)
        lines = []
        lines.append(f"  📈 策略净值走势图 (★ 策略 / · 标普500基准)")
        lines.append(f"  {'-' * (width + 12)}")

        for h in range(height, -1, -1):
            val_at_h = min_val + (max_val - min_val) * (h / height)
            val_str = f"${val_at_h:,.0f}".rjust(9)
            row = [" "] * width

            for col in range(width):
                idx = int(col * (n - 1) / (width - 1))
                s_v = strat_values[idx]
                b_v = bench_values[idx]

                s_row = int(round((s_v - min_val) / (max_val - min_val) * height))
                b_row = int(round((b_v - min_val) / (max_val - min_val) * height))

                if b_row == h:
                    row[col] = "·"
                if s_row == h:
                    row[col] = "★"

            lines.append(f"{val_str} |{''.join(row)}")

        lines.append(f"          +{'-' * width}")
        start_label = dates[0]
        end_label = dates[-1]
        padding = width - len(start_label) - len(end_label)
        if padding > 0:
            lines.append(f"           {start_label}{' ' * padding}{end_label}")
        else:
            lines.append(f"           {start_label} ... {end_label}")

        return "\n".join(lines)

    def print_summary(self) -> str:
        chart = self.render_ascii_chart()
        summary = f"""
================================================================================
🎯 SMG 量化投资策略历史回测报告 (Point-in-Time 严谨复盘)
================================================================================
回测标的池:    {', '.join(self.config.get('tickers', []))}
回测基准:      {self.config.get('benchmark', 'SPY')} (S&P 500 ETF)
测试区间:      {self.config.get('start_date')} -> {self.config.get('end_date')}
初始本金:      ${self.initial_capital:,.2f}
最终账户权益:  ${self.final_equity:,.2f}

--------------------------------------------------------------------------------
📊 核心量化绩效指标 (Performance Metrics)
--------------------------------------------------------------------------------
相对基准超额收益 (Excess Return vs SPY): {self.excess_return * 100:+.2f}%  (策略 {self.total_return * 100:+.2f}% vs 基准 {self.benchmark_total_return * 100:+.2f}%)
策略累计收益率 (Total Return):           {self.total_return * 100:+.2f}%
基准累计收益率 (Benchmark Return):         {self.benchmark_total_return * 100:+.2f}%
年化复合收益率 (CAGR):                   {self.cagr * 100:+.2f}%  (基准 SPY: {self.benchmark_cagr * 100:+.2f}%)
年化波动率 (Annualized Volatility):      {self.annualized_volatility * 100:.2f}%
夏普比率 (Sharpe Ratio, Rf=4.75%):       {self.sharpe_ratio:.2f}
最大回撤 (Maximum Drawdown):             {self.max_drawdown * 100:.2f}%
最长回撤周期 (Max DD Duration):          {self.max_drawdown_duration_days} 交易日
卡玛比率 (Calmar Ratio):                 {self.calmar_ratio:.2f}
阿尔法 (Jensen's Alpha):                 {self.alpha * 100:+.2f}%
贝塔 (Beta):                             {self.beta:.2f}

--------------------------------------------------------------------------------
⚔️ 交易执行与胜率统计 (Trade Execution Stats)
--------------------------------------------------------------------------------
总完成交易笔数: {self.total_trades} 笔
交易胜率 (Win Rate):        {self.win_rate * 100:.1f}%
盈亏比 (Profit Factor):     {self.profit_factor:.2f}

{chart}
================================================================================
"""
        return summary


class BacktestEngine:
    """
    量化回测执行引擎
    支持传入外部 price_data DataFrame 字典以用于单元测试和离线仿真，
    也支持通过 yfinance 自动拉取实盘历史数据。
    """

    def __init__(self, config: BacktestConfig, price_data: Optional[Dict[str, pd.DataFrame]] = None):
        self.config = config
        self.price_data: Dict[str, pd.DataFrame] = price_data or {}

    def fetch_data(self):
        """如果未显式提供 price_data，则通过 yfinance 抓取相关标的与基准行情"""
        import yfinance as yf
        symbols = list(set(self.config.tickers + [self.config.benchmark]))
        for sym in symbols:
            if sym in self.price_data:
                continue
            try:
                df = yf.download(
                    sym,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    progress=False,
                    auto_adjust=True
                )
                if df is not None and not df.empty:
                    # 扁平化 MultiIndex 列名 (若 yfinance 返回多层索引)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    self.price_data[sym] = df.sort_index()
            except Exception as e:
                print(f"⚠️ 下载标的 {sym} 历史行情失败: {e}")

    def _compute_point_in_time_score(self, ticker: str, history_df: pd.DataFrame) -> float:
        """
        根据截止到当前时刻 t 的历史切片（严格防未来函数）计算选股评分 (0-100)
        基于生产环境 scoring.py (score_long) 的统一多因子框架：
        - 30% 相对强弱 (20日收益率)
        - 20% 趋势质量 (EMA 短长均线形态)
        - 15% 成交量确认 (量比)
        - 20% 催化剂质量/技术形态 (RSI 动量指标代理)
        - 10% 盈利修正/动量加速度 (短期动量 vs 长期动量)
        - 05% 流动性指标 (日均成交额)
        - 风险惩罚 (高波动惩罚)
        """
        if len(history_df) < 20:
            return 50.0

        closes = history_df["Close"].values
        volumes = history_df["Volume"].values if "Volume" in history_df else np.ones(len(closes))
        c_now = closes[-1]

        # 1. 相对强弱 (20日收益率)
        c_20 = closes[-20]
        ret_20 = (c_now - c_20) / c_20 if c_20 > 0 else 0.0
        relative_strength = min(100.0, max(0.0, 50.0 + ret_20 * 200.0))

        # 2. 趋势质量 (5日 / 20日 EMA 关系及收盘价相对均线位置)
        ema_5 = closes[-5:].mean()
        ema_20 = closes[-20:].mean()
        trend_quality = 80.0 if (ema_5 > ema_20 and c_now > ema_5) else (60.0 if ema_5 > ema_20 else 30.0)

        # 3. 成交量确认 (3日量比)
        v_3 = volumes[-3:].mean() if len(volumes) >= 3 else 1.0
        v_20 = volumes[-20:].mean() if len(volumes) >= 20 else 1.0
        vol_ratio = v_3 / v_20 if v_20 > 0 else 1.0
        volume_confirmation = min(100.0, max(0.0, vol_ratio * 50.0))

        # 4. 催化剂/技术形态 (14日 RSI 动量)
        if len(closes) >= 15:
            diffs = np.diff(closes[-15:])
            gains = np.maximum(diffs, 0).mean()
            losses = np.maximum(-diffs, 0).mean()
            rs = gains / losses if losses > 0 else 1.0
            rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0
        catalyst_quality = rsi

        # 5. 盈利/动量加速度 (5日短期动量 vs 20日动量)
        ret_5 = (c_now - closes[-5]) / closes[-5] if len(closes) >= 5 and closes[-5] > 0 else 0.0
        earnings_revision = min(100.0, max(0.0, 50.0 + (ret_5 - ret_20 / 4.0) * 300.0))

        # 6. 流动性指标 (日均成交额)
        dollar_vol = float(c_now * v_3)
        liquidity = min(100.0, max(20.0, math.log10(max(1.0, dollar_vol)) * 12.0))

        # 7. 风险惩罚 (20日已实现波动率过高时惩罚)
        log_rets = np.diff(np.log(closes[-20:]))
        realized_vol = np.std(log_rets) * math.sqrt(252) if len(log_rets) > 1 else 0.2
        risk_penalty = max(0.0, (realized_vol - 0.40) * 50.0) if realized_vol > 0.40 else 0.0

        composite_score = score_long(
            relative_strength=relative_strength,
            trend_quality=trend_quality,
            volume_confirmation=volume_confirmation,
            catalyst_quality=catalyst_quality,
            earnings_revision=earnings_revision,
            liquidity=liquidity,
            risk_penalty=risk_penalty,
        )
        return float(composite_score)

    def run(self) -> BacktestResult:
        if not self.price_data:
            self.fetch_data()

        # 提取共同的对齐交易日历 (以基准 SPY 为准)
        benchmark_sym = self.config.benchmark
        if benchmark_sym not in self.price_data or self.price_data[benchmark_sym].empty:
            raise ValueError(f"缺少基准行情数据: {benchmark_sym}")

        bench_df = self.price_data[benchmark_sym]
        bench_index = bench_df.index
        trading_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10] for d in bench_index]

        # 【缺陷 B 修复：日历对齐】所有标的数据强制 reindex 到基准交易日历，消除空交易日漂移
        for sym in list(self.price_data.keys()):
            if sym == benchmark_sym:
                continue
            df = self.price_data[sym]
            reindexed = df.reindex(bench_index)
            if "Close" in reindexed:
                reindexed["Close"] = reindexed["Close"].ffill()
            for col in ["Open", "High", "Low"]:
                if col in reindexed and "Close" in reindexed:
                    reindexed[col] = reindexed[col].fillna(reindexed["Close"])
            if "Volume" in reindexed:
                reindexed["Volume"] = reindexed["Volume"].fillna(0.0)
            self.price_data[sym] = reindexed

        cash = self.config.initial_capital
        positions: Dict[str, Dict[str, Any]] = {}
        completed_trades: List[TradeRecord] = []
        daily_history: List[Dict[str, Any]] = []
        pending_buys: List[str] = []

        initial_bench_close = float(bench_df["Close"].iloc[0])
        bench_shares = self.config.initial_capital / initial_bench_close

        slippage = self.config.slippage_bps / 10000.0

        for t_idx, current_date in enumerate(trading_dates):
            # 获取当日基准净值
            bench_close_today = float(bench_df["Close"].iloc[t_idx])
            bench_equity_today = bench_shares * bench_close_today

            # =========================================================
            # Step 1: 【缺陷 A 修复：前视偏差消除】次日开盘价 (t+1 Open) 撮合买入
            # 执行前日收盘打分筛选出的 pending_buys 订单
            # =========================================================
            if pending_buys:
                current_open_equity = cash
                for sym, pos in positions.items():
                    sym_df = self.price_data[sym]
                    sym_open_val = float(sym_df["Open"].iloc[t_idx]) if "Open" in sym_df and not pd.isna(sym_df["Open"].iloc[t_idx]) else float(sym_df["Close"].iloc[t_idx])
                    current_open_equity += pos["shares"] * sym_open_val

                for sym in pending_buys:
                    if sym in positions:
                        continue
                    if sym not in self.price_data:
                        continue
                    sym_df = self.price_data[sym]
                    if t_idx >= len(sym_df):
                        continue
                    sym_open = float(sym_df["Open"].iloc[t_idx]) if "Open" in sym_df and not pd.isna(sym_df["Open"].iloc[t_idx]) else float(sym_df["Close"].iloc[t_idx])
                    if pd.isna(sym_open) or sym_open <= 0:
                        continue

                    buy_price = sym_open * (1.0 + slippage)
                    max_alloc = current_open_equity * self.config.position_cap
                    available_cash = cash * 0.85
                    alloc_dollars = min(max_alloc, available_cash)
                    if alloc_dollars <= 0:
                        continue

                    shares = int(alloc_dollars // buy_price)
                    if shares >= self.config.min_shares:
                        cost = shares * buy_price
                        cash -= cost
                        positions[sym] = {
                            "shares": shares,
                            "entry_price": buy_price,
                            "peak_price": max(buy_price, sym_open),
                            "entry_date": current_date
                        }
                pending_buys = []

            # =========================================================
            # Step 2: 盘中风控检查（硬止损、移动止损）
            # 【缺陷 D 修复：跳空低开止损计入】min(target_stop, sym_open)
            # =========================================================
            tickers_to_close = []
            for sym, pos in list(positions.items()):
                if sym not in self.price_data or t_idx >= len(self.price_data[sym]):
                    continue
                sym_df = self.price_data[sym]
                sym_close = float(sym_df["Close"].iloc[t_idx])
                sym_open = float(sym_df["Open"].iloc[t_idx]) if "Open" in sym_df and not pd.isna(sym_df["Open"].iloc[t_idx]) else sym_close
                sym_high = float(sym_df["High"].iloc[t_idx]) if "High" in sym_df and not pd.isna(sym_df["High"].iloc[t_idx]) else sym_close
                sym_low = float(sym_df["Low"].iloc[t_idx]) if "Low" in sym_df and not pd.isna(sym_df["Low"].iloc[t_idx]) else sym_close

                if pd.isna(sym_close):
                    continue

                # 更新历史峰值
                if sym_high > pos["peak_price"]:
                    pos["peak_price"] = sym_high

                # (a) GAES -6% 硬止损
                target_stop = pos["entry_price"] * (1.0 + self.config.stop_loss)
                if sym_low <= target_stop:
                    # 计入跳空低开亏损：若开盘即击穿止损线，按开盘价撮合
                    actual_exit = min(target_stop, sym_open)
                    exit_price = actual_exit * (1.0 - slippage)
                    tickers_to_close.append((sym, exit_price, "STOP_LOSS"))
                    continue

                # (b) 5% 移动止损 (High-Water Mark Drawdown)
                if pos["peak_price"] > pos["entry_price"]:
                    target_trailing = pos["peak_price"] * (1.0 - self.config.trailing_stop)
                    if sym_low <= target_trailing:
                        actual_exit = min(target_trailing, sym_open)
                        exit_price = actual_exit * (1.0 - slippage)
                        tickers_to_close.append((sym, exit_price, "TRAILING_STOP"))
                        continue

            # 执行平仓
            for sym, exit_p, reason in tickers_to_close:
                pos = positions.pop(sym)
                proceeds = pos["shares"] * exit_p
                cash += proceeds
                pnl = proceeds - (pos["shares"] * pos["entry_price"])
                pnl_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
                completed_trades.append(TradeRecord(
                    ticker=sym,
                    entry_date=pos["entry_date"],
                    entry_price=pos["entry_price"],
                    exit_date=current_date,
                    exit_price=exit_p,
                    shares=pos["shares"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=reason
                ))

            # =========================================================
            # Step 3: 收盘核算当日投资组合总权益
            # =========================================================
            eod_equity = cash
            for sym, pos in positions.items():
                sym_df = self.price_data[sym]
                sym_close = float(sym_df["Close"].iloc[t_idx])
                eod_equity += pos["shares"] * sym_close

            daily_history.append({
                "date": current_date,
                "equity": eod_equity,
                "cash": cash,
                "benchmark_equity": bench_equity_today,
                "holdings_count": len(positions)
            })

            # =========================================================
            # Step 4: 收盘后产生次日候选开仓决策 (需要至少 20 天历史进行指标计算)
            # 严格使用截止到 t_idx 的所有历史切片 (防未来函数)
            # =========================================================
            if t_idx >= 20 and t_idx < len(trading_dates) - 1:
                candidate_scores = []
                for sym in self.config.tickers:
                    if sym in positions:
                        continue
                    if sym not in self.price_data:
                        continue
                    sym_df = self.price_data[sym]
                    if t_idx >= len(sym_df):
                        continue

                    # 严格使用 t_idx (包含 t_idx 当天收盘) 进行历史打分
                    history_slice = sym_df.iloc[:t_idx + 1]
                    score = self._compute_point_in_time_score(sym, history_slice)
                    if qualifies_long(score, threshold=self.config.min_buy_score):
                        candidate_scores.append((sym, score))

                # 按得分降序排序，放入次日撮合队列
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                pending_buys = [sym for sym, _ in candidate_scores]

        # 回测结束时清算所有剩余未平仓头寸（以最终价格计算）
        final_date = trading_dates[-1]
        for sym, pos in list(positions.items()):
            sym_df = self.price_data[sym]
            sym_close = float(sym_df["Close"].iloc[-1])
            exit_p = sym_close * (1.0 - slippage)
            proceeds = pos["shares"] * exit_p
            cash += proceeds
            pnl = proceeds - (pos["shares"] * pos["entry_price"])
            pnl_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
            completed_trades.append(TradeRecord(
                ticker=sym,
                entry_date=pos["entry_date"],
                entry_price=pos["entry_price"],
                exit_date=final_date,
                exit_price=exit_p,
                shares=pos["shares"],
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason="END_OF_TEST"
            ))

        # 4. 计算综合量化指标
        equities = np.array([d["equity"] for d in daily_history])
        bench_equities = np.array([d["benchmark_equity"] for d in daily_history])
        total_days = len(daily_history)

        final_equity = float(equities[-1])
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
        benchmark_total_return = (float(bench_equities[-1]) - self.config.initial_capital) / self.config.initial_capital
        excess_return = total_return - benchmark_total_return

        # CAGR
        years = max(total_days / float(TRADING_DAYS_YEAR), 1.0 / float(TRADING_DAYS_YEAR))
        cagr = (final_equity / self.config.initial_capital) ** (1.0 / years) - 1.0
        benchmark_cagr = (float(bench_equities[-1]) / self.config.initial_capital) ** (1.0 / years) - 1.0

        strat_returns = np.diff(equities) / equities[:-1] if len(equities) > 1 else np.array([0.0])
        bench_returns = np.diff(bench_equities) / bench_equities[:-1] if len(bench_equities) > 1 else np.array([0.0])

        ann_vol = float(np.std(strat_returns) * math.sqrt(TRADING_DAYS_YEAR)) if len(strat_returns) > 1 else 0.0

        # Sharpe Ratio
        excess_cagr = cagr - self.config.risk_free_rate
        sharpe = (excess_cagr / ann_vol) if ann_vol > 1e-6 else 0.0

        # 最大回撤
        peak_series = np.maximum.accumulate(equities)
        drawdowns = (equities - peak_series) / peak_series
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # 回撤持续天数
        max_dd_duration = 0
        current_dd_duration = 0
        for dd in drawdowns:
            if dd < 0:
                current_dd_duration += 1
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
            else:
                current_dd_duration = 0

        calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

        # Beta 与 Alpha
        if len(strat_returns) > 5 and np.var(bench_returns) > 1e-8:
            covariance = np.cov(strat_returns, bench_returns)[0][1]
            beta = float(covariance / np.var(bench_returns))
            alpha = float(cagr - (self.config.risk_free_rate + beta * (benchmark_cagr - self.config.risk_free_rate)))
        else:
            beta = 1.0
            alpha = 0.0

        # 胜率与盈亏比
        win_trades = [t for t in completed_trades if t.pnl > 0]
        loss_trades = [t for t in completed_trades if t.pnl <= 0]
        win_rate = len(win_trades) / len(completed_trades) if completed_trades else 0.0

        total_gain = sum(t.pnl for t in win_trades)
        total_loss = abs(sum(t.pnl for t in loss_trades))
        profit_factor = (total_gain / total_loss) if total_loss > 1e-6 else (99.9 if total_gain > 0 else 0.0)

        return BacktestResult(
            config=asdict(self.config),
            initial_capital=self.config.initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            cagr=cagr,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            calmar_ratio=calmar,
            total_trades=len(completed_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            benchmark_total_return=benchmark_total_return,
            benchmark_cagr=benchmark_cagr,
            excess_return=excess_return,
            alpha=alpha,
            beta=beta,
            daily_history=daily_history,
            trades=[asdict(t) for t in completed_trades]
        )


def run_backtest(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    benchmark: str = "SPY"
) -> BacktestResult:
    """便利函数：快速执行历史回测"""
    cfg = BacktestConfig(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        benchmark=benchmark
    )
    engine = BacktestEngine(cfg)
    return engine.run()


def run_walk_forward(
    tickers: List[str],
    start_date: str,
    end_date: str,
    n_splits: int = 3,
    train_ratio: float = 0.7,
    initial_capital: float = 100_000.0,
    benchmark: str = "SPY",
    price_data: Optional[Dict[str, pd.DataFrame]] = None
) -> List[Dict[str, Any]]:
    """
    简易 Walk-Forward 滚动回测辅助函数：
    将指定回测区间划分为多个滚动窗口，验证策略样本外（Out-of-Sample）表现与抗过拟合能力
    """
    cfg = BacktestConfig(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        benchmark=benchmark
    )
    engine = BacktestEngine(cfg, price_data=price_data)
    if not engine.price_data:
        engine.fetch_data()

    bench_df = engine.price_data[benchmark]
    dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10] for d in bench_df.index]
    total_len = len(dates)
    if total_len < 60:
        raise ValueError(f"回测日历总长度过短 ({total_len}天)，不足以执行 Walk-Forward 分析")

    window_size = total_len // n_splits
    results = []

    for i in range(n_splits):
        start_idx = i * (window_size // 2) if n_splits > 1 else 0
        end_idx = min(start_idx + window_size, total_len)
        if end_idx - start_idx < 30:
            break

        split_dates = dates[start_idx:end_idx]
        split_point = int(len(split_dates) * train_ratio)

        train_start = split_dates[0]
        train_end = split_dates[split_point - 1]
        test_start = split_dates[split_point]
        test_end = split_dates[-1]

        # 训练集 (In-Sample)
        train_cfg = BacktestConfig(tickers=tickers, start_date=train_start, end_date=train_end, initial_capital=initial_capital, benchmark=benchmark)
        train_eng = BacktestEngine(train_cfg, price_data={k: v.loc[train_start:train_end] for k, v in engine.price_data.items()})
        train_res = train_eng.run()

        # 测试集 (Out-of-Sample)
        test_cfg = BacktestConfig(tickers=tickers, start_date=test_start, end_date=test_end, initial_capital=initial_capital, benchmark=benchmark)
        test_eng = BacktestEngine(test_cfg, price_data={k: v.loc[test_start:test_end] for k, v in engine.price_data.items()})
        test_res = test_eng.run()

        results.append({
            "split": i + 1,
            "train_range": f"{train_start} -> {train_end}",
            "test_range": f"{test_start} -> {test_end}",
            "is_return": train_res.total_return,
            "oos_return": test_res.total_return,
            "oos_excess_return": test_res.excess_return,
            "oos_sharpe": test_res.sharpe_ratio,
            "oos_max_dd": test_res.max_drawdown,
            "oos_trades": test_res.total_trades
        })

    return results
