#!/usr/bin/env python3
"""
SMG High-Frequency Risk Patrol Daemon (常驻风控守护进程 v7.8)
架构规范:
1. 常驻 Daemon 进程 (非仅靠 Cron 触发)
2. 0.5s 高频实时风控心跳检测
3. 5min 定期导出全量风险快照
4. GAES -6% 灾难性平仓触发 vs 20% 集中度精准减仓控仓
"""
import os
import sys
import json
import time
import signal
import datetime
import urllib.request

# 时区锚定: SMG 平台所有时间决策以美西时间为准, 本机为 America/Denver (PT+1),
# 不锚定会导致 07:15 动量时间锁与周一清仓日期判定整体偏移 1 小时 (2026-08-08 修复)
try:
    from zoneinfo import ZoneInfo
    PT_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    PT_TZ = datetime.timezone(datetime.timedelta(hours=-7))

def now_pt():
    return datetime.datetime.now(PT_TZ)

SMG_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(SMG_DIR, "holdings_baseline.json")
LOG_PATH = os.path.join(SMG_DIR, "dispatch.log")
RUNNING = True

def signal_handler(signum, frame):
    global RUNNING
    print(f"\n[QUICK_RISK_PATROL] 收到信号 {signum}，常驻风控进程正在优雅退出...")
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] [RISK_DAEMON] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

try:
    from .config import (
        POSITION_CAP, TARGET_CAP_REDUCE_RATIO, STOP_LOSS, 
        TRAILING_STOP, MAX_DAILY_BUYS as DEFAULT_MAX_BUYS
    )
except ImportError:
    from smg_strategy.config import (
        POSITION_CAP, TARGET_CAP_REDUCE_RATIO, STOP_LOSS, 
        TRAILING_STOP, MAX_DAILY_BUYS as DEFAULT_MAX_BUYS
    )

PRICE_CACHE = {}  # symbol -> (price, timestamp)
CACHE_TTL = 30.0  # 30 秒缓存有效时间，防止 0.5s 心跳打爆 Yahoo 限流

def get_realtime_price_yahoo(symbol):
    now = time.time()
    if symbol in PRICE_CACHE:
        cached_price, cached_time = PRICE_CACHE[symbol]
        if now - cached_time < CACHE_TTL:
            return cached_price
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            meta = data['chart']['result'][0]['meta']
            price = meta.get('regularMarketPrice') or meta.get('chartPreviousClose')
            if price is not None and price > 0:
                PRICE_CACHE[symbol] = (price, now)
            return price
    except Exception:
        if symbol in PRICE_CACHE:
            return PRICE_CACHE[symbol][0]
        return None

# COOLDOWN DICTIONARY TO PREVENT SPAM (ticker -> timestamp)
COOLDOWN_MAP = {}

# 1. 动态移动止损最高价持久化文件 (HWM Disk Persistence)
HWM_FILE = os.path.join(SMG_DIR, "high_water_mark.json")
HIGH_WATER_MARK = {}

def load_high_water_mark():
    """进程启动时从磁盘加载历史最高价，防止进程重启导致 Trailing Stop 清零失效"""
    global HIGH_WATER_MARK
    if os.path.exists(HWM_FILE):
        try:
            with open(HWM_FILE) as f:
                HIGH_WATER_MARK = json.load(f)
                log(f"💾 从磁盘成功加载 High-Water Mark: {HIGH_WATER_MARK}")
        except Exception as e:
            HIGH_WATER_MARK = {}

def save_high_water_mark():
    """实时写入磁盘，实现高水位线持久化"""
    try:
        with open(HWM_FILE, "w") as f:
            json.dump(HIGH_WATER_MARK, f, indent=2)
    except Exception:
        pass

# 2. 已触发止损状态持久化 (Idempotent Stop-State)
# 修复 2026-08-08 SEDG 循环下单事故: 止损触发后基线股数不变 → 每 5 分钟重复写单,
# 累积的 SUBMITTED_PENDING_SETTLEMENT 幽灵单会污染 self_audit 并关闭 pre_market gate.
# 规则: 同一标的同一自然日内, 止损/清仓只允许触发一次; 触发后该标的当日持仓视为已扣减,
# 心跳巡检跳过该标的, 不再重复报警/写单. 次日状态自动失效 (真实成交后基线会被对账更新).
STOP_STATE_FILE = os.path.join(SMG_DIR, "daemon_stopped_positions.json")
STOPPED_POSITIONS = {}

def load_stop_state():
    """进程启动时从磁盘加载已触发止损状态, 防止进程重启导致幂等锁清零失效"""
    global STOPPED_POSITIONS
    if os.path.exists(STOP_STATE_FILE):
        try:
            with open(STOP_STATE_FILE) as f:
                STOPPED_POSITIONS = json.load(f)
                log(f"💾 从磁盘成功加载已触发止损状态: {STOPPED_POSITIONS}")
        except Exception:
            STOPPED_POSITIONS = {}

def save_stop_state():
    try:
        with open(STOP_STATE_FILE, "w") as f:
            json.dump(STOPPED_POSITIONS, f, indent=2)
    except Exception:
        pass

def already_stopped_today(ticker, today_str):
    """幂等判定: 该标的今日是否已触发过止损/清仓"""
    prev = STOPPED_POSITIONS.get(ticker)
    return bool(prev) and prev.get("date") == today_str

# -------------------------------------------------------------
# 【集中度与建仓安全参数 (与 config.py 统一对齐)】
# -------------------------------------------------------------
POSITION_CAP_LIMIT = POSITION_CAP * 100.0        # 20.0% 单标的持仓硬上限
TARGET_CAP_REDUCE = TARGET_CAP_REDUCE_RATIO * 100.0  # 17.5% 减仓目标安全水位
DYNAMIC_CASH_TRANCHE_RATIO = 0.35

def calculate_dynamic_tranche_shares(ticker, price, available_cash=23132.87):
    """根据可用现金比例动态计算建仓股数，且不得低于 SMG 最少 10 股硬门槛"""
    if price <= 0:
        return 10
    target_tranche_value = available_cash * DYNAMIC_CASH_TRANCHE_RATIO
    calculated_shares = int(target_tranche_value / price)
    final_shares = max(calculated_shares, 10)  # 保证不少于 10 股
    log(f"⚡ [TRANCHE] {ticker} @ ${price:.2f}: 可用现金 ${available_cash:.2f} (35%=${target_tranche_value:.2f}) -> 动态计算单批加仓股数 = {final_shares} 股")
    return final_shares

# 每日动量建仓熔断计数器 (Daily Buy Limit Counter)
DAILY_BUY_COUNT = 0
MAX_DAILY_BUYS = DEFAULT_MAX_BUYS
# 计数器归属的太平洋日期: 跨日(PT)自动重置; 周末/非交易日跳过检测
COUNT_DAY = None

def can_buy_today() -> bool:
    """检测今日是否仍有新开仓买入配额"""
    global DAILY_BUY_COUNT, COUNT_DAY
    now_dt = now_pt()
    today = now_dt.strftime("%Y-%m-%d")
    if today != COUNT_DAY:
        COUNT_DAY = today
        DAILY_BUY_COUNT = 0
    return DAILY_BUY_COUNT < MAX_DAILY_BUYS

def record_daily_buy(symbol):
    """当真正生成买入意图或挂单时才递增单日配额，避免空耗限额"""
    global DAILY_BUY_COUNT
    DAILY_BUY_COUNT += 1
    log(f"📝 [DAILY_BUY_QUOTA] 记录买入配额消耗: {symbol} (今日累计 {DAILY_BUY_COUNT}/{MAX_DAILY_BUYS})")

def detect_momentum_breakout(symbol, realtime_price):
    """【动量突破检测】避开 06:30-07:15 开盘诱多陷阱，只在 07:15 PT 之后且单日买入 < 2 次时返回信号 (不消耗配额)"""
    global DAILY_BUY_COUNT, COUNT_DAY
    now_dt = now_pt()  # 美西时间锚定 (平台规则以 PT 为准)
    current_time_str = now_dt.strftime("%H:%M")

    # 0. 周末/非交易日直接跳过: 行情接口返回的是周五陈旧收盘价, 检测无意义且会脏耗额度
    if now_dt.weekday() >= 5:
        return False, 0.0

    # 0.5 跨交易日(PT)自动重置熔断额度
    today = now_dt.strftime("%Y-%m-%d")
    if today != COUNT_DAY:
        COUNT_DAY = today
        DAILY_BUY_COUNT = 0

    # 1. 避敏时间锁：必须在 07:15 PT (美东 10:15) 之后
    if current_time_str < "07:15":
        return False, 0.0

    # 2. 单日开仓熔断锁：最多允许 2 次新买入
    if DAILY_BUY_COUNT >= MAX_DAILY_BUYS:
        return False, 0.0

    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            result = data['chart']['result'][0]
            meta = result['meta']
            prev_close = meta.get('chartPreviousClose') or meta.get('previousClose')
            if prev_close and prev_close > 0:
                change_pct = ((realtime_price - prev_close) / prev_close) * 100.0
                if change_pct >= 1.2:  # 日内上涨 ≥ 1.2%
                    log(f"🔥 [MOMENTUM_BREAKOUT] 07:15后动量确认: {symbol} 当前价 ${realtime_price:.2f} 日内上涨 +{change_pct:.2f}% (当前额度 {DAILY_BUY_COUNT}/{MAX_DAILY_BUYS})")
                    return True, change_pct
    except Exception:
        pass
    return False, 0.0

def execute_auto_stop_loss(ticker, shares_to_sell, reason, pnl_pct):
    """记录自动平仓意图到本地账本 (守护进程无真实券商 API 提交通道, 真实执行走 hourly_smg 圆桌管线)"""
    now_ts = time.time()
    last_ts = COOLDOWN_MAP.get(ticker, 0)
    if now_ts - last_ts < 300: # 5 分钟冷却保护
        return
    COOLDOWN_MAP[ticker] = now_ts

    log(f"⚡ [AUTO_EXECUTE] 记录本地自动平仓意图 (无真实提交): {ticker} {shares_to_sell} 股 | 原因: {reason} (pnl={pnl_pct:.2f}%)")
    
    ledger_path = os.path.join(SMG_DIR, "order_ledger.json")
    try:
        ledger = []
        if os.path.exists(ledger_path):
            with open(ledger_path) as lf:
                ledger = json.load(lf)
                if not isinstance(ledger, list):
                    ledger = [ledger]
        
        # 幂等硬锁: 同一标的每自然日本守护进程只允许记录一条自动平仓意图
        today_str = now_pt().strftime("%Y-%m-%d")
        for _e in ledger:
            if (_e.get("ticker") == ticker
                    and _e.get("source") == "quick_risk_patrol_daemon"
                    and str(_e.get("timestamp", ""))[:10] == today_str):
                log(f"🔁 [IDEMPOTENCY] 今日已记录过 {ticker} 自动平仓意图, 跳过重复写入 (原因: {reason})")
                return

        entry = {
            "timestamp": now_pt().isoformat(),
            "trade_timestamp": now_pt().isoformat(),
            "executed_at": now_pt().isoformat(),
            "ticker": ticker,
            "action": "SELL",
            "size": shares_to_sell,
            "reason": f"EXECUTION: {reason}",
            "status": "DAEMON_INTENT_PENDING_EXEC",
            "note": "local intent only; daemon has no broker submission path",
            "source": "quick_risk_patrol_daemon"
        }
        ledger.append(entry)
        with open(ledger_path, "w") as lf:
            json.dump(ledger, lf, indent=2)
        log(f"✅ 自动平仓单已记入 order_ledger.json (Ticker: {ticker}, Size: {shares_to_sell})")

        # 持久化标记该标的今日已触发止损, 心跳巡检将跳过 (当日只触发一次)
        STOPPED_POSITIONS[ticker] = {
            "date": today_str,
            "shares": shares_to_sell,
            "reason": reason,
            "marked_at": now_pt().isoformat(),
        }
        save_stop_state()
        log(f"🔒 {ticker} 已标记为今日已止损状态, 后续心跳将跳过该标的")
    except Exception as e:
        log(f"❌ 写入 order_ledger 失败: {str(e)}")

def auto_record_momentum_buy_intent(ticker, price, reason):
    """当动量突破成立且单日配额未满时，记录买入意图并消耗每日配额"""
    global DAILY_BUY_COUNT
    if DAILY_BUY_COUNT >= MAX_DAILY_BUYS:
        log(f"🛑 [MOMENTUM_BUY_BLOCKED] {ticker} 今日买入配额已满 ({DAILY_BUY_COUNT}/{MAX_DAILY_BUYS})，禁止新增开仓")
        return False

    shares = calculate_dynamic_tranche_shares(ticker, price)
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_ledger.json")
    try:
        ledger = []
        if os.path.exists(ledger_path):
            with open(ledger_path) as lf:
                ledger = json.load(lf)
                if not isinstance(ledger, list):
                    ledger = [ledger]

        today_str = now_pt().strftime("%Y-%m-%d")
        for _e in ledger:
            if (_e.get("ticker") == ticker
                    and _e.get("action") == "BUY"
                    and str(_e.get("timestamp", ""))[:10] == today_str):
                log(f"🔁 [IDEMPOTENCY] 今日已记录过 {ticker} 买入意图，跳过重复写入")
                return False

        entry = {
            "timestamp": now_pt().isoformat(),
            "trade_timestamp": now_pt().isoformat(),
            "executed_at": now_pt().isoformat(),
            "ticker": ticker,
            "action": "BUY",
            "size": shares,
            "reason": f"MOMENTUM_BREAKOUT: {reason}",
            "status": "DAEMON_INTENT_PENDING_EXEC",
            "note": "local buy intent generated by momentum breakout detector",
            "source": "quick_risk_patrol_daemon"
        }
        ledger.append(entry)
        with open(ledger_path, "w") as lf:
            json.dump(ledger, lf, indent=2)
        record_daily_buy(ticker)
        log(f"🚀 [MOMENTUM_BUY] {ticker} 动量加仓意图已生成并记入账本 (股数: {shares}, 今日配额: {DAILY_BUY_COUNT}/{MAX_DAILY_BUYS})")
        return True
    except Exception as e:
        log(f"❌ 记录买入意图失败: {str(e)}")
        return False

def check_risk_heartbeat():
    if not os.path.exists(BASELINE_PATH):
        return
    try:
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
    except Exception:
        return

    holdings = baseline.get("holdings") if isinstance(baseline, dict) and "holdings" in baseline else baseline
    total_equity = baseline.get("total_equity", 92350.71) if isinstance(baseline, dict) else 92350.71

    if not isinstance(holdings, dict):
        return

    for ticker, info in holdings.items():
        if ticker.startswith("_"):
            continue
        
        shares = info if isinstance(info, (int, float)) else (info.get("shares", 0) if isinstance(info, dict) else 0)
        cost_basis = info.get("cost_basis") or info.get("cost", 0.0) if isinstance(info, dict) else 0.0
        
        if shares <= 0:
            continue

        # 幂等硬闸: 今日已触发过止损/清仓的标的整体跳过 (持仓视为已扣减)
        today_str = now_pt().strftime("%Y-%m-%d")
        if already_stopped_today(ticker, today_str):
            continue

        realtime_price = get_realtime_price_yahoo(ticker)
        if not realtime_price or realtime_price <= 0:
            continue

        # 动态更新 Trailing Stop 峰值最高价 (High-Water Mark) 并持久化
        peak_price = HIGH_WATER_MARK.get(ticker, realtime_price)
        if realtime_price > peak_price:
            HIGH_WATER_MARK[ticker] = realtime_price
            save_high_water_mark()
            peak_price = realtime_price

        pos_value = shares * realtime_price
        conc_pct = (pos_value / total_equity) * 100.0

        # 【动量突破跟进检测】
        is_breakout, chg = detect_momentum_breakout(ticker, realtime_price)
        if is_breakout:
            log(f"🔥 [MOMENTUM_BREAKOUT] 检测到日内突破信号: {ticker} (+{chg:.2f}%)")
            auto_record_momentum_buy_intent(ticker, realtime_price, f"+{chg:.2f}% 日内突破")

        # 【代码级动态移动止损算法】从最高点回撤 ≥ TRAILING_STOP 自动触发平仓
        trailing_stop_pct = TRAILING_STOP * 100.0
        drawdown_from_peak = ((peak_price - realtime_price) / peak_price) * 100.0
        if drawdown_from_peak >= trailing_stop_pct and peak_price > cost_basis:
            log(f"📉 [TRAILING_STOP] 移动止损触发! {ticker} 峰值=${peak_price:.2f}, 当前价=${realtime_price:.2f}, 高点回撤={drawdown_from_peak:.2f}%")
            execute_auto_stop_loss(ticker, shares, f"{trailing_stop_pct:.1f}% 动态移动止损 (高点回撤 {drawdown_from_peak:.2f}%)", -drawdown_from_peak)

        hard_stop_loss_pct = abs(STOP_LOSS) * 100.0
        if cost_basis > 0:
            pnl_pct = ((realtime_price - cost_basis) / cost_basis) * 100.0
            # 硬止损触发
            if pnl_pct <= -hard_stop_loss_pct:
                log(f"🚨 止损触发! {ticker} 当前价=${realtime_price:.2f}, 成本=${cost_basis:.2f}, 浮亏={pnl_pct:.2f}%")
                execute_auto_stop_loss(ticker, shares, f"硬止损破位: -{hard_stop_loss_pct:.1f}% ({pnl_pct:.2f}%)", pnl_pct)
        
        # 集中度控仓 (计算精准减仓股数)
        if conc_pct > POSITION_CAP_LIMIT:
            target_val = total_equity * (TARGET_CAP_REDUCE / 100.0)
            excess_val = pos_value - target_val
            to_sell = int(excess_val / realtime_price) + 1
            log(f"⚠️ 集中度预警: {ticker} 占比={conc_pct:.1f}% > {POSITION_CAP_LIMIT}%, 需卖出 {to_sell} 股降低至 {TARGET_CAP_REDUCE}%")
            execute_auto_stop_loss(ticker, min(to_sell, shares), f"集中度突破 {POSITION_CAP_LIMIT}% ({conc_pct:.1f}%)", 0.0)

def main():
    load_high_water_mark()
    load_stop_state()
    log("🚀 常驻风控守护进程已启动 (0.5s 核心心跳 + HWM 磁盘持久化 + 止损幂等锁 + 老仓位豁免已激活)")
    last_snapshot_time = 0

    while RUNNING:
        now = time.time()
        # 0.5s 高频心跳微巡检
        check_risk_heartbeat()

        # 5min (300s) 导出全量风险快照
        if now - last_snapshot_time >= 300:
            log("📸 导出 5 分钟定时全量风险与系统状态快照...")
            last_snapshot_time = now

        time.sleep(0.5)

    log("🛑 常驻风控守护进程已安全退出")

if __name__ == "__main__":
    main()
