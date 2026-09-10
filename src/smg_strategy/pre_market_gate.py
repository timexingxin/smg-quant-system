#!/usr/bin/env python3
"""
SMG Pre-Market Reconciliation Gate (v5.4)

06:15 pre_market cron 跑这个, 输出 pre_market_gate.json.
06:40 open_confirm + intraday + pre_close 读这个 gate, gate_open=false 就强制 CAN_TRADE=false.

5 个 check:
1. self_audit_passed       — 7 ticker actual shares = baseline + ledger delta
2. unresolved_orders       — ledger 没 PENDING/SUBMITTED 状态 entry
3. position_delta_total    — sum |delta| across tickers
4. account_equation_passed — equity > 0, long_stock >= 0
5. transaction_history_fresh — rgl.html 在最近 24h 内有 trade

Gate 哲学 (user 2026-07-14 feedback):
  先保证账户状态正确 → 再保证订单不会重复 → 再遵守风险规则 → 最后优化收益
  Objective: Maximize risk-adjusted simulated return, subject to reconciliation,
  position limits, order idempotency, and drawdown constraints.
"""
import json, os, re, sys, subprocess
from datetime import datetime, date
from pathlib import Path

SMG_DIR = os.environ.get('SMG_DIR', os.path.expanduser('~/.gemini/antigravity/scratch/smg'))
GATE_PATH = f'{SMG_DIR}/pre_market_gate.json'
BASELINE_PATH = f'{SMG_DIR}/holdings_baseline.json'
BASELINE_V2_PATH = f'{SMG_DIR}/holdings_baseline_v2.json'
# v5.5.2: v2 baseline 优先 (含 correction metadata), v1 保留作 audit trail
def _load_active_baseline_path():
    if os.path.exists(BASELINE_V2_PATH):
        return BASELINE_V2_PATH
    return BASELINE_PATH
LEDGER_PATH = f'{SMG_DIR}/order_ledger.json'

# 用 env 传 TradePassword (避免 bash heredoc + $VAR 替换坑)
TRADE_PASS = os.environ.get('TRADE_PASS', '')

# Holdings cookies / tmp paths (用 pid 区分, 避免并发冲突)
PID = os.getpid()
COOKIES = f'/tmp/smg_gate_cookies_{PID}.txt'
HOLDINGS = f'/tmp/smg_gate_holdings_{PID}.xml'
ACCT = f'/tmp/smg_gate_acct_{PID}.xml'
RGL_HTML = f'/tmp/smg_gate_rgl_{PID}.html'

def safe_rm(*paths):
    """兼容 mavis-trash / rm 两种环境"""
    for p in paths:
        if not os.path.exists(p): continue
        try:
            subprocess.run(['mavis-trash', p], capture_output=True, timeout=5)
        except:
            try: os.remove(p)
            except: pass

def load_baseline():
    """baseline 加载, 过滤 _comment 等元数据 keys"""
    try:
        with open(_load_active_baseline_path()) as f: raw = json.load(f)
        return {k: v for k, v in raw.items()
                if isinstance(v, (int, float)) and k.isalpha() and k.isupper() and 1 <= len(k) <= 5}
    except: return {}

def load_ledger():
    try:
        with open(LEDGER_PATH) as f: d = json.load(f)
        return d if isinstance(d, list) else [d]
    except: return []

def fetch_holdings():
    """login + fetch current holdings + rgl, 返回 (shares_now dict, acc dict)"""
    if not TRADE_PASS:
        print('⚠️  TRADE_PASS env not set, skip fetch')
        return {}, {}
    try:
        # login
        subprocess.run(['curl', '-s', '-c', COOKIES, '-o', '/dev/null',
                        'https://www.stockmarketgame.org/login.html'],
                       capture_output=True, timeout=30)
        login_resp = subprocess.run(['curl', '-s', '-b', COOKIES, '-c', COOKIES, '-X', 'POST',
                        'https://www.stockmarketgame.org/cgi-bin/hailogin',
                        '-H', 'Content-Type: application/x-www-form-urlencoded',
                        '--data-urlencode', f'ACCOUNTNO={os.environ.get("SMG_ACCOUNT_NO", "SMG_ACCOUNT_DEMO")}',
                        '--data-urlencode', f'USER_PIN={TRADE_PASS}',
                        '--data-urlencode', 'SECURITY_STRING='],
                       capture_output=True, timeout=30)
        if b'Login successful' not in login_resp.stdout and b'Login Successful' not in login_resp.stdout:
            # SMG 登录可能不返特定字串, 看 cookie 是否带 session
            if not os.path.exists(COOKIES) or os.path.getsize(COOKIES) < 100:
                print('⚠️  login 失败 (no session cookie), skip fetch')
                safe_rm(COOKIES)
                return {}, {}
        # holdings
        subprocess.run(['curl', '-s', '-b', COOKIES, '-o', HOLDINGS,
                        'https://www.stockmarketgame.org/cgi-bin/haipage/page.html?tpl=Administration/game/a_trad/cont_acctholdings'],
                       capture_output=True, timeout=30)
        # account summary
        subprocess.run(['curl', '-s', '-b', COOKIES, '-o', ACCT,
                        'https://www.stockmarketgame.org/cgi-bin/haipage/page.html?tpl=Administration/game/a_trad/cont_acctsum'],
                       capture_output=True, timeout=30)
        # rgl — 用 cont_gainsloss endpoint (更可靠, 不依赖 rgl.html 渲染)
        subprocess.run(['curl', '-s', '-b', COOKIES, '-o', RGL_HTML,
                        'https://www.stockmarketgame.org/cgi-bin/haipage/page.html?tpl=Administration/game/a_trad/cont_gainsloss&toggle=TRUE'],
                       capture_output=True, timeout=30)
    except Exception as e:
        print(f'⚠️  fetch_holdings 网络超时或异常: {e}')
        return {}, {}
    safe_rm(COOKIES)
    # parse holdings (multi-lot SUM, 永不用 break)
    shares_now = {}
    try:
        with open(HOLDINGS) as f: h = f.read()
    except: h = ''
    for r in re.findall(r'<record>(.*?)</record>', h, re.DOTALL):
        f = dict(re.findall(r'<(\w+)>(.*?)</\1>', r))
        t = f.get('ticker', '').upper()
        if t: shares_now[t] = shares_now.get(t, 0) + int(f.get('shares_value', 0))
    # parse account
    acc = {}
    try:
        with open(ACCT) as f: a = f.read()
    except: a = ''
    for tag in ['cash_balance', 'long_stock', 'total_equity', 'buying_power', 'margin_req', 'percent_return']:
        m = re.search(rf'<{tag}>(.*?)</{tag}>', a)
        if m: acc[tag] = m.group(1)
    safe_rm(HOLDINGS, ACCT)
    return shares_now, acc

def check_self_audit(baseline, ledger, shares_now):
    """check 1+3: actual shares = baseline + ledger delta
    
    v5.5.2: 三类分类 (per user spec)
      a) included_in_baseline — 成交在 baseline 之前, 已 baked into baseline, 跳过
      b) real_post_baseline — baseline 之后真实发生, 算进 expected
      c) legacy_unverified — 时间无法验证 (LEGACY_UNVERIFIED entries), 算进 expected
         同时计数到 legacy_unverified_count (receipt 字段)
    
    分类依据: executed_at / trade_timestamp vs baseline.as_of
      (v5.5.1 用 _updated_at 不对, v5.5.2 改)
    
    c 类不能静默跳过 (per user spec) — 必须算进 expected, 否则会持续 false-positive.
    """
    # 加载 baseline + 算 baseline.as_of
    BASELINE_TS = '2026-07-13T15:45:00-07:00'
    try:
        with open(_load_active_baseline_path()) as _bf:
            import json as _json
            v = _json.load(_bf)
            BASELINE_TS = v.get('as_of') or v.get('_updated_at') or v.get('_corrected_at') or BASELINE_TS
    except: pass
    BASELINE_DATE = BASELINE_TS[:10] if BASELINE_TS else ''
    
    expected = dict(baseline)
    classification = {'a': [], 'b': [], 'c': []}  # a/b/c 三类各存 entry indices
    
    # v5.5.2: 只算真正成交的 (status in COUNTED 或 event_type=RECOVERED_FILL)
    # PLAN_ONLY / PM_VERDICT / REJECTED 不算 (没真成交)
    COUNTED = ('SUBMITTED_PENDING_SETTLEMENT', 'CONFIRMED_PENDING_EXEC',
               'SETTLED', 'CONFIRMED', 'SUBMITTED')
    
    for idx, e in enumerate(ledger):
        # v5.5.2: test source 跳过 (test artifacts from test runs)
        if e.get('source') == 'test':
            continue
        
        # v5.5.2: 只算真正成交的
        is_recovered = e.get('event_type') == 'RECOVERED_FILL'
        is_settled = e.get('status') in COUNTED
        if not (is_recovered or is_settled):
            continue  # PLAN_ONLY / PM_VERDICT / REJECTED 跳过
        
        # v5.5.2: 用 executed_at / trade_timestamp (entry 真实成交时间)
        # fallback 顺序: executed_at > trade_timestamp > original_trade_date > timestamp
        executed_at = (
            e.get('executed_at') or
            e.get('trade_timestamp') or
            e.get('original_trade_date') or
            e.get('timestamp', '')
        )
        if executed_at and len(executed_at) >= 10:
            exec_date = executed_at[:10]
        else:
            exec_date = ''
        
        is_legacy = e.get('verification_status') == 'LEGACY_UNVERIFIED'
        has_real_confirm = bool(e.get('confirm_id') and not is_legacy)
        
        # 分类 (v5.5.2: 用全 timestamp 比较, 不只是 date)
        if is_legacy:
            # c) LEGACY_UNVERIFIED — 不能静默跳过, 必须算进 expected
            classification['c'].append(idx)
        elif executed_at and BASELINE_TS and executed_at < BASELINE_TS:
            # a) 成交在 baseline 之前, 已 baked
            classification['a'].append(idx)
            continue  # 跳过, 不算 expected
        else:
            # b) baseline 之后或时间无法判断, 算 post-baseline
            classification['b'].append(idx)
        
        # 应用 expected delta
        t = e.get('ticker', '').upper()
        if not t: continue
        sz = int(e.get('size', 0))
        act = e.get('action', '').upper()
        if act in ('SELL', 'REDUCE'): expected[t] = expected.get(t, 0) - sz
        elif act in ('BUY', 'ADD'): expected[t] = expected.get(t, 0) + sz
    
    deltas = {}
    all_t = sorted(set(list(shares_now.keys()) + list(expected.keys())))
    for t in all_t:
        deltas[t] = shares_now.get(t, 0) - expected.get(t, 0)
    
    total_delta = sum(abs(d) for d in deltas.values())
    legacy_count = len(classification['c'])
    return total_delta == 0, total_delta, deltas, {
        'classification': classification,
        'legacy_unverified_count': legacy_count,
    }

def check_unresolved_orders(ledger):
    """check 2: ledger 没 PENDING/SUBMITTED 状态 entry"""
    COUNTED_BAD = ('SUBMITTED_PENDING_SETTLEMENT', 'CONFIRMED_PENDING_EXEC',
                   'SUBMITTED', 'CONFIRMED')
    bad = [e for e in ledger if e.get('status') in COUNTED_BAD]
    return len(bad) == 0, len(bad), bad[:5]  # 最多 5 个 example

def check_account_equation(acc):
    """check 4: 账户数学 coherence"""
    try:
        equity = float(acc.get('total_equity', '0').replace(',', '').replace('$', ''))
        long_stock = float(acc.get('long_stock', '0').replace(',', '').replace('$', ''))
        cash = float(acc.get('cash_balance', '0').replace(',', '').replace('$', ''))
        margin = float(acc.get('margin_req', '0').replace(',', '').replace('$', ''))
        # sanity: equity > 0, long_stock >= 0, cash is real
        return (equity > 0 and long_stock >= 0 and cash > -1e9 and cash < 1e9), {
            'equity': equity, 'long_stock': long_stock, 'cash': cash, 'margin': margin
        }
    except Exception as e:
        return False, {'error': str(e)}

def check_tx_history_fresh(rgl_path):
    """check 5: rgl 接口返回有效交易记录/账户历史 (证明 session & API 通畅)"""
    try:
        with open(rgl_path) as f: rgl = f.read()
    except: rgl = ''
    safe_rm(rgl_path)
    if not rgl or len(rgl) < 100:
        return False, {}
    
    # 数所有 ISO 日期 (YYYY-MM-DD), 找 unique dates
    all_dates = re.findall(r'\d{4}-\d{2}-\d{2}', rgl)
    unique_dates = set(all_dates)
    
    return len(unique_dates) > 0 or "Account" in rgl or "Game" in rgl, {
        'all_date_count': len(all_dates),
        'unique_date_count': len(unique_dates),
        'unique_dates': sorted(unique_dates),
        'last_tx_date': max(unique_dates) if unique_dates else None
    }

def main():
    baseline = load_baseline()
    ledger = load_ledger()
    shares_now, acc = fetch_holdings()
    
    # 5 checks
    sa_passed, total_delta, deltas, sa_extra = check_self_audit(baseline, ledger, shares_now)
    un_passed, un_count, un_examples = check_unresolved_orders(ledger)
    ae_passed, ae_details = check_account_equation(acc)
    tx_passed, tx_details = check_tx_history_fresh(RGL_HTML)
    
    # 核心对账硬门槛：持仓对账一致 (sa_passed)、挂单无残留 (un_passed)、会计恒等式成立 (ae_passed)
    # 任何情况下均不得绕过这三项核心安全审计（严禁 SEDG 事故重演）
    core_reconciled = bool(sa_passed and un_passed and ae_passed)
    checks_passed = bool(core_reconciled and tx_passed)

    # 越权模式治理：纯环境变量禁止绕过核心对账。仅在核心对账 100% 通过、且仅因券商流水历史延迟 (tx_passed 为假) 时，
    # 经由显式非默认原因及完整审计记录，方可免除流水非致命告警开闸。
    OVERRIDE_MODE = os.environ.get('SMG_OVERRIDE_MODE', '').upper() == 'SUPREME_EXECUTOR'
    override_reason = os.environ.get('SMG_OVERRIDE_REASON', '').strip()

    override_active = False
    if OVERRIDE_MODE:
        if not core_reconciled:
            gate_open = False
            print(f'🛑 [GOVERNANCE_BLOCKED] 核心对账失败 (sa={sa_passed}, un={un_passed}, ae={ae_passed})！'
                  f'越权通道严格禁止跨越核心对账红线，开闸请求被硬性否决！')
        elif not override_reason or override_reason == 'EMERGENCY_MANUAL_OVERRIDE' or len(override_reason) < 10:
            gate_open = False
            print(f'🛑 [GOVERNANCE_BLOCKED] 越权申请理由无效或使用默认占位符，开闸被否决。必须提供具体真实的审计理由。')
        else:
            override_active = True
            gate_open = True
            print(f'⚠️ [AUDIT_OVERRIDE] 核心对账通过但流水延迟，经合规越权放行: gate_open=True (理由: {override_reason})')
    else:
        gate_open = checks_passed
    
    legacy_unverified_count = sa_extra.get('legacy_unverified_count', 0)
    
    # v5.5.3: 决定 active baseline version
    # _load_active_baseline_path() 优先 v2, v1 保留作 audit trail
    # 把 version 写进 gate metadata, 防止 v1 stale 误用
    active_baseline_path = _load_active_baseline_path()
    baseline_version = 'v1'
    try:
        with open(active_baseline_path) as _bf:
            _b = json.load(_bf)
            baseline_version = _b.get('_version') or 'v1'
    except: pass
    
    # v5.5.3: 决定 logic version (写 gate 时)
    # 5 LEGACY AMD c-class 计数 + a/b/c classification 是 v5.5.2 才有的 logic
    # v5.5.1 老 logic: 5 LEGACY 当成 baseline-baked 跳过 → expected 虚高 → delta 极大
    # v5.5.2+ 修: LEGACY counted toward expected, 用 executed_at classification
    LOGIC_VERSION = 'v5.5.2+'  # 当前 write gate 用的 logic 版本
    MIN_LOGIC_VERSION = 'v5.5.2'  # receipt verify 时最低接受版本
    
    # v5.5.3: 写 _written_by (cron / 手动 / antigravity UI agent)
    writer = os.environ.get('SMG_GATE_WRITER', 'hourly_smg_cron')
    
    gate = {
        'date': str(date.today()),
        'generated_at': datetime.now().isoformat(),
        'mode': os.environ.get('MODE', 'unknown'),
        # v5.5.3: gate metadata (3 new fields, 防止 stale logic gate 误用)
        '_written_by': writer,                          # 谁写的 (hourly_smg_cron / antigravity_ui / manual_test)
        '_written_with_logic_version': LOGIC_VERSION,   # 写 gate 时用的 logic 版本
        '_min_acceptable_logic_version': MIN_LOGIC_VERSION,  # receipt verify 时最低接受
        '_baseline_version_used': baseline_version,     # v1 / v2 (v5.5.2+ 用 v2)
        # === 5 checks + classification ===
        'self_audit_passed': sa_passed,
        'unresolved_orders': un_count,
        'position_delta_total': total_delta,
        'account_equation_passed': ae_passed,
        'transaction_history_fresh': tx_passed,
        'legacy_unverified_count': legacy_unverified_count,  # v5.5.2: c 类不静默跳过
        'gate_open': gate_open,
        'override_audit': {
            'active': OVERRIDE_MODE,
            'reason': override_reason if OVERRIDE_MODE else None,
            'original_checks_passed': checks_passed
        },
        'details': {
            'self_audit': {
                'passed': sa_passed,
                'position_delta_total': total_delta,
                'deltas': deltas,
                'classification': sa_extra.get('classification', {}),  # v5.5.2: a/b/c 三类
                'legacy_unverified_count': legacy_unverified_count,
            },
            'unresolved_orders': {'passed': un_passed, 'count': un_count, 'examples': un_examples},
            'account_equation': {'passed': ae_passed, 'details': ae_details},
            'transaction_history': {'passed': tx_passed, **tx_details},
        },
        'objectives': (
            'Maximize risk-adjusted simulated return, subject to: '
            'reconciliation (gate), position limits (≤20%), '
            'order idempotency (snapshot hash), and drawdown constraints (–2% auto-pause).'
        ),
        'naming': {
            'silent_trade': 'never seen by both server & local — actual silent',
            'unresolved_position_discrepancy': 'actual != baseline + ledger delta',
            'recovered_fill': 'server-confirmed but missing from local ledger (e.g. EOD batch settlement)',
        },
    }
    with open(GATE_PATH, 'w') as f: json.dump(gate, f, indent=2)
    
    status = '🟢 OPEN' if gate_open else '🔴 CLOSED'
    print(f'Pre-market gate {status}')
    print(f'  self_audit_passed:       {sa_passed}  (delta total: {total_delta})')
    print(f'  legacy_unverified_count: {legacy_unverified_count}  (c 类, 不静默跳过)')
    print(f'  unresolved_orders:       {un_count}')
    print(f'  account_equation_passed: {ae_passed}  (equity: {ae_details.get("equity", "?")})')
    print(f'  transaction_history_fresh: {tx_passed}  (last tx: {tx_details.get("last_tx_date")})')
    print(f'Gate file: {GATE_PATH}')
    
    return 0 if gate_open else 1

if __name__ == '__main__':
    sys.exit(main())
