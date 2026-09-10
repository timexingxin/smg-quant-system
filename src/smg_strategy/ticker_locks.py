#!/usr/bin/env python3
"""
SMG Per-Ticker Lock Manager (v5.4)

从 server_lockout.json (v1: {TICKER: [timestamps]}) 迁移到 ticker_locks.json
(v2: per-ticker 结构含 status/reason/attempt_count/unlock_condition).

新格式 (ticker_locks.json):
{
  "version": 2,
  "updated_at": "...",
  "tickers": {
    "MU": {
      "status": "TICKER_LOCKED" | "ACTIVE",
      "reason": "REPEATED_INSUFFICIENT_POSITION_SERVER_ERROR",
      "first_locked_at": <unix_ts>,
      "last_attempt_at": <unix_ts>,
      "attempt_count": N,
      "max_attempts_per_session": 0,  # 0 = 永远不重试 (人工解锁)
      "unlock_condition": "fresh_holdings_AND_server_validation_pass",
      "lock_history": [
        {"at": <ts>, "session": "HH:MM", "error": "...", "n": 1}
      ]
    }
  }
}

CLI:
    python3 ticker_locks.py check TICKER       # 返回 LOCKED / ACTIVE / UNKNOWN
    python3 ticker_locks.py record TICKER ERROR  # 记录一次 attempt
    python3 ticker_locks.py unlock TICKER      # 人工解锁
    python3 ticker_locks.py migrate            # 从 server_lockout.json (v1) 迁移
    python3 ticker_locks.py status             # print 整个 JSON
"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path

SMG_DIR = os.environ.get('SMG_DIR', os.path.expanduser('~/.gemini/antigravity/scratch/smg'))
LOCKS_PATH = f'{SMG_DIR}/ticker_locks.json'
OLD_PATH = f'{SMG_DIR}/server_lockout.json'
LOCK_THRESHOLD = 2  # v1 旧格式 10 min 内 2+ 次 attempts 升级为 TICKER_LOCKED

def load_locks():
    if not os.path.exists(LOCKS_PATH):
        return {'version': 2, 'updated_at': '', 'tickers': {}}
    try:
        with open(LOCKS_PATH) as f: return json.load(f)
    except: return {'version': 2, 'updated_at': '', 'tickers': {}}

def save_locks(locks):
    locks['updated_at'] = datetime.now().isoformat()
    with open(LOCKS_PATH, 'w') as f: json.dump(locks, f, indent=2)

def migrate():
    """从 server_lockout.json (v1) 一次性迁移到 ticker_locks.json (v2)"""
    if not os.path.exists(OLD_PATH):
        print('No old server_lockout.json, skip migrate')
        return
    locks = load_locks()
    try:
        with open(OLD_PATH) as f: old = json.load(f)
    except:
        print(f'⚠️  failed to parse {OLD_PATH}, skip')
        return
    if not isinstance(old, dict):
        print(f'⚠️  {OLD_PATH} not dict, skip')
        return
    now = time.time()
    for ticker, ts_list in old.items():
        if not isinstance(ts_list, list) or not ts_list: continue
        if len(ts_list) >= LOCK_THRESHOLD:
            # 升级为 TICKER_LOCKED
            history = []
            for i, ts in enumerate(ts_list):
                history.append({
                    'at': ts, 'session': datetime.fromtimestamp(ts).strftime('%H:%M'),
                    'error': 'Insufficient position (migrated from v1)', 'n': i + 1
                })
            locks['tickers'][ticker] = {
                'status': 'TICKER_LOCKED',
                'reason': 'REPEATED_INSUFFICIENT_POSITION_SERVER_ERROR',
                'first_locked_at': ts_list[0],
                'last_attempt_at': ts_list[-1],
                'attempt_count': len(ts_list),
                'max_attempts_per_session': 0,  # 0 = 永远不重试
                'unlock_condition': 'fresh_holdings_AND_server_validation_pass',
                'lock_history': history,
            }
            print(f'  {ticker}: {len(ts_list)} attempts → TICKER_LOCKED')
        else:
            print(f'  {ticker}: {len(ts_list)} attempts (below threshold), no lock')
    save_locks(locks)
    # 备份旧文件
    backup = f'{OLD_PATH}.v1.bak.{int(now)}'
    os.rename(OLD_PATH, backup)
    print(f'✅ Migrated to {LOCKS_PATH}, old file backed up to {backup}')

def check(ticker):
    locks = load_locks()
    t = locks.get('tickers', {}).get(ticker.upper(), {})
    return t.get('status', 'ACTIVE')

def record(ticker, error):
    """记录一次 attempt. 达到 max_attempts_per_session 升级为 TICKER_LOCKED."""
    locks = load_locks()
    t = ticker.upper()
    tk = locks.get('tickers', {}).get(t, {})
    if not tk:
        tk = {
            'status': 'ACTIVE', 'reason': '',
            'first_locked_at': 0, 'last_attempt_at': 0,
            'attempt_count': 0, 'max_attempts_per_session': 3,
            'unlock_condition': 'fresh_holdings_AND_server_validation_pass',
            'lock_history': [],
        }
    tk['last_attempt_at'] = time.time()
    tk['attempt_count'] = tk.get('attempt_count', 0) + 1
    tk.setdefault('lock_history', []).append({
        'at': tk['last_attempt_at'],
        'session': datetime.fromtimestamp(tk['last_attempt_at']).strftime('%H:%M'),
        'error': error, 'n': tk['attempt_count']
    })
    if tk['max_attempts_per_session'] == 0:
        # 0 = 永远不重试, 直接 lock
        tk['status'] = 'TICKER_LOCKED'
        tk['reason'] = error
        if tk['first_locked_at'] == 0:
            tk['first_locked_at'] = tk['last_attempt_at']
    elif tk['attempt_count'] >= tk['max_attempts_per_session']:
        tk['status'] = 'TICKER_LOCKED'
        tk['reason'] = error
        if tk['first_locked_at'] == 0:
            tk['first_locked_at'] = tk['last_attempt_at']
    locks['tickers'][t] = tk
    save_locks(locks)
    return tk['status']

def unlock(ticker):
    locks = load_locks()
    t = ticker.upper()
    if t in locks.get('tickers', {}):
        locks['tickers'][t]['status'] = 'ACTIVE'
        locks['tickers'][t]['attempt_count'] = 0
        locks['tickers'][t]['lock_history'] = []
        save_locks(locks)
        print(f'✅ {t} unlocked (status=ACTIVE, attempt_count=0)')
    else:
        print(f'  {t} not in locks, no-op')

def status():
    locks = load_locks()
    if not locks.get('tickers'):
        print('(no ticker locks)')
        return
    for t, info in sorted(locks['tickers'].items()):
        s = info.get('status', '?')
        ac = info.get('attempt_count', 0)
        reason = info.get('reason', '')[:50]
        print(f'  {t}: {s}  attempts={ac}  reason={reason}')

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'check' and len(sys.argv) >= 3:
        print(check(sys.argv[2]))
    elif cmd == 'record' and len(sys.argv) >= 4:
        print(record(sys.argv[2], sys.argv[3]))
    elif cmd == 'unlock' and len(sys.argv) >= 3:
        unlock(sys.argv[2])
    elif cmd == 'migrate':
        migrate()
    elif cmd == 'status':
        status()
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()
