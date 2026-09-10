import yfinance as yf
import pandas as pd
import numpy as np

class SMGRules:
    def __init__(self):
        self.local_single_position_cap_enabled = True
        self.local_single_position_cap = 0.20
        self.minimum_shares = 10
        self.minimum_price = 3.0
        self.minimum_market_cap = 25_000_000

class SMGAccount:
    def __init__(self, equity, buying_power):
        self.total_equity = equity
        self.buying_power = buying_power

def rules_checker(ticker, account, rules):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        exchange = info.get('exchange', '')
        if exchange not in ["NMS", "NYQ", "NASDAQ", "NYSE"]:
            return False, f"Not NASDAQ/NYSE (got {exchange})"
            
        market_cap = info.get('marketCap', 0)
        if market_cap < rules.minimum_market_cap:
            return False, f"Market cap {market_cap} below $25M"
            
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        prev_close = info.get('previousClose', 0)
        
        if prev_close < rules.minimum_price or current_price < rules.minimum_price:
            return False, "Price below $3 rule"
            
        return True, "Pass", current_price
    except Exception as e:
        return False, f"Data error: {str(e)}", 0

def score_momentum(data):
    # Max 30 pts
    score = 0
    close = data['Close'].iloc[-1].item()
    close_5d = data['Close'].iloc[-6].item()
    close_20d = data['Close'].iloc[-21].item()
    
    ret_5d = (close / close_5d) - 1
    ret_20d = (close / close_20d) - 1
    
    if ret_5d > 0.08: score += 8
    elif ret_5d > 0.04: score += 5
    elif ret_5d > 0: score += 2
    
    if ret_20d > 0.15: score += 8
    elif ret_20d > 0.08: score += 5
    elif ret_20d > 0: score += 3
    
    # 跑赢 SPY 和突破近期高点 (simplified)
    score += 8 # Assume strong trend for demonstration
    return min(30, score)

def score_volume(data):
    # Max 15 pts
    vol = data['Volume'].iloc[-1].item()
    avg_vol = data['Volume'].iloc[-21:-1].mean().item()
    
    if avg_vol == 0: return 0
    vol_ratio = vol / avg_vol
    
    if vol_ratio >= 2.0: return 15
    if vol_ratio >= 1.5: return 12
    if vol_ratio >= 1.2: return 8
    return 4

def run_smg_algorithm():
    print("Running SMG Algorithm V2.0 Scanner...")
    rules = SMGRules()
    account = SMGAccount(equity=100000, buying_power=150000)
    
    candidates = ["AAPL", "NVDA", "TSLA", "MSFT", "PLTR", "GEV", "AMD"]
    results = []
    
    for ticker in candidates:
        passed, reason, price = rules_checker(ticker, account, rules)
        if not passed:
            print(f"[{ticker}] REJECTED: {reason}")
            continue
            
        data = yf.download(ticker, period="2mo", progress=False)
        if len(data) < 22:
            continue
            
        mom_score = score_momentum(data)
        vol_score = score_volume(data)
        
        # 催化(20), 风险收益(20), 组合(10), 规则(5) -> 简化用定值，或者需 LLM 补全
        catalyst_score = 15 # Placeholder
        risk_score = 15     # Placeholder
        port_score = 8      # Placeholder
        rule_score = 5
        
        total_score = mom_score + vol_score + catalyst_score + risk_score + port_score + rule_score
        
        results.append({
            "ticker": ticker,
            "score": total_score,
            "price": price,
            "momentum": mom_score,
            "volume": vol_score
        })
        
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    print("\n--- FINAL V2.0 CANDIDATE SCORES ---")
    for _, row in df.iterrows():
        print(f"{row['ticker']}: {row['score']} pts (Price: ${row['price']:.2f})")

    # V4.2: 写出 JSON 供 hourly_job.sh 消费
    out = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "ticker": r["ticker"],
                "score": int(r["score"]),
                "price": float(r["price"]),
                "momentum": int(r["momentum"]),
                "volume": int(r["volume"]),
            }
            for r in results
        ],
        "buy_candidates": [r["ticker"] for r in results if r["score"] >= 70],
        "watchlist": [r["ticker"] for r in results if 60 <= r["score"] < 70],
        "sell_candidates": [r["ticker"] for r in results if r["score"] < 45],
    }
    out_path = os.environ.get("SMG_SCAN_RESULTS", "/tmp/smg_scan_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[JSON] {len(results)} results written to {out_path}")

if __name__ == "__main__":
    from datetime import datetime
    import json
    import os
    run_smg_algorithm()
