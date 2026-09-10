#!/usr/bin/env python3
"""
SMG Order State Machine & Pre-Trade Risk Engine (v7.8)
遵从金融级工程规范：
1. 9 态状态机：CREATED, SUBMITTING, ACKNOWLEDGED, PARTIALLY_FILLED, FILLED, CANCEL_PENDING, CANCELLED, REJECTED, UNKNOWN
2. UNKNOWN 状态绝不自动补单，强行挂起并触发告警
3. 20% 集中度超限：仅精准减仓至 17.5% 安全带，严禁暴力清仓
4. Pre-trade Risk Gate 确定性校验：代码层硬性拦截 Agent 越权指令
"""

import enum
import json
import time

class OrderState(enum.Enum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

class OrderStateMachine:
    def __init__(self, client_order_id, ticker, action, target_shares):
        self.client_order_id = client_order_id
        self.ticker = ticker
        self.action = action
        self.target_shares = target_shares
        self.filled_shares = 0
        self.state = OrderState.CREATED
        self.history = [(self.state.value, time.time())]

    def transition_to(self, new_state: OrderState, reason=""):
        # UNKNOWN 状态下强行隔离，禁止任意非人工解锁操作
        if self.state == OrderState.UNKNOWN and new_state != OrderState.UNKNOWN:
            raise RuntimeError(f"🚨 ORDER LOCKED IN UNKNOWN STATE: client_order_id={self.client_order_id}. Automatic transition prohibited!")
        
        self.state = new_state
        self.history.append((new_state.value, time.time()))
        print(f"[OrderStateMachine] Order {self.client_order_id} ({self.ticker} {self.action}) -> {new_state.value} ({reason})")

def calculate_concentration_reduction(ticker, current_shares, current_price, total_equity, target_ratio=0.175):
    """
    精准计算集中度控仓卖出股数
    当市值占比 > 20% 时，计算降低至目标 17.5% 所需卖出的最小股数，绝不盲目全额清仓。
    """
    pos_value = current_shares * current_price
    conc_ratio = pos_value / total_equity
    if conc_ratio <= 0.20:
        return 0  # 未突破硬红线
    
    target_value = total_equity * target_ratio
    excess_value = pos_value - target_value
    shares_to_sell = int(excess_value / current_price) + 1
    shares_to_sell = min(current_shares, max(1, shares_to_sell))
    return shares_to_sell

def enforce_agent_schema_gate(agent_role, output_payload):
    """代码层硬性 Schema 拦截，防止 Agent 软约束失效"""
    if agent_role == "risk_officer":
        # 首席风控官绝不包含 BUY 指令
        text = str(output_payload).upper()
        if "ACTION: BUY" in text or "SUGGEST: BUY" in text:
            raise ValueError("🚨 Pre-Trade Risk Gate Violation: Risk officer output contains prohibited BUY instruction!")
    return True

if __name__ == "__main__":
    # 逻辑验证单元测试
    osm = OrderStateMachine("ORD-TEST-001", "NVDA", "BUY", 50)
    osm.transition_to(OrderState.SUBMITTING, "Sending request to SMG")
    osm.transition_to(OrderState.ACKNOWLEDGED, "Broker acknowledged")
    osm.transition_to(OrderState.FILLED, "Executed fully")
    
    # 测试集中度精准减仓计算
    # 假设 AAPL 持仓占比 22%，总资产 $100,000，单价 $200，持仓 110 股
    to_sell = calculate_concentration_reduction("AAPL", 110, 200, 100000, target_ratio=0.175)
    print(f"集中度 22% -> 精准减仓股数: {to_sell} 股 (剩余持仓占比预估 17.5%)")
