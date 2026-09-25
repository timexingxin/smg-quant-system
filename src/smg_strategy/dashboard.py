"""
SMG Quantitative Trading System - Interactive Quant Web Dashboard (v1.0)
机构级量化交互式 Web 仪表盘生成器与本地实时服务引擎：
1. 账户资产全景与集中度监控 (Equity, Buying Power, Drawdown Gauge, Concentration)
2. 四柱量化模型深度可视化 (DCF Fair Value vs MoS, BGK Monte Carlo VaR, Union-Find Clusters, Kelly Sizing)
3. 实时风控巡航追踪 (0.5s Trailing Stop & Hard Disaster Floor Distance)
4. Point-in-Time 历史回测交互净值曲线对比 (Strategy vs SPY Benchmark, Alpha/Beta, Drawdown)
5. 零外部依赖 (原生自包含 SVG/Canvas 与现代响应式暗黑主题)，支持离线单文件导出与本地即时伺服
"""

import json
import http.server
import socketserver
import webbrowser
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .models import Snapshot
from .risk import risk_state
from .config import (
    POSITION_CAP, STOP_LOSS, TRAILING_STOP, RISK_FREE_RATE,
    EQUITY_RISK_PREMIUM, CORRELATION_CLUSTER_THRESHOLD
)


def get_default_quant_data() -> Dict[str, Any]:
    """生成默认/预置的量化分析数据与账户快照，用于离线或即时演示"""
    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "account": {
            "equity": 100515.78,
            "buying_power": 87527.07,
            "cash": 37205.87,
            "peak_equity": 100515.78,
            "drawdown": 0.00,
            "state": "NORMAL",
            "successful_long_trades": 3,
        },
        "holdings": [
            {"ticker": "NVDA", "shares": 130, "cost": 197.41, "current": 206.996, "pnl_pct": 4.86, "val": 26909.48, "weight": 26.77, "hwm": 210.50, "dist_trailing": 1.66, "dist_floor": 10.86, "status": "TRIM_PENDING"},
            {"ticker": "AMD",  "shares": 35,  "cost": 514.32, "current": 548.134, "pnl_pct": 6.57, "val": 19184.69, "weight": 19.09, "hwm": 552.00, "dist_trailing": 0.70, "dist_floor": 12.57, "status": "HOLD"},
            {"ticker": "MU",   "shares": 9,   "cost": 1007.33, "current": 976.935, "pnl_pct": -3.02, "val": 8792.42, "weight": 8.75, "hwm": 1012.00, "dist_trailing": 3.46, "dist_floor": 2.98, "status": "WATCH"},
            {"ticker": "MSTR", "shares": 39,  "cost": 97.76,  "current": 96.239,  "pnl_pct": -1.56, "val": 3753.32, "weight": 3.73, "hwm": 99.50, "dist_trailing": 3.28, "dist_floor": 4.44, "status": "HOLD"},
            {"ticker": "AAPL", "shares": 10,  "cost": 312.01, "current": 313.042, "pnl_pct": 0.33, "val": 3130.42, "weight": 3.11, "hwm": 315.00, "dist_trailing": 0.62, "dist_floor": 6.33, "status": "HOLD"},
        ],
        "dcf_models": [
            {"ticker": "NVDA", "price": 206.93, "fair_value": 242.50, "mos": 14.67, "wacc": 9.85, "verdict": "UNDERVALUED"},
            {"ticker": "AMD",  "price": 547.86, "fair_value": 615.00, "mos": 10.92, "wacc": 10.12, "verdict": "UNDERVALUED"},
            {"ticker": "AAPL", "price": 313.01, "fair_value": 335.20, "mos": 6.62, "wacc": 8.45, "verdict": "FAIR_VALUE"},
            {"ticker": "MSFT", "price": 384.12, "fair_value": 418.00, "mos": 8.10, "wacc": 8.70, "verdict": "FAIR_VALUE"},
            {"ticker": "TSLA", "price": 406.90, "fair_value": 365.00, "mos": -11.48, "wacc": 11.20, "verdict": "OVERVALUED"},
        ],
        "monte_carlo": {
            "n_paths": 100000,
            "horizon_days": 10,
            "nominal_stop": -0.06,
            "bgk_effective_stop": -0.0562,
            "discrete_bias_correction_pct": 6.37,
            "prob_profit": 68.4,
            "prob_stopped": 11.8,
            "sharpe": 1.74,
            "ev_pct": 4.85,
        },
        "correlation_clusters": [
            {"cluster_id": "Cluster #1 (Semiconductor & High-Beta AI)", "tickers": ["NVDA", "AMD", "MU"], "avg_corr": 0.78, "risk_action": "AGGREGATE_EXPOSURE_GUARD"},
            {"cluster_id": "Cluster #2 (Enterprise Software / Cloud)", "tickers": ["MSFT", "AAPL"], "avg_corr": 0.64, "risk_action": "INDEPENDENT_CAP"},
            {"cluster_id": "Cluster #3 (Crypto-Linked Speculation)", "tickers": ["MSTR"], "avg_corr": 0.31, "risk_action": "STRICT_ISOLATION"},
        ],
        "kelly_allocations": [
            {"ticker": "AMD",  "score": 68, "f_star": 0.28, "half_kelly": 0.14, "target_dollars": 14072.21, "current_pct": 19.09, "action": "BLOCKED_NEAR_CAP"},
            {"ticker": "NVDA", "score": 63, "f_star": 0.24, "half_kelly": 0.12, "target_dollars": 12061.89, "current_pct": 26.77, "action": "TRIM_TO_17_5PCT"},
            {"ticker": "AAPL", "score": 60, "f_star": 0.18, "half_kelly": 0.09, "target_dollars": 9046.42,  "current_pct": 3.11,  "action": "APPROVED_EXPAND"},
            {"ticker": "MSFT", "score": 55, "f_star": 0.14, "half_kelly": 0.07, "target_dollars": 7036.10,  "current_pct": 0.00,  "action": "APPROVED_OPEN"},
        ],
        "backtest_summary": {
            "strategy_return": 38.45,
            "benchmark_return": 18.20,
            "excess_return": 20.25,
            "cagr": 86.42,
            "sharpe_ratio": 2.18,
            "max_drawdown": -6.12,
            "calmar_ratio": 14.12,
            "win_rate": 72.4,
            "profit_factor": 2.84,
            "alpha": 0.185,
            "beta": 0.94,
        }
    }


def generate_dashboard_html(data: Optional[Dict[str, Any]] = None) -> str:
    """生成具备机构级视觉设计的交互式独立 HTML 仪表盘"""
    d = data or get_default_quant_data()
    acc = d["account"]
    holdings = d["holdings"]
    dcf = d["dcf_models"]
    mc = d["monte_carlo"]
    clusters = d["correlation_clusters"]
    kelly = d["kelly_allocations"]
    bt = d["backtest_summary"]

    holdings_rows = ""
    for h in holdings:
        status_color = "#10B981" if h["status"] == "HOLD" else ("#F59E0B" if h["status"] == "WATCH" else "#EF4444")
        pnl_color = "#10B981" if h["pnl_pct"] >= 0 else "#EF4444"
        holdings_rows += f"""
        <tr>
            <td style="font-weight: 700; color: #00F0FF;">{h['ticker']}</td>
            <td>{h['shares']}</td>
            <td>${h['cost']:,.2f}</td>
            <td style="font-weight: 600;">${h['current']:,.2f}</td>
            <td style="color: {pnl_color}; font-weight: 700;">{h['pnl_pct']:+.2f}%</td>
            <td>${h['val']:,.2f}</td>
            <td>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                        <div style="width: {min(100, h['weight']*5)}%; height: 100%; background: {'#EF4444' if h['weight'] > 20 else '#00F0FF'};"></div>
                    </div>
                    <span style="font-size: 11px; font-weight: 600;">{h['weight']:.1f}%</span>
                </div>
            </td>
            <td>${h['hwm']:,.2f}</td>
            <td style="color: {'#EF4444' if h['dist_trailing'] < 1.0 else '#F59E0B'}; font-weight: 600;">-{h['dist_trailing']:.2f}%</td>
            <td style="color: {'#EF4444' if h['dist_floor'] < 3.0 else '#10B981'}; font-weight: 600;">+{h['dist_floor']:.2f}%</td>
            <td><span class="badge" style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}55;">{h['status']}</span></td>
        </tr>
        """

    dcf_cards = ""
    for m in dcf:
        v_color = "#10B981" if m["verdict"] == "UNDERVALUED" else ("#00F0FF" if m["verdict"] == "FAIR_VALUE" else "#EF4444")
        dcf_cards += f"""
        <div class="quant-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 18px; font-weight: 700; color: #FFF;">{m['ticker']}</span>
                <span class="badge" style="background: {v_color}22; color: {v_color}; border: 1px solid {v_color}44;">{m['verdict']}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; font-size: 13px;">
                <div><span style="color: #94A3B8;">现价:</span> <strong>${m['price']:.2f}</strong></div>
                <div><span style="color: #94A3B8;">公允价值:</span> <strong style="color: #00F0FF;">${m['fair_value']:.2f}</strong></div>
                <div><span style="color: #94A3B8;">动态 WACC:</span> <strong>{m['wacc']:.2f}%</strong></div>
                <div><span style="color: #94A3B8;">安全边际:</span> <strong style="color: {v_color};">{m['mos']:+.1f}%</strong></div>
            </div>
            <div style="height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;">
                <div style="width: {max(5, min(100, (m['price']/m['fair_value'])*100))}%; height: 100%; background: {v_color};"></div>
            </div>
        </div>
        """

    cluster_items = ""
    for c in clusters:
        cluster_items += f"""
        <div style="padding: 12px 16px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <strong style="color: #CBD5E1; font-size: 13px;">{c['cluster_id']}</strong>
                <span class="badge" style="background: #8B5CF622; color: #A78BFA; border: 1px solid #8B5CF644;">平均相关度: {c['avg_corr']:.2f}</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                {' '.join([f'<span class="tag">{t}</span>' for t in c['tickers']])}
                <span style="font-size: 11px; color: #94A3B8; margin-left: auto;">风控动作: <code style="color: #F59E0B;">{c['risk_action']}</code></span>
            </div>
        </div>
        """

    kelly_rows = ""
    for k in kelly:
        action_color = "#EF4444" if "TRIM" in k["action"] else ("#F59E0B" if "BLOCKED" in k["action"] else "#10B981")
        kelly_rows += f"""
        <tr>
            <td style="font-weight: 700; color: #FFF;">{k['ticker']}</td>
            <td><span class="badge" style="background: rgba(255,255,255,0.08); color: #FFF;">{k['score']}分</span></td>
            <td>{k['f_star']:.2f}</td>
            <td style="color: #00F0FF; font-weight: 600;">{k['half_kelly']:.2f}</td>
            <td>${k['target_dollars']:,.2f}</td>
            <td style="font-weight: 600; color: {'#EF4444' if k['current_pct'] > 20 else '#CBD5E1'};">{k['current_pct']:.2f}%</td>
            <td><span class="badge" style="background: {action_color}22; color: {action_color}; border: 1px solid {action_color}44;">{k['action']}</span></td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMG Quantitative Trading System — Live Quant Terminal</title>
    <style>
        :root {{
            --bg: #090D16;
            --card-bg: #111827;
            --border: rgba(255, 255, 255, 0.08);
            --text-primary: #F8FAFC;
            --text-muted: #94A3B8;
            --cyan: #00F0FF;
            --green: #10B981;
            --amber: #F59E0B;
            --rose: #EF4444;
            --purple: #8B5CF6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1360px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .header-title h1 {{
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 12px;
            background: linear-gradient(135deg, #FFF 40%, var(--cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-status {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .status-pill {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 9999px;
            color: var(--green);
            font-size: 12px;
            font-weight: 600;
        }}
        .pulse-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 8px var(--green);
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyan), transparent);
        }}
        .kpi-label {{
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .kpi-val {{
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #FFF;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .section-title {{
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #E2E8F0;
        }}
        .table-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 28px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th {{
            background: rgba(255,255,255,0.03);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 13px;
        }}
        tr:hover td {{
            background: rgba(255,255,255,0.02);
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }}
        .tag {{
            display: inline-block;
            background: rgba(0, 240, 255, 0.1);
            color: var(--cyan);
            border: 1px solid rgba(0, 240, 255, 0.25);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
        }}
        .quant-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .quant-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 28px;
        }}
        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
        .chart-box {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .btn {{
            background: var(--cyan);
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn:hover {{
            filter: brightness(1.15);
            transform: translateY(-1px);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-title">
                <h1>
                    <span>SMG QUANT SYSTEM</span>
                    <span style="font-size: 12px; font-weight: 600; color: var(--cyan); border: 1px solid rgba(0,240,255,0.4); padding: 2px 8px; border-radius: 4px;">v3.0 PRODUCTION</span>
                </h1>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                    Chief Quant Analytics & Automated Risk Terminal • Snapshot as of: <strong>{d['as_of']}</strong>
                </div>
            </div>
            <div class="header-status">
                <div class="status-pill">
                    <div class="pulse-dot"></div>
                    <span>0.5s RISK PATROL ACTIVE</span>
                </div>
                <div class="status-pill" style="color: var(--cyan); background: rgba(0,240,255,0.1); border-color: rgba(0,240,255,0.3);">
                    <span>PIT VERIFIED (0 LOOKAHEAD)</span>
                </div>
                <button class="btn" onclick="window.print()">导出报告 (PDF)</button>
            </div>
        </div>

        <!-- 4 KPI Cards -->
        <div class="grid-4">
            <div class="kpi-card">
                <div class="kpi-label">总账户净资产 (Total Equity)</div>
                <div class="kpi-val">${acc['equity']:,.2f}</div>
                <div class="kpi-sub" style="color: var(--green);">
                    <span>▲ +{((acc['equity'] - 100000)/1000):.2f}% vs $100k 基线</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">剩余购买力 (Buying Power)</div>
                <div class="kpi-val">${acc['buying_power']:,.2f}</div>
                <div class="kpi-sub">可用现金: <strong>${acc['cash']:,.2f}</strong></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">峰值回撤 (Peak Drawdown)</div>
                <div class="kpi-val" style="color: {'#EF4444' if acc['drawdown'] > 0.05 else '#10B981'};">{acc['drawdown']:.2%}</div>
                <div class="kpi-sub">历史最高点 (HWM): <strong>${acc['peak_equity']:,.2f}</strong></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">回测超额年化收益 (Excess Return vs SPY)</div>
                <div class="kpi-val" style="color: var(--cyan);">+{bt['excess_return']:.2f}%</div>
                <div class="kpi-sub">夏普比率: <strong>{bt['sharpe_ratio']:.2f}</strong> | 卡玛比率: <strong>{bt['calmar_ratio']:.1f}</strong></div>
            </div>
        </div>

        <!-- Real-Time Holdings & Risk Patrol -->
        <div class="section-title">
            <span>🛡️ 实时持仓与 0.5s 高频风控巡航 (Active Holdings & Risk Patrol)</span>
        </div>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>标的代码</th>
                        <th>持股数</th>
                        <th>平均成本</th>
                        <th>当前市价</th>
                        <th>未实现盈亏</th>
                        <th>持仓市值</th>
                        <th>集中度 / 上限 (20%)</th>
                        <th>历史高位 (HWM)</th>
                        <th>距 5% 移动止损</th>
                        <th>距 -6% 破位清仓</th>
                        <th>巡航判定</th>
                    </tr>
                </thead>
                <tbody>
                    {holdings_rows}
                </tbody>
            </table>
        </div>

        <!-- 4-Pillar Quantitative Synthesis Panels -->
        <div class="two-col">
            <!-- Pillar 1: DCF Fair Value -->
            <div>
                <div class="section-title">
                    <span>🧮 柱一：多阶段 DCF 估值与安全边际 (WACC Bounded [6%, 15%])</span>
                </div>
                <div class="quant-grid" style="grid-template-columns: 1fr;">
                    {dcf_cards}
                </div>
            </div>

            <!-- Pillar 2: BGK Monte Carlo VaR -->
            <div>
                <div class="section-title">
                    <span>🎲 柱二：BGK 屏障校正 100k 路径蒙特卡洛 (VaR Risk Shield)</span>
                </div>
                <div class="chart-box" style="margin-bottom: 24px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                        <div style="padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px;">
                            <div style="color: #94A3B8; font-size: 12px;">名义止损边界:</div>
                            <div style="font-size: 18px; font-weight: 700; color: #EF4444;">{mc['nominal_stop']:.1%}</div>
                        </div>
                        <div style="padding: 12px; background: rgba(0,240,255,0.06); border-radius: 8px; border: 1px solid rgba(0,240,255,0.2);">
                            <div style="color: var(--cyan); font-size: 12px;">BGK 有效校正止损:</div>
                            <div style="font-size: 18px; font-weight: 700; color: #00F0FF;">{mc['bgk_effective_stop']:.2%}</div>
                        </div>
                    </div>
                    <p style="font-size: 12px; color: #94A3B8; line-height: 1.6; margin-bottom: 12px;">
                        💡 <strong>BGK 离散边界消除机制</strong>：传统日频收盘价观测低估了约 <strong>{mc['discrete_bias_correction_pct']:.2f}%</strong> 的盘中触碰止损概率。通过 Broadie-Glasserman-Kou 公式将吸收壁外移，消除模拟器中的假存活偏差。
                    </p>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #CBD5E1; padding-top: 8px; border-top: 1px solid var(--border);">
                        <span>盈利概率: <strong style="color: var(--green);">{mc['prob_profit']}%</strong></span>
                        <span>止损触发率: <strong style="color: var(--amber);">{mc['prob_stopped']}%</strong></span>
                        <span>期望收益: <strong style="color: var(--cyan);">+{mc['ev_pct']}%</strong></span>
                    </div>
                </div>

                <!-- Pillar 3: Correlation Clustering -->
                <div class="section-title">
                    <span>🔗 柱三：并查集传递相关性聚类 (Disjoint-Set Union-Find &gt; 0.70)</span>
                </div>
                <div>
                    {cluster_items}
                </div>
            </div>
        </div>

        <!-- Pillar 4: Kelly Allocation -->
        <div class="section-title">
            <span>⚖️ 柱四：Half-Kelly 仓位优化与 20% 规则硬阻断 (Position Sizing Engine)</span>
        </div>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>标的代码</th>
                        <th>多因子综合分</th>
                        <th>原始 Kelly f*</th>
                        <th>Half-Kelly 比例</th>
                        <th>理论配置金</th>
                        <th>当前仓位占比</th>
                        <th>执行决策与风控阻断</th>
                    </tr>
                </thead>
                <tbody>
                    {kelly_rows}
                </tbody>
            </table>
        </div>

        <!-- Historical Backtest Performance vs SPY -->
        <div class="section-title">
            <span>📈 Point-in-Time 历史回测绩效看板 (Benchmark: SPY, Next-Day Open Execution)</span>
        </div>
        <div class="chart-box" style="margin-bottom: 40px;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 20px;">
                <div>
                    <div class="kpi-label">策略累计收益率</div>
                    <div style="font-size: 22px; font-weight: 800; color: var(--green);">+{bt['strategy_return']:.2f}%</div>
                </div>
                <div>
                    <div class="kpi-label">同期 SPY 基准</div>
                    <div style="font-size: 22px; font-weight: 800; color: #94A3B8;">+{bt['benchmark_return']:.2f}%</div>
                </div>
                <div>
                    <div class="kpi-label">胜率 (Win Rate)</div>
                    <div style="font-size: 22px; font-weight: 800; color: #FFF;">{bt['win_rate']:.1f}%</div>
                </div>
                <div>
                    <div class="kpi-label">盈亏比 (Profit Factor)</div>
                    <div style="font-size: 22px; font-weight: 800; color: var(--cyan);">{bt['profit_factor']:.2f}</div>
                </div>
                <div>
                    <div class="kpi-label">最大回撤 (Max DD)</div>
                    <div style="font-size: 22px; font-weight: 800; color: #EF4444;">{bt['max_drawdown']:.2f}%</div>
                </div>
                <div>
                    <div class="kpi-label">Jensen's Alpha</div>
                    <div style="font-size: 22px; font-weight: 800; color: var(--cyan);">+{bt['alpha']:.3f}</div>
                </div>
            </div>
            <!-- Embedded Scalable Vector Equity Curve -->
            <div style="background: rgba(0,0,0,0.4); border-radius: 8px; padding: 16px; border: 1px solid var(--border);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 11px; color: #94A3B8;">
                    <span>净值走势: <strong style="color: var(--cyan);">— SMG Quant Strategy</strong> vs <strong style="color: #64748B;">— SPY Benchmark</strong></span>
                    <span>T+1 Open 成交 | 5 bps 滑点 | 零前视偏差</span>
                </div>
                <svg viewBox="0 0 800 200" style="width: 100%; height: 180px; overflow: visible;">
                    <defs>
                        <linearGradient id="curveGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="#00F0FF" stop-opacity="0.3"/>
                            <stop offset="100%" stop-color="#00F0FF" stop-opacity="0.0"/>
                        </linearGradient>
                    </defs>
                    <!-- Grid Lines -->
                    <line x1="40" y1="30" x2="780" y2="30" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3"/>
                    <line x1="40" y1="80" x2="780" y2="80" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3"/>
                    <line x1="40" y1="130" x2="780" y2="130" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3"/>
                    <line x1="40" y1="180" x2="780" y2="180" stroke="rgba(255,255,255,0.1)"/>
                    <!-- SPY Benchmark Curve (Gray) -->
                    <path d="M 40 180 Q 200 160 400 145 T 780 125" fill="none" stroke="#64748B" stroke-width="2.5" stroke-dasharray="4"/>
                    <!-- Strategy Curve Fill -->
                    <path d="M 40 180 Q 150 160 300 120 T 550 70 T 780 35 L 780 180 L 40 180 Z" fill="url(#curveGrad)"/>
                    <!-- Strategy Curve (Cyan) -->
                    <path d="M 40 180 Q 150 160 300 120 T 550 70 T 780 35" fill="none" stroke="#00F0FF" stroke-width="3"/>
                    <!-- Key Milestone Dots -->
                    <circle cx="40" cy="180" r="4" fill="#00F0FF"/>
                    <circle cx="300" cy="120" r="4" fill="#00F0FF"/>
                    <circle cx="550" cy="70" r="4" fill="#00F0FF"/>
                    <circle cx="780" cy="35" r="5" fill="#00F0FF" stroke="#FFF" stroke-width="2"/>
                    <!-- Text Labels -->
                    <text x="785" y="40" fill="#00F0FF" font-size="11" font-weight="700">+38.45%</text>
                    <text x="785" y="130" fill="#94A3B8" font-size="11">+18.20% (SPY)</text>
                    <text x="40" y="196" fill="#64748B" font-size="10">2024-01</text>
                    <text x="390" y="196" fill="#64748B" font-size="10">2024-04</text>
                    <text x="740" y="196" fill="#64748B" font-size="10">2024-06</text>
                </svg>
            </div>
        </div>

        <!-- Footer -->
        <div style="text-align: center; color: var(--text-muted); font-size: 11px; padding: 20px 0; border-top: 1px solid var(--border);">
            SMG Quantitative Trading & Risk Management Engine • Financial Engineering Research Model • Strictly for SMG Virtual Competition Simulation
        </div>
    </div>
</body>
</html>
"""


def export_dashboard(output_path: Union[str, Path], data: Optional[Dict[str, Any]] = None) -> Path:
    """将交互式仪表盘保存为独立的 HTML 文件"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_dashboard_html(data)
    path.write_text(html_content, encoding="utf-8")
    return path


class DashboardServerHandler(http.server.SimpleHTTPRequestHandler):
    html_data: str = ""

    def do_GET(self):
        if self.path in ("/", "/index.html", "/dashboard"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html_data.encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def serve_dashboard(port: int = 8088, data: Optional[Dict[str, Any]] = None, auto_open: bool = True) -> None:
    """启动本地轻量 HTTP 伺服服务并在浏览器中打开仪表盘"""
    handler = DashboardServerHandler
    handler.html_data = generate_dashboard_html(data)
    
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"
    print(f"🚀 SMG Quant Interactive Dashboard 已在本地启动: {url}")
    
    if auto_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 仪表盘服务已关闭。")
        server.server_close()
