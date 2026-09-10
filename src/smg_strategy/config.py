"""
SMG Quantitative Trading System - Global Configuration & Parameter Governance
统一参数治理中心：集中维护所有交易、风控、合规与量化模型参数，杜绝多处硬编码与阈值散落。
"""
from typing import Set

# 1. 官方规则与合规准入 (SMG Official Rules)
MIN_STOCK_PRICE: float = 3.0            # 初始开仓股价不得低于 $3.00
MIN_MARKET_CAP: float = 25_000_000.0    # 最低市值不得低于 $2500 万
MIN_ORDER_SHARES: int = 10              # 委托股数不得少于 10 股
ALLOWED_EXCHANGES: Set[str] = {"NYSE", "NASDAQ", "NMS", "NYQ"}

# 2. 集中度与仓位管理 (Position & Concentration Limits)
POSITION_CAP: float = 0.20              # 单个标的最大持仓上限：20% 账户总权益
TARGET_CAP_REDUCE: float = 0.175        # 集中度超标时平滑减仓目标：17.5% 安全带
TARGET_CAP_REDUCE_RATIO: float = TARGET_CAP_REDUCE  # 别名兼容
SECTOR_CAP: float = 0.35                # 单一行业最大暴露上限：35%
MIN_UNUSED_BUYING_POWER: float = 0.15   # 预留可用买入力缓冲：至少 15%

# 3. 止损与风控参数 (Risk Defense & Stop Loss)
STOP_LOSS: float = -0.06                # GAES 硬止损线：成本价 -6.0% 强制斩仓
TRAILING_STOP: float = 0.05             # 动态移动止损：距历史最高点回撤 >= 5.0% 触发平仓
MAX_KELLY_FRACTION: float = 0.40        # 凯利公式仓位建议上限（Half-Kelly 封顶）

# 回撤状态机阶段阈值 (Drawdown State Machine)
DRAWDOWN_CAUTION: float = 0.05          # 5% 回撤预警 -> 降单笔风险至 0.75%，毛敞口 90%
DRAWDOWN_DEFENSE: float = 0.075         # 7.5% 回撤防御 -> 降单笔风险至 0.5%，毛敞口 50%
DRAWDOWN_STOP: float = 0.09             # 9% 绝对熔断 -> 封死新开仓，留 1% 缓冲守住 10% 底线

# 4. 选股评分与入选门槛 (Scoring & Entry Thresholds)
MIN_BUY_SCORE: int = 72                 # 标准入选门槛评分 (0-100)
SHORT_BUY_SCORE: int = 78               # 做空要求更强确认分
CORRELATION_CLUSTER_THRESHOLD: float = 0.70  # 相关系数聚类隔离阈值

# 5. 宏观与金融工程参数 (Macro & Financial Engineering)
RISK_FREE_RATE: float = 0.0475          # 无风险利率 Rf (当前约为 4.75%)
EQUITY_RISK_PREMIUM: float = 0.055      # 股权风险溢价 ERP
DEFAULT_TERMINAL_GROWTH: float = 0.03   # DCF 永续终值增长率
TRADING_DAYS_YEAR: int = 252            # 年交易日标准基数

# 6. 盘中高频守护与限流参数 (Patrol & Rate Limiting)
MOMENTUM_START_TIME: str = "07:15"      # 避开早盘假突破的时间锁 (美西 07:15 PT)
MAX_DAILY_BUYS: int = 2                 # 单交易日最大开仓次数熔断锁
CACHE_TTL_SECONDS: int = 30             # 本地行情接口缓存时长（防止高频封禁）\n