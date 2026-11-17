"""
Settings for the moving-average crossover bot.
Edit these values to change what the bot trades and how.
"""

# --- Which stocks to scan -----------------------------------------------------
# Each morning the bot pulls Alpaca's list of the most active stocks (by share
# volume) and checks each one for a 20/50-day crossover. Stocks it already
# holds are always checked too, so it can sell them even after they stop trending.
MOST_ACTIVE_TOP = 50          # How many of the most active stocks to scan (max 100)
MIN_PRICE = 10.0              # Skip stocks that closed below this price ($)
ALLOW_ETFS = False            # False = only regular company stocks (no ETFs/funds)

# Never trade these (leveraged/inverse ETFs and anything else you want to avoid)
EXCLUDE_SYMBOLS = [
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS", "SPXU", "UPRO", "TNA", "TZA",
    "LABU", "LABD", "NVDL", "NVD", "TSLL", "TSLQ", "UVXY", "SVXY", "SPDN", "RWM",
]

# --- Moving averages ----------------------------------------------------------
SHORT_MA_PERIOD = 20          # Fast moving average (days)
LONG_MA_PERIOD = 50           # Slow moving average (days)
LOOKBACK_DAYS = 100           # Trading days of daily price history used

# --- Position sizing ------------------------------------------------------------
POSITION_SIZE_PCT = 5.0       # Each buy spends this % of total account value (whole shares)
MAX_POSITIONS = 10            # Most different stocks held at once
MAX_INVESTED_PCT = 50.0       # Never let stocks held be worth more than this % of the account

# --- Behaviour ----------------------------------------------------------------
# True  = log what the bot WOULD do, but never send an order.
# False = actually place orders in your Alpaca paper account.
DRY_RUN = False

# A sell signal sells the whole position in that stock. With ALLOW_SHORT = False
# a sell signal for a stock you don't own does nothing (no short selling).
ALLOW_SHORT = False

# Files the bot writes next to this script
LOG_FILE = "trading_bot.log"
STATE_FILE = "bot_state.json"       # remembers the last crossover acted on per stock
SCAN_FILE = "scan_results.json"     # what the latest run saw for every stock it scanned

# =============================================================================
# Stocks in Play day trader (sip_bot.py) - a second bot in the same paper account
# =============================================================================
# Each market day at 9:35 it ranks S&P 500 stocks by opening relative volume
# (first 5-minute volume vs its 14-day average), keeps the top SIP_TOP, and trades
# a break of the first 5-minute candle in that candle's direction, with a stop
# SIP_STOP_ATR x the 14-day ATR away. Everything is closed by SIP_FLATTEN_AT.
# It never trades a stock the crossover bot holds.
SIP_ENABLED = True            # False = the scheduled day trader does nothing
SIP_TOP = 20                  # Stocks in play to consider each day
SIP_MIN_RVOL = 1.0            # Opening volume at least this many times normal
SIP_MIN_AVG_VOLUME = 1_000_000  # 14-day average daily volume (shares)
SIP_MIN_ATR = 0.50            # 14-day average daily range ($)
SIP_STOP_ATR = 0.10           # Stop distance as a fraction of the 14-day ATR
SIP_RISK_PCT = 0.5            # Max loss at the stop, % of the account, per trade
SIP_MAX_POS_PCT = 10.0        # Largest single position, % of the account
SIP_MAX_INVESTED_PCT = 100.0  # All open day-trade positions, % of the account (100 = the whole account, no borrowed money)
SIP_ALLOW_SHORT = True        # Trade down-candle breakdowns as short sales
SIP_ENTRY_UNTIL = "15:30"     # No new entries after this time (New York)
SIP_DAY_JOB_UNTIL = "15:00"   # The morning GitHub job hands over to the afternoon job (6-hour job limit)
SIP_FLATTEN_AT = "15:55"      # Close every day-trade position at this time
SIP_DRY_RUN = False           # True = log the plan, place no orders
SIP_LOG_FILE = "sip_bot.log"
SIP_PLAN_FILE = "sip_plan.json"   # today's picks and orders, for the dashboard
