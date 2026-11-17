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
POSITION_SIZE_PCT = 2.0       # Each buy spends this % of total account value (whole shares)
MAX_POSITIONS = 10            # Most different stocks held at once

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
