"""
Settings for the moving-average crossover bot.
Edit these values to change what the bot trades and how.
"""

# --- What to trade -----------------------------------------------------------
SYMBOL = "AAPL"          # Stock ticker (Apple)
TRADE_QTY = 1            # Shares per buy/sell order

# --- Moving averages ---------------------------------------------------------
SHORT_MA_PERIOD = 20     # Fast moving average (days)
LONG_MA_PERIOD = 50      # Slow moving average (days)

# --- Data --------------------------------------------------------------------
LOOKBACK_DAYS = 100      # Trading days of daily price history pulled from Yahoo Finance

# --- Behaviour ---------------------------------------------------------------
# True  = log what the bot WOULD do, but never send an order.
# False = actually place orders in your Alpaca paper account.
DRY_RUN = False

# If False, a sell signal only sells shares you already hold (no short selling).
ALLOW_SHORT = False

# If False, a buy signal is skipped when you already hold the stock.
ALLOW_ADD_TO_POSITION = False

# Files the bot writes next to this script
LOG_FILE = "trading_bot.log"
STATE_FILE = "bot_state.json"   # remembers the last crossover it acted on
