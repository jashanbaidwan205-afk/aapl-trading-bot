"""
Backtest of the most-active moving-average crossover bot.

Replays the bot's rules (from config.py) day by day on historical daily prices:

  * Stock list: the bot scans Alpaca's "most active by volume" list, which only
    exists for today. Here it is rebuilt for every past day: the S&P 500 stocks
    ranked by the previous day's share volume, top MOST_ACTIVE_TOP kept.
  * Filters: EXCLUDE_SYMBOLS, closes under MIN_PRICE (ETFs aren't in the S&P 500).
  * Signal: 20-day average crosses the 50-day between the last two completed days.
  * Timing: signals come from the previous close; orders fill at today's OPEN
    (the bot runs at 9:30 AM), with SLIPPAGE_PCT against you on each fill.
  * Sizing: each buy = POSITION_SIZE_PCT of the previous close's account value,
    whole shares, at most MAX_POSITIONS stocks, never more than the cash on hand.
    A sell signal sells the whole position. Sells happen before buys.

Limits: uses today's S&P 500 members (companies removed since are missing), daily
bars only, and prices adjusted for splits and dividends.

Usage:  python backtest.py [--years 3] [--slippage 0.05] [--refresh]
Writes results to backtest_results/ (summary.json, equity.json, trades.json).
"""

import argparse
import io
import json
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import config

BASE = Path(__file__).resolve().parent
CACHE = BASE / "backtest_cache"
OUT = BASE / "backtest_results"
START_CASH = 100_000.0
BENCHMARK = "SPY"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def sp500_symbols(refresh: bool) -> list[str]:
    path = CACHE / "sp500.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0 (backtest script)"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    syms = sorted({str(s).replace(".", "-") for s in table["Symbol"]})  # BRK.B -> BRK-B for Yahoo
    path.write_text(json.dumps(syms))
    return syms


def download(symbols: list[str], years: int, refresh: bool) -> dict[str, pd.DataFrame]:
    """Daily Open/Close/Volume per field, one column per symbol."""
    path = CACHE / f"prices_{years}y.pkl"
    if not refresh:
        # Reuse any cached download that covers at least this many years.
        for n in range(years, 11):
            p = CACHE / f"prices_{n}y.pkl"
            if p.exists():
                return pickle.loads(p.read_bytes())
    period = f"{years + 1}y"  # extra year so the 50-day average is ready on day one
    frames = []
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                                 group_by="ticker", threads=True, progress=False)
                if not df.empty:
                    frames.append(df)
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"  chunk {i // 100 + 1} attempt {attempt + 1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
        print(f"  downloaded {min(i + 100, len(symbols))}/{len(symbols)}")
    raw = pd.concat(frames, axis=1)
    data = {f: raw.xs(f, axis=1, level=1).sort_index() for f in ("Open", "Close", "Volume")}
    data = {f: d.loc[:, ~d.columns.duplicated()] for f, d in data.items()}
    path.write_bytes(pickle.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run(years: int, slippage_pct: float, refresh: bool, withdraw_pct: float = 0.0) -> dict:
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    symbols = sp500_symbols(refresh)
    print(f"S&P 500 list: {len(symbols)} stocks")
    data = download(symbols + [BENCHMARK], years, refresh)
    opens, closes, volumes = data["Open"], data["Close"], data["Volume"]

    sma_s = closes.rolling(config.SHORT_MA_PERIOD, min_periods=config.SHORT_MA_PERIOD).mean()
    sma_l = closes.rolling(config.LONG_MA_PERIOD, min_periods=config.LONG_MA_PERIOD).mean()
    diff = sma_s - sma_l
    buy_sig = (diff.shift(1) <= 0) & (diff > 0)    # crossed above on this day's close
    sell_sig = (diff.shift(1) >= 0) & (diff < 0)   # crossed below on this day's close

    days = closes.index
    start_i = max(config.LONG_MA_PERIOD + 2, len(days) - years * 252)
    excluded = {s.upper() for s in config.EXCLUDE_SYMBOLS} | {BENCHMARK}
    slip = slippage_pct / 100

    cash = START_CASH
    pos: dict[str, dict] = {}       # symbol -> {qty, price, date}
    trades, equity_rows, months = [], [], []
    signals_seen = skipped_full = skipped_cash = skipped_cap = 0
    cap_pct = getattr(config, "MAX_INVESTED_PCT", 100.0)
    withdrawn_total = 0.0
    month_pnl = 0.0

    def value_at(i: int) -> float:
        return cash + sum(p["qty"] * (closes.iloc[i][s] if pd.notna(closes.iloc[i][s]) else p["price"])
                          for s, p in pos.items())

    for i in range(start_i, len(days)):
        prev = i - 1
        day = days[i]
        o = opens.iloc[i]

        # Today's scan list = top N by previous day's volume, plus anything held
        vol = volumes.iloc[prev].drop(labels=[c for c in excluded if c in volumes.columns], errors="ignore")
        active = list(vol.dropna().sort_values(ascending=False).index[:config.MOST_ACTIVE_TOP])
        rank = {s: r for r, s in enumerate(active, start=1)}
        scan = list(dict.fromkeys(active + list(pos)))

        # Sells first (fill at today's open)
        for s in scan:
            if s in pos and bool(sell_sig.iloc[prev].get(s, False)) and pd.notna(o.get(s)):
                p = pos.pop(s)
                fill = o[s] * (1 - slip)
                cash += p["qty"] * fill
                pnl = (fill - p["price"]) * p["qty"]
                month_pnl += pnl
                trades.append({"symbol": s, "bought": p["date"], "sold": day.strftime("%Y-%m-%d"),
                               "shares": p["qty"], "buy_price": round(p["price"], 2), "sell_price": round(fill, 2),
                               "pnl": round(pnl, 2), "return_pct": round((fill / p["price"] - 1) * 100, 2),
                               "days_held": int((day - pd.Timestamp(p["date"])).days)})

        # Buys, most active first
        equity_prev = value_at(prev)
        budget = equity_prev * config.POSITION_SIZE_PCT / 100
        invested = sum(p["qty"] * (closes.iloc[prev][s] if pd.notna(closes.iloc[prev][s]) else p["price"])
                       for s, p in pos.items())
        buys = [s for s in active if bool(buy_sig.iloc[prev].get(s, False))]
        for s in sorted(buys, key=lambda x: rank[x]):
            signals_seen += 1
            c_prev = closes.iloc[prev][s]
            if s in pos or pd.isna(c_prev) or c_prev < config.MIN_PRICE or pd.isna(o.get(s)):
                continue
            if len(pos) >= config.MAX_POSITIONS:
                skipped_full += 1
                continue
            qty = math.floor(budget / c_prev)
            fill = o[s] * (1 + slip)
            if qty < 1:
                continue
            if qty * fill > cash:
                skipped_cash += 1
                continue
            if invested + qty * c_prev > equity_prev * cap_pct / 100:
                skipped_cap += 1
                continue
            invested += qty * c_prev
            cash -= qty * fill
            pos[s] = {"qty": qty, "price": fill, "date": day.strftime("%Y-%m-%d")}

        # Month end: withdraw a share of the month's net profit from closed trades
        month_end = i == len(days) - 1 or days[i + 1].month != day.month
        if month_end:
            take = min(cash, max(0.0, month_pnl) * withdraw_pct / 100)
            cash -= take
            withdrawn_total += take
            months.append({"month": day.strftime("%Y-%m"), "realized_pl": round(month_pnl, 2),
                           "withdrawn": round(take, 2), "withdrawn_total": round(withdrawn_total, 2),
                           "account_value": round(value_at(i), 2)})
            month_pnl = 0.0

        equity_rows.append({"date": day.strftime("%Y-%m-%d"), "equity": round(value_at(i), 2),
                            "positions": len(pos), "cash": round(cash, 2),
                            "withdrawn_total": round(withdrawn_total, 2)})

    # Benchmark: buy and hold SPY from the first day's open
    spy = closes[BENCHMARK].iloc[start_i:]
    spy_shares = START_CASH / opens[BENCHMARK].iloc[start_i]
    for row, v in zip(equity_rows, spy.values):
        row["spy_equity"] = round(float(spy_shares * v), 2)

    eq = pd.Series([r["equity"] for r in equity_rows], index=pd.to_datetime([r["date"] for r in equity_rows]))
    spy_eq = pd.Series([r["spy_equity"] for r in equity_rows], index=eq.index)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25

    def max_dd(s: pd.Series) -> float:
        return float((s / s.cummax() - 1).min() * 100)

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    open_value = sum(p["qty"] * closes.iloc[-1][s] for s, p in pos.items())
    summary = {
        "start": equity_rows[0]["date"], "end": equity_rows[-1]["date"], "years": round(yrs, 2),
        "start_value": START_CASH, "end_value": round(float(eq.iloc[-1]), 2),
        "total_return_pct": round((eq.iloc[-1] / START_CASH - 1) * 100, 2),
        "annual_return_pct": round(((eq.iloc[-1] / START_CASH) ** (1 / yrs) - 1) * 100, 2),
        "max_drawdown_pct": round(max_dd(eq), 2),
        "spy_end_value": round(float(spy_eq.iloc[-1]), 2),
        "spy_total_return_pct": round((spy_eq.iloc[-1] / START_CASH - 1) * 100, 2),
        "spy_annual_return_pct": round(((spy_eq.iloc[-1] / START_CASH) ** (1 / yrs) - 1) * 100, 2),
        "spy_max_drawdown_pct": round(max_dd(spy_eq), 2),
        "closed_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else None,
        "realized_pl": round(sum(t["pnl"] for t in trades), 2),
        "avg_days_held": round(sum(t["days_held"] for t in trades) / len(trades), 1) if trades else None,
        "avg_positions": round(sum(r["positions"] for r in equity_rows) / len(equity_rows), 2),
        "avg_invested_pct": round(sum(1 - r["cash"] / r["equity"] for r in equity_rows) / len(equity_rows) * 100, 1),
        "open_positions_at_end": len(pos), "open_value_at_end": round(float(open_value), 2),
        "buy_signals": signals_seen, "skipped_full": skipped_full, "skipped_cash": skipped_cash,
        "skipped_cap": skipped_cap, "max_invested_pct": cap_pct,
        "withdraw_pct": withdraw_pct, "total_withdrawn": round(withdrawn_total, 2),
        "end_value_plus_withdrawn": round(float(eq.iloc[-1]) + withdrawn_total, 2),
        "total_gain_incl_withdrawn": round(float(eq.iloc[-1]) + withdrawn_total - START_CASH, 2),
        "total_return_incl_withdrawn_pct": round((float(eq.iloc[-1]) + withdrawn_total) / START_CASH * 100 - 100, 2),
        "slippage_pct": slippage_pct, "universe": f"S&P 500 ({len(symbols)} stocks today)",
        "settings": f"top {config.MOST_ACTIVE_TOP} by volume, min ${config.MIN_PRICE:g}, "
                    f"SMA{config.SHORT_MA_PERIOD}/{config.LONG_MA_PERIOD}, {config.POSITION_SIZE_PCT:g}% per buy, "
                    f"max {config.MAX_POSITIONS} stocks",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    trades.sort(key=lambda t: t["sold"], reverse=True)
    for i, t in enumerate(sorted(trades, key=lambda t: (t["sold"], t["symbol"])), start=1):
        t["trade"] = i
    meta = [
        {"label": "backtest.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"},
        {"label": "config.py", "value": (BASE / "config.py").read_text(encoding="utf-8"), "type": "code", "lang": "python"},
        {"label": "Input", "value": f"Yahoo Finance daily prices (split/dividend adjusted) for {summary['universe']} and SPY"},
    ]
    note = [{"label": "Made by", "value": "backtest.py (code is attached to the Backtest summary source)"}]
    (OUT / "summary.json").write_text(json.dumps({"rows": [summary], "meta": meta}, indent=1))
    (OUT / "equity.json").write_text(json.dumps({"rows": equity_rows, "meta": note}, separators=(",", ":")))
    (OUT / "trades.json").write_text(json.dumps({"rows": trades, "meta": note}, separators=(",", ":")))
    (OUT / "months.json").write_text(json.dumps({"rows": months, "meta": note}, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the most-active crossover bot")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--slippage", type=float, default=0.05, help="percent against you on each fill")
    ap.add_argument("--refresh", action="store_true", help="re-download the S&P 500 list and prices")
    ap.add_argument("--size", type=float, help="override POSITION_SIZE_PCT from config.py")
    ap.add_argument("--max", type=int, help="override MAX_POSITIONS from config.py")
    ap.add_argument("--out", help="folder for results (default backtest_results)")
    ap.add_argument("--cap", type=float, help="override MAX_INVESTED_PCT from config.py")
    ap.add_argument("--withdraw", type=float, default=0.0,
                    help="each month end, withdraw this %% of the month's net profit from closed trades")
    a = ap.parse_args()
    global OUT
    if a.size is not None:
        config.POSITION_SIZE_PCT = a.size
    if a.max is not None:
        config.MAX_POSITIONS = a.max
    if a.cap is not None:
        config.MAX_INVESTED_PCT = a.cap
    if a.out:
        OUT = BASE / a.out
    s = run(a.years, a.slippage, a.refresh, a.withdraw)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
