"""
Backtest of the day-trader-skill plugin (github.com/Jakeschincariol/day-trader-skill, commit 3d0f4ef)
with a $100 account, over the last year of 5-minute bars.

It calls the plugin's own code, copied unchanged into vendor/day_trader_skill/:
  * trader-scanner  scan.usable() + scan.orb_hit(): a stock counts only if every 5-minute bar of the
                    day is there; the setup is the FIRST 5-minute bar that CLOSES above the 15-minute
                    opening-range high (long only). Scanner default floor: price >= $5.
  * trader-exit     exit.compute(): stop at the opening-range midpoint (orb default), target 2R
                    (default); refuses a stop closer than 0.05% of the price.
  * trader-budget   size.size_trade(): whole shares, the smallest of: $50 max loss per trade,
                    75% position cap, $4,000 max order, cash after a 10% buffer; refuses
                    reward:risk under 1 and orders under $5. Budget $100 (everything else default).
  * trader-risk     its day-trade rules, as written in check.py: entries only 9:45-15:30, at most
                    3 entries a day, the max losses at the stops of today's entries <= $150.
  * trader-broker   a bracket order (stop-loss + take-profit) and flat by 15:55.

What the plugin leaves to you, and how this backtest decides it:
  * Which stock: the scanner says "never pick for them". Here every breakout is taken in the
    scanner's own order (earliest breakout first, then symbol) while the limits allow.
  * Universe: S&P 500 stocks (the plugin can scan ~5,900 US stocks from Yahoo; Yahoo keeps only
    60 days of 5-minute bars, so a year needs Alpaca's bars, which we have for the S&P 500).
  * Fill: the market order fills at the next 5-minute bar's open, +0.05% slippage; stop and target
    fill at their price (or the open if a bar gaps past); inside a bar, prices go
    open-low-high-close on a rising bar and open-high-low-close on a falling bar.
  * Cash: the config budget stays $100; cash moves with every trade (as trader-broker's
    portfolio.json would), and sizing uses the lower of the two.

Usage: python backtest_plugin.py [--years 1] [--budget 100]
Writes backtest_plugin_results/plugin_{summary,equity,trades,months}.json
"""

import argparse
import datetime as dt
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

import backtest_daytrade as bd
import backtest_orb as orb

BASE = Path(__file__).resolve().parent
PLUG = BASE / "vendor" / "day_trader_skill"
OUT = BASE / "backtest_plugin_results"
SLIP = 0.05 / 100


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PLUG / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scan = _load("dts_scan", "trader-scanner/scan.py")
exitp = _load("dts_exit", "trader-exit/exit.py")
budget_mod = _load("dts_size", "trader-budget/size.py")


def scanner_bars(b, day):
    """DayBars -> the scanner's bar dicts (naive New York datetimes)."""
    d = dt.date.fromisoformat(day)
    out = []
    for i in range(len(b)):
        h, m = map(int, str(b.hm[i]).split(":"))
        out.append({"start": dt.datetime.combine(d, dt.time(h, m)), "c": float(b.close[i]),
                    "h": float(b.high[i]), "l": float(b.low[i]), "v": float(b.volume[i])})
    return out


def exit_bracket(b, k, stop, target, entry_px):
    """Bracket from bar k: stop or target (bar path), else the 15:55 flatten at that bar's open."""
    cut = next((i for i in range(len(b)) if b.hm[i] >= "15:55"), len(b))
    if k >= cut:
        return None
    hm_, px, why = bd.exit_from(bd._View(b, cut), k, "long", stop, target, entry_px)
    if why == "close" and cut < len(b):
        px, hm_ = b.open[cut] * (1 - SLIP), b.hm[cut]
    return str(hm_), float(px), why


def run(years: int, budget: float, max_trades: int = 3):
    cfg = dict(budget_mod.DEFAULTS)
    cfg["budget"] = budget
    cfg["max_trades_per_day"] = max_trades
    cfg["max_orders_per_day"] = max(cfg["max_orders_per_day"], 3 * max_trades)
    symbols = orb.sp500_symbols()
    daily = orb.daily_data(symbols + ["SPY"], years)
    closes, opens = daily["Close"], daily["Open"]
    today = pd.Timestamp.now(tz=orb.NY).strftime("%Y-%m-%d")
    window = [d for d in closes.index if d.strftime("%Y-%m-%d") < today][-252 * years:]
    days = [d.strftime("%Y-%m-%d") for d in window]
    syms = [s for s in symbols if s in closes.columns]
    start = pd.Timestamp(days[0], tz=orb.NY).to_pydatetime()
    end = (pd.Timestamp(days[-1], tz=orb.NY) + pd.Timedelta(hours=23)).to_pydatetime()
    bars_by = orb.day_bars(orb.minute_bars(syms, start, end))
    spy_shares = budget / float(opens["SPY"].loc[window[0]])

    n_range = 15 // scan.BAR_MIN
    before, after = cfg["no_entries_before"], cfg["no_entries_after"]
    cash = budget
    trades, equity_rows, months, skipped = [], [], [], {}
    month_pnl = 0.0

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for di, day in enumerate(days):
        d = dt.date.fromisoformat(day)
        cutoff = dt.datetime.combine(d, dt.time(16, 0))
        # trader-scanner: every usable chart's first close above the 15-minute range high
        hits = []
        for s in syms:
            b = bars_by.get((s, day))
            if b is None or len(b) < n_range + 1:
                continue
            sb = scanner_bars(b, day)
            if not scan.usable(sb, d, cutoff, n_range):
                continue
            h = scan.orb_hit(sb, n_range)
            if not h:
                continue
            if h["price"] < 5:                     # scanner default --min-price 5
                continue
            hits.append((h["_t"], s, h, b))
        hits.sort(key=lambda x: (x[0], x[1]))     # the scanner's order

        open_pos, entries, loss_today, day_pnl = [], 0, 0.0, 0.0
        for t_close, s, h, b in hits:
            at = t_close.strftime("%H:%M")
            if at < before or at > after:           # trader-risk entry window
                skip("outside the 9:45-3:30 entry window")
                continue
            if entries >= cfg["max_trades_per_day"]:
                skip("already %d trades that day" % cfg["max_trades_per_day"])
                continue
            # positions that closed before this one triggers give their cash back
            for p in [p for p in open_pos if p["exit_close_at"] <= at]:
                open_pos.remove(p)
            k = h["bar_index"] + 1                  # the market order fills on the next bar
            if k >= len(b):
                continue
            # trader-exit: or-mid stop, 2R target
            try:
                plan = exitp.compute(Decimal(str(h["price"])), "or_mid", None, "r", Decimal("2"),
                                     or_high=h["or_high"], or_low=h["or_low"])
            except exitp.PlanError:
                skip("stop too tight (exit planner refused)")
                continue
            fresh = float(b.open[k])
            in_use = sum(p["cost"] for p in open_pos)
            # trader-budget: whole shares within every limit
            try:
                sz = budget_mod.size_trade({"symbol": s, "entry": float(plan["entry"]), "stop": float(plan["stop"]),
                                            "target": float(plan["target"]), "date": day, "setup": "orb"},
                                           cfg, budget=budget, price=fresh, cash=cash - in_use)
            except budget_mod.SizeError:
                skip("price already past the stop or target at the fill")
                continue
            if sz["refusal"] or sz["qty"] < 1:
                why = sz["refusal"] or ""
                skip("can't afford one share" if "costs" in why else
                     "reward:risk under 1 (price already moved)" if "reward:risk" in why else
                     "target already passed" if "already happened" in why else "budget manager refused")
                continue
            # trader-risk: today's max losses at the stops
            if loss_today + sz["max_loss"] > cfg["max_daily_loss_usd"] + 0.005:
                skip("daily loss limit")
                continue
            qty = int(sz["qty"])
            entry = fresh * (1 + SLIP)
            ex = exit_bracket(b, k, float(sz["stop"]), float(sz["target"]), fresh)
            if not ex:
                continue
            xhm, xpx, why = ex
            if why == "close":
                pass
            elif why == "stop":
                xpx = xpx  # exit_from already applied slippage on the way out
            pnl = (xpx - entry) * qty
            cost = qty * entry
            trades.append({"day": day, "symbol": s, "signal_at": at, "entry_hm": str(b.hm[k]), "exit_hm": xhm,
                           "shares": qty, "entry": round(entry, 2), "stop": float(sz["stop"]),
                           "target": float(sz["target"]), "exit": round(xpx, 2), "exit_reason": why,
                           "pnl": round(pnl, 4), "cost": round(cost, 2), "bound_by": sz["bound_by"],
                           "cash_before": round(cash, 2)})
            open_pos.append({"cost": cost, "exit_close_at": xhm})
            entries += 1
            loss_today += sz["max_loss"]
            day_pnl += pnl
        cash += day_pnl
        month_pnl += day_pnl
        if di == len(days) - 1 or days[di + 1][:7] != day[:7]:
            months.append({"month": day[:7], "realized_pl": round(month_pnl, 2), "account_value": round(cash, 2)})
            month_pnl = 0.0
        equity_rows.append({"date": day, "equity": round(cash, 4), "trades": entries,
                            "spy_equity": round(spy_shares * float(closes["SPY"].loc[pd.Timestamp(day)]), 4)})

    eq = pd.Series([r["equity"] for r in equity_rows])
    spy = pd.Series([r["spy_equity"] for r in equity_rows])
    dd = lambda x: round(float((x / x.cummax() - 1).min() * 100), 2)  # noqa: E731
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    summary = {
        "start": days[0], "end": days[-1], "trading_days": len(days), "start_value": budget,
        "end_value": round(cash, 2), "total_return_pct": round((cash / budget - 1) * 100, 2),
        "max_drawdown_pct": dd(eq), "spy_end_value": round(float(spy.iloc[-1]), 2),
        "spy_total_return_pct": round((float(spy.iloc[-1]) / budget - 1) * 100, 2), "spy_max_drawdown_pct": dd(spy),
        "trades": len(trades), "days_with_trades": len({t["day"] for t in trades}),
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win": round(gw / len(wins), 2) if wins else None,
        "avg_loss": round(-gl / len(losses), 2) if losses else None,
        "profit_factor": round(gw / gl, 2) if gl else None,
        "exits": {r: sum(t["exit_reason"] == r for t in trades) for r in ("stop", "target", "close")},
        "avg_shares": round(sum(t["shares"] for t in trades) / len(trades), 2) if trades else None,
        "avg_position_usd": round(sum(t["cost"] for t in trades) / len(trades), 2) if trades else None,
        "skipped": skipped, "months_positive": sum(m["realized_pl"] > 0 for m in months), "months": len(months),
        "settings": "day-trader-skill defaults with budget $%g: 15-min opening range, close above the high, "
                    "or-mid stop, 2R target, $50 max loss/trade, 75%% position cap, 10%% cash buffer, whole shares, "
                    "%d trades/day, $150 daily loss, entries 9:45-3:30, flat by 3:55; S&P 500 stocks >= $5; "
                    "fills at the next bar's open +0.05%%" % (budget, cfg["max_trades_per_day"]),
        "run_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT.mkdir(exist_ok=True)
    meta = [{"label": "backtest_plugin.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"}]
    for name, rows, m in (("plugin_summary.json", [summary], meta), ("plugin_equity.json", equity_rows, []),
                          ("plugin_trades.json", trades, []), ("plugin_months.json", months, [])):
        (OUT / name).write_text(bd.to_json({"rows": rows, "meta": m}, separators=(",", ":")))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Backtest the day-trader-skill plugin with a small account")
    ap.add_argument("--years", type=int, default=1)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--max-trades", type=int, default=3, help="trades allowed per day (the skills default to 3)")
    a = ap.parse_args()
    print(bd.to_json(run(a.years, a.budget, a.max_trades), indent=1))


if __name__ == "__main__":
    main()
