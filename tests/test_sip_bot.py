"""Tests for sip_bot.py with a fake broker and fake quotes (no network, no orders)."""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sip_bot  # noqa: E402

NY = ZoneInfo("America/New_York")
DAY = "2026-11-17"
T = datetime(2026, 11, 17, 9, 36, tzinfo=NY)
PRE = sip_bot.prefix(DAY)


class FakeBroker:
    def __init__(self, equity=100_000.0, shortable=True):
        self.equity = equity
        self.orders = []
        self.pos = {}
        self.shortable = shortable

    def account(self):
        return {"equity": self.equity, "cash": self.equity}

    def positions(self):
        return dict(self.pos)

    def orders_today(self, since):
        return [dict(o) for o in self.orders]

    def asset(self, symbol):
        return {"tradable": True, "shortable": self.shortable, "easy_to_borrow": self.shortable}

    def stop_order(self, symbol, qty, side, stop_price, cid):
        self.orders.append({"id": cid, "cid": cid, "symbol": symbol, "side": side, "type": "stop", "status": "new",
                            "qty": qty, "filled_qty": 0.0, "filled_avg_price": None, "stop_price": round(stop_price, 2)})

    def market_order(self, symbol, qty, side, cid):
        o = {"id": cid, "cid": cid, "symbol": symbol, "side": side, "type": "market", "status": "new",
             "qty": qty, "filled_qty": 0.0, "filled_avg_price": None, "stop_price": None}
        self.orders.append(o)
        if "-x-" in cid:  # end-of-day close-outs fill at once
            o.update(status="filled", filled_qty=qty)
            self.pos.pop(symbol, None)

    def cancel(self, oid):
        for o in self.orders:
            if o["id"] == oid:
                o["status"] = "canceled"

    def fill(self, cid, price):
        o = next(o for o in self.orders if o["cid"].startswith(cid))
        o.update(status="filled", filled_qty=o["qty"], filled_avg_price=price)
        q = o["qty"] if o["side"] == "buy" else -o["qty"]
        self.pos[o["symbol"]] = {"qty": float(q), "market_value": q * price, "avg_entry_price": price}


def picks():
    out = []
    for i, (s, side, lvl, dist) in enumerate([("AAA", "long", 100, 0.5), ("BBB", "short", 50, 0.3),
                                              ("CCC", "long", 200, 1.0), ("DDD", "long", 80, 0.4),
                                              ("EEE", "short", 40, 0.2), ("FFF", "long", 25, 0.1),
                                              ("GGG", "long", 60, 0.3)], start=1):
        out.append({"symbol": s, "side": side, "level": lvl, "stop_dist": dist, "rank": i, "rvol": 10 - i, "atr": dist * 10})
    return out


def quiet(ps):
    """Quotes where nothing has reached its level yet."""
    q = {}
    for p in ps:
        px = p["level"] * (0.99 if p["side"] == "long" else 1.01)
        q[p["symbol"]] = {"last": px, "high": px, "low": px}
    return q


def broke(q, p, last=None):
    px = last if last is not None else p["level"] * (1.0005 if p["side"] == "long" else 0.9995)
    q[p["symbol"]] = {"last": px, "high": max(px, p["level"]), "low": min(px, p["level"])}


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.b = FakeBroker()
        self.p = picks()
        self.plan = {"events": []}
        # the tests use a 50% limit (5 positions of ~10%), whatever config.py says
        self._cap = sip_bot.config.SIP_MAX_INVESTED_PCT
        sip_bot.config.SIP_MAX_INVESTED_PCT = 50.0

    def tearDown(self):
        sip_bot.config.SIP_MAX_INVESTED_PCT = self._cap

    def run_once(self, quotes, entries=True):
        sip_bot.manage(self.b, self.p, DAY, T, T, entries, self.plan, quotes=quotes)

    def entries(self):
        return [o for o in self.b.orders if "-e-" in o["cid"]]

    def test_nothing_until_a_breakout(self):
        self.run_once(quiet(self.p))
        self.assertEqual(self.b.orders, [])

    def test_first_breakout_gets_the_room_not_the_top_rank(self):
        q = quiet(self.p)
        broke(q, self.p[2])   # CCC (rank 3) breaks out first
        broke(q, self.p[4])   # EEE (rank 5, short)
        self.run_once(q)
        got = {o["symbol"]: o for o in self.entries()}
        self.assertEqual(set(got), {"CCC", "EEE"})
        self.assertEqual(got["CCC"]["cid"], PRE + "e-CCC-a20000")   # aimed-for price in cents
        self.assertEqual((got["CCC"]["type"], got["CCC"]["side"]), ("market", "buy"))
        self.assertEqual(got["EEE"]["side"], "sell")
        # a breakout is acted on once
        self.run_once(q)
        self.assertEqual(len(self.entries()), 2)

    def test_limit_counts_unfilled_market_orders_and_skips_when_full(self):
        q = quiet(self.p)
        for p in self.p[:6]:
            broke(q, p)   # six break out in the same few seconds; each position is ~10% of the account
        self.run_once(q)
        self.assertEqual([o["symbol"] for o in self.entries()], ["AAA", "BBB", "CCC", "DDD", "EEE"])
        fff = next(p for p in self.p if p["symbol"] == "FFF")
        self.assertEqual(fff["state"], "missed: no room")
        # FFF is not retried later even if room frees up (same as the backtest)
        self.b.fill(PRE + "e-AAA", 100.05)
        self.b.pos.pop("AAA")   # stopped out
        self.run_once(q)
        self.assertNotIn("FFF", [o["symbol"] for o in self.entries()])

    def test_full_account_limit_allows_ten_positions(self):
        sip_bot.config.SIP_MAX_INVESTED_PCT = 100.0
        more = picks() + [dict(p, symbol=p["symbol"] + "X", rank=p["rank"] + 7) for p in picks()]
        q = quiet(more)
        for p in more:
            broke(q, p)
        sip_bot.manage(self.b, more, DAY, T, T, True, self.plan, quotes=q)
        self.assertEqual(len(self.entries()), 10)   # ten positions of ~10% each, then the limit is full

    def test_room_frees_after_an_exit(self):
        q = quiet(self.p)
        for p in self.p[:5]:
            broke(q, p)
        self.run_once(q)
        for o in self.entries():
            self.b.fill(o["cid"], o["qty"] and next(p["level"] for p in self.p if p["symbol"] == o["symbol"]))
        self.b.pos.pop("AAA")   # AAA was stopped out: room for one more
        broke(q, self.p[6])     # GGG breaks out now
        self.run_once(q)
        self.assertIn("GGG", [o["symbol"] for o in self.entries()])

    def test_fill_places_stop_and_records_slippage(self):
        q = quiet(self.p)
        broke(q, self.p[0])
        broke(q, self.p[1])
        self.run_once(q)
        self.b.fill(PRE + "e-AAA", 100.10)   # aimed 100.00 -> 0.10% worse
        self.b.fill(PRE + "e-BBB", 49.95)    # short aimed 50.00 -> 0.10% worse
        self.run_once(q)
        self.run_once(q)
        stops = {o["symbol"]: o for o in self.b.orders if "-s-" in o["cid"]}
        self.assertEqual((stops["AAA"]["side"], stops["AAA"]["qty"], stops["AAA"]["stop_price"]), ("sell", 99, 99.6))   # sized at the price seen at the breakout (100.05)
        self.assertEqual((stops["BBB"]["side"], stops["BBB"]["stop_price"]), ("buy", 50.25))
        fills = {e["symbol"]: e for e in self.plan["events"] if e["event"] == "entry filled"}
        self.assertAlmostEqual(fills["AAA"]["slippage_pct"], 0.1, places=3)
        self.assertAlmostEqual(fills["BBB"]["slippage_pct"], 0.1, places=3)

    def test_not_shortable_is_skipped(self):
        self.b.shortable = False
        q = quiet(self.p)
        broke(q, self.p[1])
        self.run_once(q)
        self.assertEqual(self.entries(), [])
        self.assertTrue(any(e["event"].startswith("skipped: not shortable") for e in self.plan["events"]))

    def test_no_entries_after_cutoff(self):
        q = quiet(self.p)
        broke(q, self.p[0])
        self.run_once(q, entries=False)
        self.assertEqual(self.b.orders, [])

    def test_flatten_closes_only_own_positions(self):
        q = quiet(self.p)
        broke(q, self.p[0])
        broke(q, self.p[1])
        self.run_once(q)
        self.b.fill(PRE + "e-AAA", 100.10)
        self.b.fill(PRE + "e-BBB", 49.95)
        self.run_once(q)
        self.b.pos["XOM"] = {"qty": 10.0, "market_value": 1000.0, "avg_entry_price": 100}   # crossover bot
        sip_bot.flatten(self.b, DAY, T, self.plan, T)
        closes = {o["symbol"]: o for o in self.b.orders if "-x-" in o["cid"]}
        self.assertEqual(set(closes), {"AAA", "BBB"})
        self.assertEqual((closes["AAA"]["side"], closes["BBB"]["side"]), ("sell", "buy"))
        self.assertIn("XOM", self.b.pos)
        self.assertTrue(all(o["status"] != "new" for o in self.b.orders if o["type"] == "stop"))


class SelectTests(unittest.TestCase):
    def test_select_ranks_filters_and_sides(self):
        bars = pd.DataFrame([
            {"symbol": "UP", "open": 10.0, "high": 10.6, "low": 9.9, "close": 10.5, "volume": 5000},
            {"symbol": "DOWN", "open": 50.0, "high": 50.1, "low": 49.0, "close": 49.2, "volume": 9000},
            {"symbol": "FLAT", "open": 30.0, "high": 30.5, "low": 29.5, "close": 30.0, "volume": 9000},
            {"symbol": "QUIET", "open": 40.0, "high": 40.5, "low": 39.9, "close": 40.4, "volume": 100},
            {"symbol": "HELD", "open": 60.0, "high": 61.0, "low": 59.0, "close": 60.9, "volume": 9000},
            {"symbol": "CHEAP", "open": 5.0, "high": 5.2, "low": 4.9, "close": 5.1, "volume": 9000},
        ])
        base = pd.Series({"UP": 1000, "DOWN": 1000, "FLAT": 1000, "QUIET": 1000, "HELD": 1000, "CHEAP": 1000})
        stats = pd.DataFrame({"prev_close": [10.2, 51, 30, 40, 60, 5], "atr": [1, 2, 1, 1, 1, 1],
                              "avg_volume": [2e6] * 6}, index=["UP", "DOWN", "FLAT", "QUIET", "HELD", "CHEAP"])
        p = sip_bot.select_stocks(bars, base, stats, exclude={"HELD"})
        self.assertEqual([(x["symbol"], x["side"], x["level"], x["rank"]) for x in p],
                         [("DOWN", "short", 49.0, 1), ("UP", "long", 10.6, 2)])
        self.assertAlmostEqual(p[0]["stop_dist"], 0.2)

    def test_size(self):
        self.assertEqual(sip_bot.size(100_000, 100, 0.5), 100)   # capped by 10% position
        self.assertEqual(sip_bot.size(100_000, 20, 1.0), 500)    # capped by 0.5% risk ($500 / $1)


if __name__ == "__main__":
    unittest.main()
