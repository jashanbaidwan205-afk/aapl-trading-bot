"""Tests for sip_bot.py with a fake broker (no network, no orders)."""
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


class FakeBroker:
    def __init__(self, equity=100_000.0, shortable=True):
        self.equity = equity
        self.orders = []
        self.pos = {}
        self.shortable = shortable
        self.cancelled = []

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
        self.orders.append({"id": cid, "cid": cid, "symbol": symbol, "side": side, "type": "market", "status": "filled",
                            "qty": qty, "filled_qty": qty, "filled_avg_price": None, "stop_price": None})
        self.pos.pop(symbol, None)

    def cancel(self, oid):
        for o in self.orders:
            if o["id"] == oid:
                o["status"] = "canceled"
        self.cancelled.append(oid)

    # helpers for the test
    def fill(self, cid, price, qty_sign=1):
        o = next(o for o in self.orders if o["cid"] == cid)
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


class ManageTests(unittest.TestCase):
    def setUp(self):
        self.b = FakeBroker()
        self.p = picks()
        self.plan = {"events": []}

    def run_once(self, entries=True):
        sip_bot.manage(self.b, self.p, DAY, T, T, entries, self.plan)

    def test_entries_fill_cap_in_rank_order(self):
        self.run_once()
        entries = [o for o in self.b.orders if "-e-" in o["cid"]]
        # each position is capped at 10% of $100k, cap is 50% -> 5 entry orders
        self.assertEqual([o["symbol"] for o in entries], ["AAA", "BBB", "CCC", "DDD", "EEE"])
        a = entries[0]
        self.assertEqual((a["side"], a["qty"], a["stop_price"]), ("buy", 100, 100))
        bb = entries[1]
        self.assertEqual((bb["side"], bb["qty"], bb["stop_price"]), ("sell", 200, 50))
        # a second pass places nothing new
        self.run_once()
        self.assertEqual(len([o for o in self.b.orders if "-e-" in o["cid"]]), 5)

    def test_fill_places_protective_stop_once(self):
        self.run_once()
        self.b.fill(sip_bot.prefix(DAY) + "e-AAA", 100.10)
        self.b.fill(sip_bot.prefix(DAY) + "e-BBB", 49.95)
        self.run_once()
        self.run_once()
        stops = {o["symbol"]: o for o in self.b.orders if "-s-" in o["cid"]}
        self.assertEqual(set(stops), {"AAA", "BBB"})
        self.assertEqual((stops["AAA"]["side"], stops["AAA"]["qty"], stops["AAA"]["stop_price"]), ("sell", 100, 99.6))
        self.assertEqual((stops["BBB"]["side"], stops["BBB"]["qty"], stops["BBB"]["stop_price"]), ("buy", 200, 50.25))

    def test_cancelled_entry_frees_room_but_is_not_retried(self):
        self.run_once()
        self.b.cancel(sip_bot.prefix(DAY) + "e-CCC")
        self.run_once()
        syms = [o["symbol"] for o in self.b.orders if "-e-" in o["cid"]]
        self.assertEqual(syms.count("CCC"), 1)
        self.assertIn("FFF", syms)          # next in rank took the freed room
        self.assertNotIn("GGG", syms)

    def test_not_shortable_is_skipped(self):
        self.b.shortable = False
        self.run_once()
        syms = [o["symbol"] for o in self.b.orders if "-e-" in o["cid"]]
        self.assertNotIn("BBB", syms)
        self.assertNotIn("EEE", syms)
        self.assertTrue(any(e["event"].startswith("skipped") for e in self.plan["events"]))

    def test_no_entries_after_window(self):
        self.run_once(entries=False)
        self.assertEqual(self.b.orders, [])

    def test_flatten_closes_only_own_positions(self):
        self.run_once()
        self.b.fill(sip_bot.prefix(DAY) + "e-AAA", 100.10)
        self.b.fill(sip_bot.prefix(DAY) + "e-BBB", 49.95)
        self.run_once()
        self.b.pos["XOM"] = {"qty": 10.0, "market_value": 1000.0, "avg_entry_price": 100}   # crossover bot's stock
        sip_bot.flatten(self.b, DAY, T, self.plan, T)
        closes = {o["symbol"]: o for o in self.b.orders if "-x-" in o["cid"]}
        self.assertEqual(set(closes), {"AAA", "BBB"})
        self.assertEqual((closes["AAA"]["side"], closes["AAA"]["qty"]), ("sell", 100))
        self.assertEqual((closes["BBB"]["side"], closes["BBB"]["qty"]), ("buy", 200))
        self.assertIn("XOM", self.b.pos)
        self.assertTrue(all(o["status"] != "new" for o in self.b.orders))  # every open order cancelled


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
