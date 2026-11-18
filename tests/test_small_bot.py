"""Tests for small_bot.py (the separate $1,000 bot) with a fake broker: no network, no orders."""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import small_bot  # noqa: E402

NY = ZoneInfo("America/New_York")
DAY = "2026-11-19"
T = datetime(2026, 11, 19, 9, 36, tzinfo=NY)
PRE = small_bot.prefix(DAY)


class FakeBroker:
    def __init__(self, equity=1_000.0, fractionable=True):
        self.equity, self.frac = equity, fractionable
        self.orders, self.pos = [], {}

    def account(self):
        return {"equity": self.equity, "cash": self.equity}

    def positions(self):
        return dict(self.pos)

    def orders_today(self, since):
        return [dict(o) for o in self.orders]

    def asset(self, symbol):
        return {"tradable": True, "shortable": False, "easy_to_borrow": False, "fractionable": self.frac}

    def stop_order(self, symbol, qty, side, stop_price, cid):
        self.orders.append({"id": cid, "cid": cid, "symbol": symbol, "side": side, "type": "stop", "status": "new",
                            "qty": qty, "filled_qty": 0.0, "filled_avg_price": None, "stop_price": round(stop_price, 2)})

    def move_stop(self, oid, stop_price, cid):
        old = next(o for o in self.orders if o["id"] == oid)
        old["status"] = "replaced"
        self.orders.append(dict(old, id=cid, cid=cid, status="new", stop_price=round(stop_price, 2)))

    def market_order(self, symbol, qty, side, cid):
        o = {"id": cid, "cid": cid, "symbol": symbol, "side": side, "type": "market", "status": "new",
             "qty": qty, "filled_qty": 0.0, "filled_avg_price": None, "stop_price": None}
        self.orders.append(o)
        if "-x-" in cid:
            o.update(status="filled", filled_qty=qty)
            self.pos.pop(symbol, None)

    def cancel(self, oid):
        for o in self.orders:
            if o["id"] == oid:
                o["status"] = "canceled"

    def fill(self, cid, price):
        o = next(o for o in self.orders if o["cid"].startswith(cid))
        o.update(status="filled", filled_qty=o["qty"], filled_avg_price=price)
        self.pos[o["symbol"]] = {"qty": float(o["qty"]), "market_value": o["qty"] * price, "avg_entry_price": price}


def pick(sym="AAA", level=300.0, dist=0.5, rank=1, side="long"):
    return {"symbol": sym, "side": side, "level": level, "stop_dist": dist, "rank": rank, "rvol": 5.0, "atr": dist * 10}


class SizeTests(unittest.TestCase):
    def test_fractional_size_is_20_percent(self):
        self.assertAlmostEqual(small_bot.size(1_000, 300, 0.5), 0.6666)          # $200 / $300
        self.assertAlmostEqual(small_bot.size(1_000, 20, 1.0), 10.0)             # 1% risk: $10 / $1 = 10 shares ($200)

    def test_whole_shares_when_not_fractionable(self):
        self.assertEqual(small_bot.size(1_000, 300, 0.5, fractional=False), 0)
        self.assertEqual(small_bot.size(1_000, 60, 0.3, fractional=False), 3)

    def test_long_only_keeps_ranks(self):
        ps = [pick("A", rank=1), pick("B", rank=2, side="short"), pick("C", rank=3)]
        self.assertEqual([(p["symbol"], p["rank"]) for p in small_bot.long_only(ps)], [("A", 1), ("C", 3)])


class TradeTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"events": []}

    def run_once(self, b, picks, quotes, entries=True):
        small_bot.manage(b, picks, DAY, T, T, entries, self.plan, quotes=quotes)

    def test_breakout_buys_a_fraction_and_protects_it(self):
        b, ps = FakeBroker(), [pick()]
        q = {"AAA": {"last": 300.2, "high": 300.3, "low": 299.9}}
        self.run_once(b, ps, q)
        entry = [o for o in b.orders if "-e-" in o["cid"]][0]
        self.assertAlmostEqual(entry["qty"], 0.6662)                             # $200 / $300.20
        b.fill(PRE + "e-AAA", 300.25)
        self.run_once(b, ps, q)
        stop = [o for o in b.orders if "-s-" in o["cid"]][0]
        self.assertEqual((stop["side"], stop["qty"], stop["stop_price"]), ("sell", 0.6662, 299.75))

    def test_expensive_stock_skipped_when_not_fractionable(self):
        b, ps = FakeBroker(fractionable=False), [pick()]
        self.run_once(b, ps, {"AAA": {"last": 300.2, "high": 300.3, "low": 299.9}})
        self.assertEqual([o for o in b.orders if "-e-" in o["cid"]], [])
        self.assertEqual(ps[0]["state"], "too small")
        self.assertIn("not fractionable", self.plan["events"][-1]["event"])

    def test_breakeven_move_keeps_the_fraction(self):
        b, ps = FakeBroker(), [pick()]
        q = {"AAA": {"last": 300.2, "high": 300.3, "low": 299.9}}
        self.run_once(b, ps, q)
        b.fill(PRE + "e-AAA", 300.25)
        self.run_once(b, ps, q)
        q["AAA"] = {"last": 300.7, "high": 300.8, "low": 300.6}                  # +1R = 300.75
        self.run_once(b, ps, q)
        new = [o for o in b.orders if o["cid"].endswith("-be")][0]
        self.assertEqual((new["qty"], new["stop_price"]), (0.6662, 300.25))

    def test_end_of_day_sells_the_exact_fraction(self):
        b, ps = FakeBroker(), [pick()]
        self.run_once(b, ps, {"AAA": {"last": 300.2, "high": 300.3, "low": 299.9}})
        b.fill(PRE + "e-AAA", 300.25)
        small_bot.flatten(b, DAY, T, self.plan, T)
        x = [o for o in b.orders if "-x-" in o["cid"]][0]
        self.assertEqual((x["side"], x["qty"]), ("sell", 0.6662))

    def test_orders_are_named_sml(self):
        self.assertTrue(PRE.startswith("sml-"))


class SafetyTests(unittest.TestCase):
    def test_refuses_an_account_over_the_limit(self):
        made = []

        class Big(FakeBroker):
            def __init__(self, key, secret, dry):
                super().__init__(equity=100_000.0)
                made.append(self)

            def clock(self):
                raise AssertionError("should not get this far")

        real_broker, real_keys = small_bot.Broker, small_bot.keys
        small_bot.Broker, small_bot.keys = Big, (lambda: ("k", "s"))
        try:
            small_bot.run("day", dry_run=True)
        finally:
            small_bot.Broker, small_bot.keys = real_broker, real_keys
        self.assertEqual(made[0].orders, [])


if __name__ == "__main__":
    unittest.main()
