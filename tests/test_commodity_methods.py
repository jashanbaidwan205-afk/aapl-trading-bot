"""Checks the daily strategy engine in backtest_commodities on made-up prices."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_commodities as bc  # noqa: E402


def make_ctx(opens, highs, lows, closes, start="2026-03-02", spy=None):
    dates = pd.bdate_range(start, periods=len(closes))
    f = lambda xs: pd.DataFrame({"X": xs, "SPY": spy if spy is not None else [100.0] * len(xs),  # noqa: E731
                                 "UUP": [27.0] * len(xs)}, index=dates, dtype=float)
    daily = {"Open": f(opens), "High": f(highs), "Low": f(lows), "Close": f(closes)}
    return {"ind": {"X": bc.indicators(daily, "X")}}, [d.strftime("%Y-%m-%d") for d in dates]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.saved = bc.SLIP_PCT
        bc.SLIP_PCT = 0.0

    def tearDown(self):
        bc.SLIP_PCT = self.saved

    def test_overnight_buys_close_sells_next_open(self):
        warm = [100.0] * 15                       # indicators need 14 days of history
        o = warm + [100, 101, 103, 102]
        c = warm + [100.5, 102, 102.5, 103]
        ctx, days = make_ctx(o, [x + 1 for x in c], [x - 1 for x in o], c)
        v = {"method": "overnight", "filter": False, "slip": 0.0, "fill": "close"}
        trades, rows = bc.run_daily(v, "X", days[15:], ctx)
        closed = [t for t in trades if t["exit_reason"] == "open"]
        self.assertEqual([(t["entry"], t["exit"]) for t in closed], [(100.5, 101.0), (102.0, 103.0), (102.5, 102.0)])
        self.assertEqual(trades[-1]["exit_reason"], "still open")     # bought at the last close

    def test_stop_is_checked_on_the_daily_low(self):
        # a long entered at the next open with a stop at 98 is stopped the following day (low 97)
        ctx, days = make_ctx([100, 100, 99], [101, 101, 100], [99.5, 99.5, 97], [100, 100, 98.5])
        orders = iter([{"pos": 1, "stop": 98.0}, None, None])
        real = bc.decide
        bc.decide = lambda v, i, I, st: next(orders)
        try:
            trades, rows = bc.run_daily({"method": "x"}, "X", days, ctx)
        finally:
            bc.decide = real
        self.assertEqual((trades[0]["entry"], trades[0]["exit"], trades[0]["exit_reason"]), (100.0, 98.0, "stop"))
        # sized by 1% risk ($1,000 / $2 = 500 shares) but capped at 20% of the account ($20,000 / $100 = 200)
        self.assertEqual(trades[0]["shares"], 200)
        self.assertAlmostEqual(rows[-1]["equity"], bc.START_CASH - 400.0)

    def test_preholiday_holds_the_day_before_a_weekday_holiday(self):
        # 2026-05-21 (Thu), 22 (Fri), then Memorial Day Monday 25 is missing, 26 (Tue)
        dates = pd.to_datetime(["2026-05-20", "2026-05-21", "2026-05-22", "2026-05-26", "2026-05-27"])
        c = pd.Series([10.0, 10.1, 10.3, 10.2, 10.4], index=dates)
        I = {"c": c, "h": c + 0.1, "l": c - 0.1, "atr": pd.Series(0.2, index=dates)}
        st = {"pos": 0, "held": 0}
        # buy at the close of 05-21: the day after (05-22) is followed by a weekday holiday
        self.assertEqual(bc.decide({"method": "preholiday"}, 1, I, st), {"pos": 1})
        self.assertIsNone(bc.decide({"method": "preholiday"}, 0, I, st))
        self.assertEqual(bc.decide({"method": "preholiday"}, 2, I, {"pos": 1, "held": 1}), {"pos": 0})

    def test_fomc_drift_buys_the_close_before_statement_day(self):
        dates = pd.to_datetime(["2026-03-16", "2026-03-17", "2026-03-18"])
        c = pd.Series([1.0, 1.0, 1.0], index=dates)
        I = {"c": c, "h": c, "l": c, "atr": pd.Series(0.1, index=dates)}
        self.assertIsNone(bc.decide({"method": "fomc_drift"}, 0, I, {"pos": 0, "held": 0}))
        self.assertEqual(bc.decide({"method": "fomc_drift"}, 1, I, {"pos": 0, "held": 0}), {"pos": 1})

    def test_range_buys_oversold_and_exits_at_50(self):
        n = 40
        c = list(np.linspace(100, 101, n - 3)) + [97, 95, 99]
        ctx, days = make_ctx(c, [x + 0.5 for x in c], [x - 0.5 for x in c], c)
        I = ctx["ind"]["X"]
        v = {"method": "range", "osc": "rsi", "n": 2, "lo": 10, "hi": 90, "adx": False}
        self.assertEqual(bc.decide(v, n - 2, I, {"pos": 0, "held": 0})["pos"], 1)       # RSI(2) very low after two drops
        self.assertEqual(bc.decide(v, n - 1, I, {"pos": 1, "held": 1}), {"pos": 0})     # bounce: RSI back above 50

    def test_every_version_runs(self):
        names = [v["name"] for v in bc.versions() + bc.more_versions()]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(v["method"] in bc.MIN_FIRST_HALF_TRADES for v in bc.versions() + bc.more_versions()))


if __name__ == "__main__":
    unittest.main()
