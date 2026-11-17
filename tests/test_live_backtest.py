"""Checks that backtest_daytrade.simulate_live follows the live day trader's rules."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest_daytrade as dt  # noqa: E402
import backtest_orb as orb  # noqa: E402

DAY = "2026-01-05"


def day_bars(path):
    """path: list of (open, high, low, close) for 5-minute bars from 9:30 to 15:55 (78 bars); padded flat."""
    t = pd.Timestamp(f"{DAY} 09:30")
    rows = list(path) + [path[-1][3:] * 4] * (78 - len(path))
    return orb.DayBars(pd.DataFrame([dict(hm=(t + pd.Timedelta(minutes=5 * i)).strftime("%H:%M"),
                                          open=o, high=h, low=l, close=c, volume=1e5, vwap=c)
                                     for i, (o, h, l, c) in enumerate(rows)]))


class LiveRulesTests(unittest.TestCase):
    def run_day(self, bars, picks):
        spy = {DAY: 100_000.0}
        return dt.simulate_live([DAY], {DAY: picks}, {(s, DAY): b for s, b in bars.items()}, spy, 0.0)

    def test_rank_order_cap_and_room_release(self):
        # 6 picks, each ~$10k (10% cap) -> only 5 orders fit under the 50% limit at 9:35.
        # #1 AAA breaks out at 9:40 and is stopped at 9:45; its room frees from the 9:50 bar,
        # so #6 FFF gets an order then and fills when it breaks at 10:00.
        flat = [(100, 100.1, 99.9, 100)]
        bars = {
            "AAA": day_bars([(100, 100.5, 99.5, 100.4), (100.4, 100.45, 100.2, 100.3), (100.3, 101, 100.3, 100.9),
                             (100.9, 100.9, 100.0, 100.1)]),
            "BBB": day_bars(flat), "CCC": day_bars(flat), "DDD": day_bars(flat), "EEE": day_bars(flat),
            "FFF": day_bars([(100, 100.5, 99.5, 100.4)] + [(100.4, 100.45, 100.3, 100.4)] * 5
                            + [(100.4, 101, 100.4, 100.9), (100.9, 101.5, 100.8, 101.4)]),
        }
        picks = [{"rank": i, "symbol": s, "side": "long", "level": 100.5, "dist": 0.3}
                 for i, s in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], start=1)]
        trades, eq, _, _ = self.run_day(bars, picks)
        self.assertEqual(eq[0]["orders_placed"], 6)
        by = {t["symbol"]: t for t in trades}
        self.assertEqual(set(by), {"AAA", "FFF"})
        self.assertEqual((by["AAA"]["entry_hm"], by["AAA"]["exit_hm"], by["AAA"]["exit_reason"]), ("09:40", "09:45", "stop"))
        self.assertEqual(by["FFF"]["entry_hm"], "10:00")
        self.assertEqual(by["FFF"]["exit_reason"], "close")
        self.assertEqual(by["FFF"]["exit_hm"], "15:55")      # closed at the open of the 3:55 bar

    def test_no_entries_after_cutoff(self):
        path = [(100, 100.5, 99.5, 100.4)] + [(100.4, 100.45, 100.3, 100.4)] * 70 + [(100.4, 102, 100.4, 101.9)]
        trades, eq, _, _ = self.run_day({"AAA": day_bars(path)},
                                        [{"rank": 1, "symbol": "AAA", "side": "long", "level": 100.5, "dist": 0.3}])
        self.assertEqual(trades, [])            # the break happens after 3:00 PM

    def test_short_side(self):
        path = [(100, 100.5, 99.5, 99.6), (99.6, 99.7, 99.3, 99.4), (99.4, 99.45, 98.0, 98.1)]
        trades, eq, _, _ = self.run_day({"AAA": day_bars(path)},
                                        [{"rank": 1, "symbol": "AAA", "side": "short", "level": 99.5, "dist": 0.3}])
        t = trades[0]
        self.assertEqual((t["side"], t["entry_hm"], t["exit_reason"]), ("short", "09:35", "close"))
        self.assertGreater(t["pnl"], 0)


if __name__ == "__main__":
    unittest.main()
