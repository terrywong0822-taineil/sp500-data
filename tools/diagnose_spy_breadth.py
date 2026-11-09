from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA = Path("data/universe")
PRICES = Path("data/validation/prices.parquet")
NO_TRADE = Path("data/validation/no_trade_days.csv")
FIXTURE = Path("tests/fixtures/spy-breadth-weekly-2018-12-21_to_2026-10-09.csv")


def main() -> None:
    baseline = pd.read_csv(FIXTURE, parse_dates=["week_label", "spy_source_date"])
    intervals = pd.read_parquet(DATA / "membership_intervals.parquet")
    intervals["start"] = pd.to_datetime(intervals["start"])
    intervals["end"] = pd.to_datetime(intervals["end"])

    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "adj_close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str)
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices[(prices.date >= "2017-01-01") &
                    (prices.date <= baseline.spy_source_date.max())].copy()
    prices = prices.sort_values(["ticker", "date"], kind="mergesort")
    prices["ma200"] = prices.groupby("ticker", sort=False).adj_close.transform(
        lambda s: s.rolling(200, min_periods=200).mean()
    )

    no_trade = pd.read_csv(NO_TRADE)
    lower = {str(c).lower(): c for c in no_trade.columns}
    ticker_col = next(lower[k] for k in ("ticker", "symbol") if k in lower)
    date_col = next(lower[k] for k in ("date", "trade_date", "session_date") if k in lower)
    no_trade_pairs = set(zip(no_trade[ticker_col].astype(str),
                             pd.to_datetime(no_trade[date_col]).dt.normalize()))

    mismatches = []
    for row in baseline.itertuples(index=False):
        dt = pd.Timestamp(row.spy_source_date).normalize()
        active = set(intervals.loc[
            (intervals.start <= dt) & (intervals.end.isna() | (intervals.end > dt)),
            "ticker"].astype(str))
        day = prices[prices.date.dt.normalize() == dt].copy()
        day = day[~day.ticker.map(lambda t: (t, dt) in no_trade_pairs)]
        eligible = set(day.loc[day.ma200.notna() & day.adj_close.notna(), "ticker"])
        old_n = int(row.eligible_members)
        if len(eligible) != old_n:
            mismatches.append((row, dt, active, day, eligible))

    print(f"DIAGNOSTIC mismatching dates={len(mismatches)}")
    for row, dt, active, day, eligible in mismatches:
        active_day = day[day.ticker.isin(active)].copy()
        no_ma_active = sorted(set(active_day.loc[active_day.ma200.isna(), "ticker"]))
        missing_price = sorted(active - set(day.ticker))
        print(f"DATE {dt.date()} baseline_eligible={int(row.eligible_members)} "
              f"rebuilt_eligible={len(eligible & active)} active={len(active)}")
        print(f"ACTIVE_MISSING_PRICE {missing_price}")
        print(f"ACTIVE_NO_MA {no_ma_active}")
        print(f"ACTIVE_ELIGIBLE {sorted(eligible & active)}")
        for ticker in no_ma_active:
            hist = prices[(prices.ticker == ticker) & (prices.date <= dt) &
                          prices.adj_close.notna()]
            print(f"NO_MA_DETAIL {ticker}: valid_obs={len(hist)} "
                  f"first={hist.date.min().date() if len(hist) else None} "
                  f"last={hist.date.max().date() if len(hist) else None} "
                  f"adj_close={active_day.loc[active_day.ticker == ticker, 'adj_close'].tolist()} "
                  f"ma200={active_day.loc[active_day.ticker == ticker, 'ma200'].tolist()}")
        print("END_DATE")


if __name__ == "__main__":
    main()
