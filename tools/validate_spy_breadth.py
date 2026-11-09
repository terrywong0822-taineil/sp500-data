from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/universe")
FIXTURE = ROOT / "tests/fixtures/spy-breadth-weekly-2018-12-21_to_2026-10-09.csv"
PRICES = Path("data/validation/prices.parquet")
NO_TRADE = Path("data/validation/no_trade_days.csv")

EXPECTED_PRICES_SHA256 = "c398ee0aaf67e8a3280e21dcf99378bed42d05be62d63db8d6681ef6d0133209"
EXPECTED_NO_TRADE_SHA256 = "e6ce9efaeea91b3e3e8198a6a3b20bf56d217e82fe83c83b133fab27492de2f1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for path, expected in ((PRICES, EXPECTED_PRICES_SHA256), (NO_TRADE, EXPECTED_NO_TRADE_SHA256)):
        actual = sha256(path)
        assert actual == expected, f"Input hash changed for {path}: {actual}"

    baseline = pd.read_csv(FIXTURE, parse_dates=["week_label", "spy_source_date"])
    assert len(baseline) == 408, f"Expected 408 frozen weekly labels, got {len(baseline)}"

    intervals = pd.read_parquet(DATA / "membership_intervals.parquet")
    intervals["start"] = pd.to_datetime(intervals["start"])
    intervals["end"] = pd.to_datetime(intervals["end"])

    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "adj_close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str)
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    # Keep sufficient pre-window history to warm up a 200-trading-observation SMA.
    prices = prices[(prices["date"] >= pd.Timestamp("2017-01-01")) &
                    (prices["date"] <= baseline["spy_source_date"].max())].copy()
    prices = prices.sort_values(["ticker", "date"], kind="mergesort")
    prices["ma200"] = prices.groupby("ticker", sort=False)["adj_close"].transform(
        lambda s: s.rolling(200, min_periods=200).mean()
    )

    no_trade = pd.read_csv(NO_TRADE)
    lower = {str(c).lower(): c for c in no_trade.columns}
    ticker_col = next((lower[k] for k in ("ticker", "symbol") if k in lower), None)
    date_col = next((lower[k] for k in ("date", "trade_date", "session_date") if k in lower), None)
    assert ticker_col and date_col, f"Unrecognized no_trade_days columns: {no_trade.columns.tolist()}"
    no_trade_pairs = set(zip(
        no_trade[ticker_col].astype(str),
        pd.to_datetime(no_trade[date_col]).dt.normalize()
    ))

    rows = []
    for label in baseline.itertuples(index=False):
        source_date = pd.Timestamp(label.spy_source_date).normalize()
        active = intervals[
            (intervals["start"] <= source_date) &
            (intervals["end"].isna() | (intervals["end"] > source_date))
        ][["ticker"]].copy()
        assert not active["ticker"].duplicated().any(), f"Overlapping membership on {source_date.date()}"

        day = prices[prices["date"].dt.normalize() == source_date][
            ["ticker", "adj_close", "ma200"]
        ]
        eligible = active.merge(day, on="ticker", how="inner")
        if no_trade_pairs:
            eligible = eligible[
                ~eligible.apply(lambda r: (str(r.ticker), source_date) in no_trade_pairs, axis=1)
            ]
        eligible = eligible.dropna(subset=["adj_close", "ma200"])
        above = int((eligible["adj_close"] > eligible["ma200"]).sum())
        count = int(len(eligible))
        pct = above / count * 100 if count else float("nan")
        rows.append({
            "week_label": pd.Timestamp(label.week_label).date().isoformat(),
            "spy_source_date": source_date.date().isoformat(),
            "index_members": int(len(active)),
            "eligible_members": count,
            "above_200d_members": above,
            "pct_above_200d": round(pct, 4),
            "below_50pct": int(pct < 50.0),
        })

    rebuilt = pd.DataFrame(rows)
    expected = baseline.copy()
    expected["week_label"] = expected["week_label"].dt.strftime("%Y-%m-%d")
    expected["spy_source_date"] = expected["spy_source_date"].dt.strftime("%Y-%m-%d")

    key = ["week_label", "spy_source_date"]
    merged = expected.merge(rebuilt, on=key, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(merged) == 408
    for col in ("index_members", "eligible_members", "above_200d_members", "below_50pct"):
        diff = merged[f"{col}_old"].astype(int) - merged[f"{col}_new"].astype(int)
        assert (diff == 0).all(), f"{col} differs on {(diff != 0).sum()} weeks"
    pct_diff = (merged["pct_above_200d_old"] - merged["pct_above_200d_new"]).abs()
    assert pct_diff.max() <= 0.00005, f"Max width deviation {pct_diff.max():.8f}pp"

    below = rebuilt["below_50pct"].astype(bool)
    segments = int((below & ~below.shift(1, fill_value=False)).sum())
    assert int(below.sum()) == 94, f"Expected 94 weeks below 50%, got {int(below.sum())}"
    assert segments == 16, f"Expected 16 below-50% segments, got {segments}"

    out = Path("data/validation/spy-breadth-rebuilt.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(out, index=False)
    print("PASS: rebuilt SPY breadth using the repaired point-in-time membership intervals")
    print("PASS: 408/408 frozen weekly labels; all membership/eligibility/above-count/threshold values match")
    print(f"PASS: maximum pct-above-200d deviation = {pct_diff.max():.8f} percentage points")
    print(f"PASS: below 50% = {int(below.sum())}/408 weeks in {segments} segments; zero threshold flips")
    print(f"INFO: prices sha256={EXPECTED_PRICES_SHA256}; no_trade_days sha256={EXPECTED_NO_TRADE_SHA256}")
    print(f"INFO: rebuilt CSV saved to {out}")


if __name__ == "__main__":
    main()
