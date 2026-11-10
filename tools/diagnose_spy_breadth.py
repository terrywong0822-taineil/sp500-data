from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/universe")
PRICES = Path("data/validation/prices.parquet")
NO_TRADE = Path("data/validation/no_trade_days.csv")
FIXTURE = ROOT / "tests/fixtures/spy-breadth-weekly-2018-12-21_to_2026-10-09.csv"

# The 2026-11-05 frozen rebuild used this exact published membership input.
EXPECTED_BASELINE_INTERVALS_SHA256 = (
    "83141f0c7d9eb69793db60694432426ea94eaa8c92e7e50c741bb5f605c48d54"
)
EXPECTED_PRICES_SHA256 = (
    "c398ee0aaf67e8a3280e21dcf99378bed42d05be62d63db8d6681ef6d0133209"
)
EXPECTED_NO_TRADE_SHA256 = (
    "e6ce9efaeea91b3e3e8198a6a3b20bf56d217e82fe83c83b133fab27492de2f1"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_intervals(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    return frame


def build_weekly(
    baseline: pd.DataFrame,
    intervals: pd.DataFrame,
    prices: pd.DataFrame,
    no_trade_pairs: set[tuple[str, pd.Timestamp]],
) -> tuple[pd.DataFrame, dict[str, dict[str, set[str]]]]:
    rows = []
    member_sets: dict[str, dict[str, set[str]]] = {}
    for label in baseline.itertuples(index=False):
        dt = pd.Timestamp(label.spy_source_date).normalize()
        active = set(intervals.loc[
            (intervals.start <= dt) & (intervals.end.isna() | (intervals.end > dt)),
            "ticker",
        ])
        day = prices.loc[prices.date.dt.normalize() == dt]
        day = day.loc[~day.ticker.map(lambda t: (t, dt) in no_trade_pairs)]
        eligible_rows = day.loc[
            day.ma200.notna() & day.adj_close.notna() & day.ticker.isin(active)
        ]
        eligible = set(eligible_rows.ticker.astype(str))
        above = set(eligible_rows.loc[
            eligible_rows.adj_close > eligible_rows.ma200, "ticker"
        ].astype(str))
        pct = len(above) / len(eligible) * 100 if eligible else float("nan")
        key = dt.date().isoformat()
        member_sets[key] = {
            "active": active,
            "eligible": eligible,
            "above": above,
        }
        rows.append({
            "week_label": pd.Timestamp(label.week_label).date().isoformat(),
            "spy_source_date": key,
            "index_members": len(active),
            "eligible_members": len(eligible),
            "above_200d_members": len(above),
            "pct_above_200d": round(pct, 4),
            "below_50pct": int(pct < 50.0),
        })
    return pd.DataFrame(rows), member_sets


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python tools/diagnose_spy_breadth.py "
            "<published_membership_intervals.parquet>"
        )

    old_intervals_path = Path(sys.argv[1])
    current_intervals_path = DATA / "membership_intervals.parquet"
    for path, expected in (
        (old_intervals_path, EXPECTED_BASELINE_INTERVALS_SHA256),
        (PRICES, EXPECTED_PRICES_SHA256),
        (NO_TRADE, EXPECTED_NO_TRADE_SHA256),
    ):
        actual = sha256(path)
        print(f"INPUT_SHA256 {path} {actual}")
        if actual != expected:
            raise AssertionError(
                f"Input hash mismatch for {path}: expected {expected}, got {actual}"
            )

    baseline = pd.read_csv(FIXTURE, parse_dates=["week_label", "spy_source_date"])
    assert len(baseline) == 408, f"Expected 408 frozen labels, got {len(baseline)}"

    old_intervals = load_intervals(old_intervals_path)
    new_intervals = load_intervals(current_intervals_path)

    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "adj_close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str)
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices[
        (prices.date >= "2017-01-01")
        & (prices.date <= baseline.spy_source_date.max())
    ].copy()
    prices = prices.sort_values(["ticker", "date"], kind="mergesort")
    prices["ma200"] = prices.groupby("ticker", sort=False).adj_close.transform(
        lambda s: s.rolling(200, min_periods=200).mean()
    )

    no_trade = pd.read_csv(NO_TRADE)
    lower = {str(c).lower(): c for c in no_trade.columns}
    ticker_col = next(lower[k] for k in ("ticker", "symbol") if k in lower)
    date_col = next(lower[k] for k in ("date", "trade_date", "session_date") if k in lower)
    no_trade_pairs = set(zip(
        no_trade[ticker_col].astype(str),
        pd.to_datetime(no_trade[date_col]).dt.normalize(),
    ))

    # Both arms use the same price rows, no-trade exclusions, and MA200 series.
    old_weekly, old_members = build_weekly(baseline, old_intervals, prices, no_trade_pairs)
    new_weekly, new_members = build_weekly(baseline, new_intervals, prices, no_trade_pairs)

    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    old_weekly.to_csv(out_dir / "spy-breadth-counterfactual-original-membership.csv", index=False)
    new_weekly.to_csv(out_dir / "spy-breadth-counterfactual-repaired-membership.csv", index=False)

    key = ["week_label", "spy_source_date"]
    paired = old_weekly.merge(
        new_weekly, on=key, suffixes=("_old_membership", "_repaired_membership"),
        validate="one_to_one",
    )
    paired["eligible_delta_repaired_minus_old"] = (
        paired.eligible_members_repaired_membership
        - paired.eligible_members_old_membership
    )
    paired["above_delta_repaired_minus_old"] = (
        paired.above_200d_members_repaired_membership
        - paired.above_200d_members_old_membership
    )
    paired["width_delta_pp_repaired_minus_old"] = (
        paired.pct_above_200d_repaired_membership
        - paired.pct_above_200d_old_membership
    ).round(4)
    paired["threshold_flip"] = (
        paired.below_50pct_old_membership != paired.below_50pct_repaired_membership
    )
    paired.to_csv(out_dir / "spy-breadth-paired-membership-diff.csv", index=False)

    member_rows = []
    for dt in old_members:
        for kind in ("active", "eligible", "above"):
            old_set = old_members[dt][kind]
            new_set = new_members[dt][kind]
            for ticker in sorted(old_set - new_set):
                member_rows.append({"spy_source_date": dt, "set": kind, "change": "removed", "ticker": ticker})
            for ticker in sorted(new_set - old_set):
                member_rows.append({"spy_source_date": dt, "set": kind, "change": "added", "ticker": ticker})
    pd.DataFrame(
        member_rows, columns=["spy_source_date", "set", "change", "ticker"]
    ).to_csv(out_dir / "spy-breadth-paired-member-diff.csv", index=False)

    # Compare the old-membership arm with the frozen labels. Any remaining
    # mismatch is not caused by the repaired IR/XEC membership boundaries.
    frozen = baseline.copy()
    frozen["week_label"] = frozen.week_label.dt.strftime("%Y-%m-%d")
    frozen["spy_source_date"] = frozen.spy_source_date.dt.strftime("%Y-%m-%d")
    frozen_cmp = frozen.merge(old_weekly, on=key, suffixes=("_frozen", "_old_membership"), validate="one_to_one")
    frozen_diff = frozen_cmp[
        frozen_cmp.eligible_members_old_membership.astype(int)
        != frozen_cmp.eligible_members_frozen.astype(int)
    ]
    print(f"PAIRED_WEEKS {len(paired)}")
    print(f"MEMBERSHIP_EFFECT eligible_weeks={(paired.eligible_delta_repaired_minus_old != 0).sum()}")
    print(f"MEMBERSHIP_EFFECT above_weeks={(paired.above_delta_repaired_minus_old != 0).sum()}")
    print(f"MEMBERSHIP_EFFECT width_weeks={(paired.width_delta_pp_repaired_minus_old.abs() > 0.00005).sum()}")
    print(f"MEMBERSHIP_EFFECT threshold_flips={int(paired.threshold_flip.sum())}")
    print(f"FROZEN_VS_OLD_MEMBERSHIP eligible_mismatch_weeks={len(frozen_diff)}")
    if len(frozen_diff):
        cols = ["week_label", "eligible_members_frozen", "eligible_members_old_membership"]
        print(f"FROZEN_DENOMINATOR_DIFF {frozen_diff[cols].to_dict('records')}")
        for row in frozen_diff.itertuples(index=False):
            dt_ts = pd.Timestamp(row.spy_source_date).normalize()
            dt = dt_ts.date().isoformat()
            active = set(old_intervals.loc[
                (old_intervals.start <= dt_ts)
                & (old_intervals.end.isna() | (old_intervals.end > dt_ts)),
                "ticker",
            ].astype(str))
            day = prices.loc[prices.date.dt.normalize() == dt_ts]
            available_ma = set(day.loc[
                day.adj_close.notna() & day.ma200.notna(), "ticker"
            ].astype(str))
            active_without_ma = sorted(active - available_ma)
            print(
                f"FROZEN_MEMBER_DIFF {dt} "
                f"frozen={int(row.eligible_members_frozen)} "
                f"old_membership={int(row.eligible_members_old_membership)}"
            )
            print(f"ACTIVE_WITHOUT_MA {dt} {active_without_ma}")

    print("PASS: paired counterfactual completed with identical prices/no-trade/MA200 inputs")
    print("INFO: original-membership arm uses the pinned 2026-11-05 membership SHA")
    print("INFO: no frozen fixture or production data file was modified")


if __name__ == "__main__":
    main()
