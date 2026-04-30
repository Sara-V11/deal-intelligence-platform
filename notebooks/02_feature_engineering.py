"""
02_feature_engineering.py  —  Phase 2 (v2, improved).

Adds features that are actually discriminative for M&A targeting:
  - Relative valuation percentiles within sector
  - Z-scores of key multiples (how far from peer median)
  - Size ranking within sector
  - Interaction features (cheap AND small = classic PE target)
  - Dividend-to-size flag (mature cash cow signal)

Run from project ROOT:
    python notebooks/02_feature_engineering.py
"""

import os, sys, logging
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("logs/features.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def sector_median_impute(df, cols):
    for col in cols:
        if col not in df.columns: continue
        sec = df.groupby("sector")[col].transform("median")
        df[col] = df[col].fillna(sec).fillna(df[col].median())
    return df


def winsorize(df, cols, low=0.01, high=0.99):
    for col in cols:
        if col not in df.columns: continue
        lo, hi = df[col].quantile([low, high])
        df[col] = df[col].clip(lo, hi)
    return df


def engineer(df):
    log.info(f"Input: {df.shape}")

    numeric = ["price","pe_ratio","dividend_yield","eps","market_cap",
               "ebitda","ps_ratio","pb_ratio","ev_ebitda_proxy","price_range_pct"]

    # 1. Impute and clean
    df = sector_median_impute(df, numeric)
    df = winsorize(df, ["pe_ratio","ps_ratio","pb_ratio","ev_ebitda_proxy"])

    # 2. Sector-relative percentile ranks (0-1, lower = cheaper/smaller)
    for col in ["pe_ratio","pb_ratio","ps_ratio","ev_ebitda_proxy","market_cap","dividend_yield"]:
        df[f"{col}_sector_pct"] = df.groupby("sector")[col].rank(pct=True)

    # 3. Z-scores within sector (how many standard deviations from peer mean)
    for col in ["pe_ratio","pb_ratio","ps_ratio","ev_ebitda_proxy"]:
        sec_mean = df.groupby("sector")[col].transform("mean")
        sec_std  = df.groupby("sector")[col].transform("std")
        df[f"{col}_zscore"] = ((df[col] - sec_mean) / sec_std).replace([np.inf, -np.inf], 0).fillna(0)

    # 4. Valuation composite: how many multiples are below sector median?
    df["cheap_count"] = (
        (df["pe_ratio_sector_pct"] < 0.5).astype(int) +
        (df["pb_ratio_sector_pct"] < 0.5).astype(int) +
        (df["ps_ratio_sector_pct"] < 0.5).astype(int) +
        (df["ev_ebitda_proxy_sector_pct"] < 0.5).astype(int)
    )

    # 5. Log transforms
    df["log_market_cap"] = np.log1p(df["market_cap"].clip(lower=0))
    df["log_ebitda"]     = np.log1p(df["ebitda"].clip(lower=0))

    # 6. Size × valuation interactions (classic PE buyout target = mid-cap + cheap)
    df["cheap_and_midcap"] = (
        (df["cheap_count"] >= 2) &
        (df["market_cap_bn"] >= 2) &
        (df["market_cap_bn"] <= 50)
    ).astype(int)

    # 7. Mature cash cow signal (high dividend + moderate valuation)
    df["cash_cow_flag"] = (
        (df["dividend_yield"] > df["dividend_yield"].median()) &
        (df["pe_ratio"] < df["pe_ratio"].quantile(0.7))
    ).astype(int)

    # 8. Price momentum signal
    df["near_52w_low"]  = (df["price_range_pct"] < 30).astype(int)
    df["near_52w_high"] = (df["price_range_pct"] > 70).astype(int)

    # 9. Earnings quality proxy (positive EPS)
    df["profitable"] = (df["eps"] > 0).astype(int)

    # 10. Low P/B × small size (book value undervaluation + takeover accessibility)
    df["low_pb_small_size"] = (
        (df["pb_ratio"] < 2.0) & (df["market_cap_bn"] < 30)
    ).astype(int)

    # 11. Sector dummies
    df = pd.concat([df, pd.get_dummies(df["sector"], prefix="sector")], axis=1)
    # 12. Tier dummies
    df = pd.concat([df, pd.get_dummies(df["market_cap_tier"], prefix="tier")], axis=1)

    log.info(f"Output: {df.shape}")
    return df


def select_feature_columns(df):
    base = [
        # Raw multiples
        "pe_ratio","pb_ratio","ps_ratio","ev_ebitda_proxy","dividend_yield","eps",
        # Log sizes
        "log_market_cap","log_ebitda","price_range_pct",
        # Sector percentile ranks (key!)
        "pe_ratio_sector_pct","pb_ratio_sector_pct","ps_ratio_sector_pct",
        "ev_ebitda_proxy_sector_pct","market_cap_sector_pct","dividend_yield_sector_pct",
        # Z-scores
        "pe_ratio_zscore","pb_ratio_zscore","ps_ratio_zscore","ev_ebitda_proxy_zscore",
        # Composite flags
        "cheap_count","cheap_and_midcap","cash_cow_flag",
        "near_52w_low","near_52w_high","profitable","low_pb_small_size",
    ]
    dummies = [c for c in df.columns if c.startswith("sector_") or c.startswith("tier_")]
    return [c for c in base if c in df.columns] + dummies


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Phase 2 (v2): Feature Engineering")
    log.info("=" * 60)

    if not os.path.exists("data/training_data.csv"):
        log.error("Run src/ma_deals.py first"); sys.exit(1)

    df = pd.read_csv("data/training_data.csv")
    df = engineer(df)

    feature_cols = select_feature_columns(df)
    log.info(f"\nFeature count: {len(feature_cols)}")

    # Final sanitization
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().any():
            sec = df.groupby("sector")[col].transform("median")
            df[col] = df[col].fillna(sec).fillna(df[col].median()).fillna(0)

    df.to_csv("data/features.csv", index=False)
    log.info(f"Saved data/features.csv ({df.shape[0]} × {df.shape[1]})")

    with open("data/feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))
    log.info(f"Saved data/feature_cols.txt ({len(feature_cols)} features)")
