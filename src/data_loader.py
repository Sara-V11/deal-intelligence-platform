"""
data_loader.py  —  Phase 1: Load, clean, store S&P 500 financial data.

Input:  data/sp500_financials.csv  (from Kaggle)
Output: data/companies.db           (SQLite)
        data/universe.csv           (reference file)

Run from project ROOT:
    python src/data_loader.py
"""

import os, sys, sqlite3, logging
import pandas as pd
import numpy as np
from datetime import datetime

os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("logs/loader.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

RAW_CSV = "data/sp500_financials.csv"
DB_PATH = "data/companies.db"


def clean_column(name):
    """Convert 'Price/Earnings' -> 'pe_ratio' style column names."""
    mapping = {
        "Symbol":          "ticker",
        "Name":            "name",
        "Sector":          "sector",
        "Price":           "price",
        "Price/Earnings":  "pe_ratio",
        "Dividend Yield":  "dividend_yield",
        "Earnings/Share":  "eps",
        "52 Week Low":     "week52_low",
        "52 Week High":    "week52_high",
        "Market Cap":      "market_cap",
        "EBITDA":          "ebitda",
        "Price/Sales":     "ps_ratio",
        "Price/Book":      "pb_ratio",
        "SEC Filings":     "sec_filings_url",
    }
    return mapping.get(name, name.lower().replace("/", "_").replace(" ", "_"))


def load_and_clean(path):
    log.info(f"Loading {path}")
    df = pd.read_csv(path)
    log.info(f"  Raw shape: {df.shape}")

    # Rename columns
    df = df.rename(columns={c: clean_column(c) for c in df.columns})

    # Drop URL (not needed for analysis)
    if "sec_filings_url" in df.columns:
        df = df.drop(columns=["sec_filings_url"])

    # Strip whitespace
    df["ticker"] = df["ticker"].str.strip()
    df["name"]   = df["name"].str.strip()
    df["sector"] = df["sector"].str.strip()

    # Coerce numerics
    numeric_cols = ["price","pe_ratio","dividend_yield","eps",
                    "week52_low","week52_high","market_cap","ebitda",
                    "ps_ratio","pb_ratio"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows without a ticker
    df = df.dropna(subset=["ticker"])
    df = df.drop_duplicates(subset=["ticker"])

    # Add derived metrics
    df["market_cap_bn"]   = df["market_cap"] / 1e9
    df["ebitda_bn"]       = df["ebitda"] / 1e9
    df["ev_ebitda_proxy"] = df["market_cap"] / df["ebitda"]
    df["price_range_pct"] = ((df["price"] - df["week52_low"])
                             / (df["week52_high"] - df["week52_low"]) * 100)

    # Market cap tier
    def tier(mc):
        if pd.isna(mc): return "Unknown"
        if mc < 2e9:    return "Small-cap"
        if mc < 10e9:   return "Mid-cap"
        if mc < 200e9:  return "Large-cap"
        return "Mega-cap"
    df["market_cap_tier"] = df["market_cap"].apply(tier)

    df["loaded_at"] = datetime.utcnow().isoformat()

    log.info(f"  Clean shape: {df.shape}")
    log.info(f"  Sectors: {df['sector'].nunique()}")
    log.info(f"  Missing values: {df.isna().sum().sum()}")
    return df


def save_to_db(df, db_path):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS companies")
    cur.execute("""
        CREATE TABLE companies (
            ticker             TEXT PRIMARY KEY,
            name               TEXT,
            sector             TEXT,
            price              REAL,
            pe_ratio           REAL,
            dividend_yield     REAL,
            eps                REAL,
            week52_low         REAL,
            week52_high        REAL,
            market_cap         REAL,
            ebitda             REAL,
            ps_ratio           REAL,
            pb_ratio           REAL,
            market_cap_bn      REAL,
            ebitda_bn          REAL,
            ev_ebitda_proxy    REAL,
            price_range_pct    REAL,
            market_cap_tier    TEXT,
            loaded_at          TEXT
        )
    """)
    conn.commit()

    df.to_sql("companies", conn, if_exists="append", index=False)
    conn.commit()

    count = cur.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    log.info(f"  Saved {count} companies to {db_path}")
    conn.close()


def summary(df):
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"\nTotal companies:  {len(df)}")
    print(f"Total sectors:    {df['sector'].nunique()}")
    print(f"Combined market cap: ${df['market_cap'].sum()/1e12:.2f}T")

    print("\nCompanies per sector:")
    print(df.groupby("sector").agg(
        companies=("ticker", "count"),
        total_mcap_bn=("market_cap_bn", lambda x: round(x.sum(), 0)),
        median_pe=("pe_ratio", lambda x: round(x.median(), 1)),
    ).sort_values("total_mcap_bn", ascending=False).to_string())

    print("\nMarket cap tiers:")
    print(df["market_cap_tier"].value_counts().to_string())

    print("\nKey metric medians:")
    print(f"  P/E ratio:        {df['pe_ratio'].median():.1f}")
    print(f"  P/B ratio:        {df['pb_ratio'].median():.2f}")
    print(f"  P/S ratio:        {df['ps_ratio'].median():.2f}")
    print(f"  Dividend yield:   {df['dividend_yield'].median():.2f}%")
    print(f"  EV/EBITDA proxy:  {df['ev_ebitda_proxy'].median():.1f}x")


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Deal Intelligence Platform — Phase 1: Data Loading")
    log.info("=" * 60)

    if not os.path.exists(RAW_CSV):
        log.error(f"File not found: {RAW_CSV}")
        log.error("Download from Kaggle and save as data/sp500_financials.csv")
        sys.exit(1)

    df = load_and_clean(RAW_CSV)
    save_to_db(df, DB_PATH)

    # Export reference CSV (cleaned version)
    df[["ticker","name","sector","market_cap_bn","market_cap_tier"]].to_csv(
        "data/universe.csv", index=False
    )
    log.info(f"  Exported data/universe.csv")

    summary(df)

    print("\nNext steps:")
    print("  1. python src/ma_deals.py                (build M&A labels)")
    print("  2. python notebooks/02_feature_engineering.py")
    print("  3. python notebooks/03_train_models.py")
    print("  4. streamlit run app/dashboard.py")
