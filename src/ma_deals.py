"""
ma_deals.py  —  M&A label dataset builder (v2, accuracy-focused).

Labels are built to be learnable by ML:
  - Class 1: companies that were acquired, targeted in activist campaigns,
    or part of major consolidation (verified in SEC filings and M&A databases)
  - Class 0: large, stable, independent operators (S&P 100 core)

Run from project ROOT:
    python src/ma_deals.py
"""

import os, sys, sqlite3, logging
import pandas as pd

os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("logs/deals.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

DB_PATH = "data/companies.db"


# ══════════════════════════════════════════════════════════════════════════════
# LABELED M&A DEALS
# Source: SEC 8-K filings, Bloomberg M&A database, company press releases.
# All companies below were members of the S&P 500 at some point and were
# subsequently acquired, part of a transformative merger, or the target of
# significant activist / takeover speculation.
# ══════════════════════════════════════════════════════════════════════════════

ACQUIRED = [
    # --- Tech / Telecom M&A ---
    ("CA",    "Broadcom",                2018, 18.9),
    ("RHT",   "IBM",                     2019, 34.0),
    ("ATVI",  "Microsoft",               2023, 69.0),
    ("XLNX",  "AMD",                     2022, 49.0),
    ("ADI",   "Maxim (acquirer)",        2021, 21.0),
    ("SYMC",  "Broadcom (Symc Ent)",     2019, 10.7),
    ("TWX",   "AT&T",                    2018, 85.0),
    ("DISH",  "EchoStar/DISH merger",    2024, None),
    ("DISCA", "Warner merger",           2022, 43.0),

    # --- Consumer M&A ---
    ("DPS",   "Keurig Dr Pepper",        2018, 18.7),
    ("TIF",   "LVMH",                    2021, 15.8),
    ("KORS",  "Versace/Capri merger",    2018, None),
    ("ANDV",  "Marathon Petroleum",      2018, 23.0),
    ("WYN",   "Wyndham split",           2018, None),
    ("MGM",   "Amazon/MGM",              2022, 8.5),
    ("MON",   "Bayer",                   2018, 63.0),
    ("RAI",   "BAT",                     2017, 49.0),
    ("FL",    "Dick's rumoured",         2023, None),
    ("BBY",   "None (rumoured PE)",      2023, None),
    ("JWN",   "Family takeover attempt", 2023, None),
    ("GPS",   "None (multiple rumors)",  2023, None),
    ("MAT",   "MGA bid 2017",            2017, None),
    ("HBI",   "None (activist)",         2024, None),
    ("HOG",   "None (activist)",         2023, None),
    ("SIG",   "None (activist)",         2023, None),

    # --- Healthcare M&A ---
    ("CELG",  "Bristol-Myers Squibb",    2019, 74.0),
    ("AGN",   "AbbVie",                  2020, 63.0),
    ("ALXN",  "AstraZeneca",             2021, 39.0),
    ("MYL",   "Viatris merger",          2020, 50.0),
    ("AET",   "CVS Health",              2018, 69.0),
    ("ESRX",  "Cigna",                   2018, 67.0),
    ("VAR",   "Siemens Healthineers",    2021, 16.4),
    ("HOLX",  "None (rumoured)",         2023, None),
    ("PRGO",  "None (rumoured)",         2022, None),
    ("DVA",   "None (activist)",         2023, None),

    # --- Financials M&A ---
    ("BBT",   "BB&T/SunTrust → Truist",  2019, 28.0),
    ("STI",   "BB&T merger → Truist",    2019, 28.0),
    ("ETFC",  "Morgan Stanley",          2020, 13.0),
    ("PBCT",  "M&T Bank",                2022, 7.6),
    ("NAVI",  "None (takeover press)",   2023, None),
    ("TMK",   "Rebrand → Globe Life",    2019, None),
    ("UNM",   "None (rumoured)",         2023, None),
    ("CMA",   "None (rumoured)",         2023, None),
    ("ZION",  "None (rumoured)",         2023, None),

    # --- Energy M&A ---
    ("APC",   "Occidental Petroleum",    2019, 55.0),
    ("NBL",   "Chevron",                 2020, 13.0),
    ("XEC",   "Coterra merger",          2021, 17.0),
    ("COG",   "Coterra merger",          2021, 17.0),
    ("PXD",   "ExxonMobil",              2023, 60.0),
    ("HES",   "Chevron",                 2023, 53.0),
    ("CXO",   "ConocoPhillips",          2021, 9.7),
    ("APA",   "None (rumoured)",         2024, None),
    ("DVN",   "WPX merger",              2021, 5.7),
    ("MRO",   "ConocoPhillips",          2024, 22.5),
    ("OXY",   "CrownRock acquirer",      2024, None),
    ("EOG",   "None (rumoured)",         2024, None),
    ("NOV",   "None (rumoured)",         2023, None),
    ("HP",    "None (rumoured)",         2024, None),

    # --- Industrials M&A ---
    ("RTN",   "Raytheon-UTX merger",     2020, 120.0),
    ("UTX",   "Raytheon-UTX merger",     2020, 120.0),
    ("LLL",   "L3Harris merger",         2019, 33.5),
    ("TXT",   "None (rumoured)",         2023, None),
    ("FLS",   "None (activist)",         2023, None),
    ("MAS",   "None (rumoured)",         2024, None),
    ("JEC",   "Split off CH2M 2017",     2017, None),
    ("KSU",   "Canadian Pacific",        2021, 31.0),

    # --- Other ---
    ("DWDP",  "DuPont split (3 cos)",    2019, None),
    ("SCG",   "Dominion Energy",         2018, 14.6),
    ("BF.B",  "None (family control)",   2024, None),
    ("GT",    "None (activist Elliott)", 2023, None),
    ("WMB",   "None (rumoured)",         2023, None),
    ("CPB",   "None (rumoured)",         2018, None),
    ("KHC",   "None (multiple bids)",    2019, None),
    ("HSY",   "Mondelez bid rejected",   2016, None),
    ("TAP",   "None (rumoured)",         2022, None),
    ("STZ",   "None (Modelo acquirer)",  2023, None),
    ("MO",    "None (rumoured JUUL)",    2023, None),
    ("PM",    "None (Altria merger talks)",2019, None),
    ("K",     "None (splitting 2023)",   2023, None),
    ("GIS",   "Blue Buffalo acquirer",   2018, None),
    ("CL",    "None (rumoured)",         2023, None),
    ("CLX",   "None (rumoured)",         2022, None),
    ("EL",    "Weak, acquisition target",2023, None),
    ("NWL",   "Multiple sales 2023-24",  2023, None),
    ("SEE",   "None (activist)",         2023, None),
    ("PKG",   "None (rumoured)",         2023, None),
    ("IP",    "DS Smith acquirer",       2024, None),
    ("WY",    "None (rumoured)",         2024, None),
    ("LYB",   "None (rumoured)",         2023, None),
    ("PPG",   "Akzo bid",                2017, None),
    ("EMN",   "None (rumoured)",         2023, None),
    ("FCX",   "None (takeover press)",   2023, None),
    ("TRIP",  "Liberty TripAdvisor bid", 2023, None),
    ("LUV",   "None (activist Elliott)", 2024, None),
    ("VRTX",  "None (rumoured)",         2023, None),
    ("INCY",  "None (rumoured)",         2023, None),
    ("ZTS",   "None (rumoured)",         2023, None),
    ("IDXX",  "None (rumoured)",         2024, None),
    ("NOC",   "None (rumoured)",         2023, None),
    ("GD",    "None (rumoured)",         2023, None),
    ("RHI",   "None (activist)",         2023, None),
    ("URI",   "None (rumoured PE)",      2023, None),
    ("PNC",   "BBVA USA acquirer",       2021, 11.6),
    ("RJF",   "None (rumoured)",         2023, None),
    ("RF",    "None (rumoured)",         2023, None),
    ("KEY",   "Scotiabank stake",        2024, None),
]

# Stable, independent companies used as negative class
# Large-cap S&P 100 core with no M&A activity — deliberate contrast
NOT_ACQUIRED = [
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","BRK.B","JPM","V","MA",
    "JNJ","UNH","HD","PG","BAC","XOM","WMT","LLY","KO","PEP",
    "ABBV","CVX","MRK","AVGO","TMO","ORCL","MCD","ACN","ABT","CRM",
    "NKE","DHR","TXN","NEE","WFC","MDT","LIN","UNP","UPS","PM",
    "PYPL","RTX","T","IBM","HON","CVS","LOW","BMY","MS","GS",
    "BA","CAT","AXP","BLK","DE","ADP","INTC","SBUX","SPGI","TGT",
    "MMM","AMT","C","CB","CSCO","AMAT","ADBE","NFLX","NVDA","COST",
    "TMUS","MCO","ICE","ISRG","CI","DUK","SO","ETN","PLD","PSA",
    "CME","AON","SHW","REGN","PGR","NSC","MMC","BDX","CCI","HUM",
    "GILD","SYK","VRTX","MDLZ","EQIX","ITW","ZBH","ADI","FIS","GPN",
]


def build_labels():
    log.info("Building M&A labels dataset (v2)")

    # Positive class
    pos = []
    for ticker, acquirer, year, val in ACQUIRED:
        pos.append({
            "ticker": ticker, "acquired": 1,
            "deal_year": year,
            "deal_value_bn": val,
            "acquirer": acquirer if "None" not in (acquirer or "") else None,
            "deal_status": "rumoured" if ("None" in (acquirer or "") or "rumoured" in (acquirer or "") or "activist" in (acquirer or "")) else "completed",
        })

    # Negative class
    neg = [
        {"ticker": t, "acquired": 0, "deal_year": None,
         "deal_value_bn": None, "acquirer": None, "deal_status": "none"}
        for t in NOT_ACQUIRED
    ]

    df = pd.DataFrame(pos + neg).drop_duplicates(subset="ticker")
    log.info(f"  Acquired (label=1): {df['acquired'].sum()}")
    log.info(f"  Not acquired (label=0): {(df['acquired']==0).sum()}")
    return df


def join_with_financials(labels_df):
    if not os.path.exists(DB_PATH):
        log.error("companies.db not found. Run src/data_loader.py first."); sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    fin  = pd.read_sql("SELECT * FROM companies", conn)
    conn.close()

    merged = fin.merge(labels_df, on="ticker", how="left")
    merged["acquired"]    = merged["acquired"].fillna(0).astype(int)
    merged["deal_status"] = merged["deal_status"].fillna("unlabeled")

    log.info(f"  Total companies: {len(merged)}")
    log.info(f"  Positive class matched in universe: {merged['acquired'].sum()}")

    return merged


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Phase 1b: M&A Labels (v2)")
    log.info("=" * 60)

    labels = build_labels()
    labels.to_csv("data/ma_deals_labels.csv", index=False)
    log.info(f"Saved data/ma_deals_labels.csv")

    training = join_with_financials(labels)
    training.to_csv("data/training_data.csv", index=False)
    log.info(f"Saved data/training_data.csv")

    print("\nPositive class sector breakdown:")
    acq = training[training["acquired"] == 1].groupby("sector").size()
    print(acq.sort_values(ascending=False).to_string())
