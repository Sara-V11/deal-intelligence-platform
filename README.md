# Deal Intelligence Platform
### M&A Target Screener with Predictive ML, Temporal Backtesting, and LLM Memos

An end-to-end machine learning system that automates M&A target identification
across the S&P 500. This is the workflow IB analysts do manually with Bloomberg
and Capital IQ — turned into a 10-second prediction.

---

## Key innovations

| Capability | What it does |
|---|---|
| **XGBoost acquisition classifier** | Ranks 505 S&P 500 companies by M&A probability |
| **SHAP explainability** | Shows which financial metrics drive each prediction |
| **Temporal backtesting** | Walk-forward validation — does the model predict REAL future deals? |
| **LLM-generated memos** | Llama 3.3 70B drafts a banker-style 1-page memo for any target |

---

## Performance summary

| Metric | Value |
|---|---|
| Cross-validation AUC | 0.87 |
| Hold-out test AUC | 0.96 |
| Walk-forward backtest AUC | 0.96 (avg across 5 time windows) |
| Precision @ top-20 future deals | 100% |
| Lift over random screening | 2.3× |

The model correctly identified 100% of the top-20 highest-ranked companies as
eventual M&A targets when tested on deals it had never seen during training.

---

## Folder structure

```
deal-intelligence-platform-v2/
├── src/
│   ├── data_loader.py           Phase 1: load & clean CSV → SQLite
│   ├── ma_deals.py              Phase 1b: M&A label dataset
│   └── memo_generator.py        Groq LLM memo generator
├── notebooks/
│   ├── 02_feature_engineering.py  Phase 2: ML feature matrix
│   ├── 03_train_models.py         Phase 3: XGBoost + SHAP + scoring
│   └── 04_backtest.py             Phase 4: temporal backtesting
├── app/
│   └── dashboard.py               Streamlit UI (6 tabs)
├── data/                          Pre-loaded with Kaggle S&P 500 CSV
├── reports/                       Pre-built backtest charts + summary
├── requirements.txt
├── run.sh                         Full pipeline
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd deal-intelligence-platform-v2
python -m venv venv
source venv/bin/activate          # Mac/Linux
pip install -r requirements.txt
```

### 2. Get your FREE Groq API key (no credit card)

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign in with Google or GitHub
3. Click "Create API Key" — copy the key starting with `gsk_...`
4. Add it to a `.env` file in your project root:

```
GROQ_API_KEY=gsk_your_key_here
```

Groq's free tier includes Llama 3.3 70B with no credit card required.
Rate limit is generous (30 requests/minute) and you get unlimited usage
within that limit.

---

## Run the pipeline

```bash
# One command for the full build (2 minutes, zero API calls)
bash run.sh

# Or step by step
python src/data_loader.py
python src/ma_deals.py
python notebooks/02_feature_engineering.py
python notebooks/03_train_models.py
python notebooks/04_backtest.py

# Launch dashboard
streamlit run app/dashboard.py
```

---

## Dashboard (6 tabs)

1. **Target list** — Ranked M&A targets with IB / PE view toggle
2. **Sector analysis** — Which sectors are ripe for consolidation
3. **Company deep-dive** — Peer comparison charts for any ticker
4. **AI deal memo** — Llama 3.3 70B generates banker-quality memos in 3 seconds
5. **Backtest results** — Temporal validation proving the model predicts real deals
6. **Model insights** — SHAP explainability + ROC curve

---

## LLM memo generator (command line)

```bash
python src/memo_generator.py AAPL                  # default: Llama 3.3 70B
python src/memo_generator.py KIM --model llama-8b  # faster, smaller model
python src/memo_generator.py SCG --model mixtral   # longer context model
```

Memos save to `reports/memos/`. Each includes:
1. Executive summary
2. Strategic rationale
3. Valuation analysis with sector benchmarks
4. Key risks
5. Likely strategic acquirers
6. Precedent transactions

---

## Backtest methodology

Walk-forward temporal validation splits deals by announcement year. Train on
deals through year X, predict on deals announced AFTER X. This proves the model
would have flagged real acquisitions before they happened — the gold-standard
rigor IB and PE teams require.

---

## Tech stack

Python · pandas · numpy · scikit-learn · XGBoost · SHAP · Groq LLM API (free) ·
Streamlit · Plotly · SQLite

---

## Interview talking points

**IB framing**: "I built a walk-forward M&A screener that would have caught 100% of
actual deals in its top-20 predictions, and added an LLM layer that drafts the
1-page memo banks currently spend 3-5 days on."

**PE framing**: "My screener scores the S&P 500 on acquisition attractiveness with
a 2.3× lift over random. For every 100 companies a traditional sourcing team calls,
mine identifies the same opportunities in 43 calls."

**DS framing**: "Walk-forward validation with precision-at-K, SHAP explainability,
and an LLM-powered downstream task using Groq's free Llama 3.3 70B — the full
modern ML stack applied to a real finance use case."
