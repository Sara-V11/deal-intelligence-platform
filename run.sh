#!/bin/bash
set -e
echo "=== Deal Intelligence Platform — Full Pipeline ==="
echo "No external APIs needed for data. Optional Groq API for memos."
echo ""
echo "[1/5] Load S&P 500 financials...";  python src/data_loader.py
echo "[2/5] Build M&A labels...";         python src/ma_deals.py
echo "[3/5] Feature engineering...";      python notebooks/02_feature_engineering.py
echo "[4/5] Train ML models...";          python notebooks/03_train_models.py
echo "[5/5] Temporal backtest...";        python notebooks/04_backtest.py
echo ""
echo "=== Pipeline complete! Launch dashboard with: ==="
echo "    streamlit run app/dashboard.py"
echo ""
echo "For LLM memos: add GROQ_API_KEY to .env (free at console.groq.com/keys)"
