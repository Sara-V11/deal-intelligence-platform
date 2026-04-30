"""
03_train_models.py  —  Phase 3 (v2, improved).

Key improvements over v1:
  1. Train on LABELED subset only (not the full 505 — unlabeled != not acquired)
  2. Logistic Regression + XGBoost ensemble comparison
  3. Hyperparameter tuning via GridSearchCV
  4. Proper feature scaling
  5. Then score ALL companies with the trained model

Run from project ROOT:
    python notebooks/03_train_models.py
"""

import os, sys, pickle, logging, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import shap

os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs",    exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_data():
    df = pd.read_csv("data/features.csv")
    with open("data/feature_cols.txt") as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    feature_cols = [c for c in feature_cols if c in df.columns]

    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    X_all = df[feature_cols].copy()
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in X_all.columns:
        if X_all[col].dtype == bool:
            X_all[col] = X_all[col].astype(int)

    y_all = df["acquired"].astype(int)

    # KEY FIX: only train on actually labeled companies
    labeled_mask = df["deal_status"].isin(["completed", "rumoured", "none"])
    X_train = X_all[labeled_mask].reset_index(drop=True)
    y_train = y_all[labeled_mask].reset_index(drop=True)

    log.info(f"Total companies: {len(df)}")
    log.info(f"Labeled companies (used for training): {len(X_train)}")
    log.info(f"  Positive (acquired): {y_train.sum()}")
    log.info(f"  Negative (stable):   {(y_train==0).sum()}")
    log.info(f"Features: {len(feature_cols)}")

    return df, X_all, y_all, X_train, y_train, feature_cols


def compare_models(X, y):
    log.info("\n" + "=" * 60)
    log.info("Model comparison (5-fold stratified CV)")
    log.info("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=(y==0).sum()/max(y.sum(),1),
            eval_metric="auc", random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = scores
        log.info(f"  {name:20s}  AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    best = max(results.keys(), key=lambda k: results[k].mean())
    log.info(f"\nBest model: {best}  (CV AUC = {results[best].mean():.3f})")
    return best, models[best]


def tune_xgboost(X, y):
    log.info("\n" + "=" * 60)
    log.info("Hyperparameter tuning — XGBoost")
    log.info("=" * 60)

    param_grid = {
        "n_estimators":    [200, 400],
        "max_depth":       [3, 5],
        "learning_rate":   [0.03, 0.07],
        "subsample":       [0.8, 0.95],
    }
    base = xgb.XGBClassifier(
        scale_pos_weight=(y==0).sum()/max(y.sum(),1),
        eval_metric="auc", random_state=42, n_jobs=-1,
    )
    gs = GridSearchCV(
        base, param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1,
    )
    gs.fit(X, y)
    log.info(f"  Best params: {gs.best_params_}")
    log.info(f"  Best CV AUC: {gs.best_score_:.3f}")
    return gs.best_estimator_, gs.best_score_


def evaluate_holdout(model, X, y):
    log.info("\n" + "=" * 60)
    log.info("Hold-out evaluation (20% test set)")
    log.info("=" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_te, y_prob)
    log.info(f"  Test AUC: {auc:.3f}")
    log.info("\n" + classification_report(y_te, y_pred))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#185FA5", lw=2.5, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], "k--", alpha=0.4, label="Random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Acquisition probability model — ROC curve")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("  Saved reports/roc_curve.png")

    return auc


def shap_plots(model, X, feature_cols):
    log.info("\nGenerating SHAP explainability")
    if not isinstance(model, xgb.XGBClassifier):
        log.info("  Skipping SHAP (only supported for XGBoost)")
        return

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout(); plt.savefig("reports/shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig("reports/shap_bar.png", dpi=120, bbox_inches="tight"); plt.close()
    log.info("  Saved reports/shap_summary.png and reports/shap_bar.png")


def compute_deal_score(df, prob_col="deal_probability"):
    df = df.copy()
    ml_score    = df[prob_col] * 100
    val_score   = (df["cheap_count"] / 4) * 100
    size_score  = np.clip(df["log_market_cap"] / df["log_market_cap"].max() * 100, 0, 100)
    yield_score = np.clip(df["dividend_yield"] * 10, 0, 100)

    df["deal_score"] = (
        0.55 * ml_score +
        0.25 * val_score +
        0.10 * size_score +
        0.10 * yield_score
    ).round(1)
    return df


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Phase 3 (v2): Model Training & Scoring")
    log.info("=" * 60)

    df_all, X_all, y_all, X_train, y_train, feature_cols = load_data()

    best_name, best_model = compare_models(X_train, y_train)

    # Tune if XGBoost is best
    if best_name == "XGBoost":
        final_model, final_cv_auc = tune_xgboost(X_train, y_train)
    else:
        final_model, final_cv_auc = best_model, None

    holdout_auc = evaluate_holdout(final_model, X_train, y_train)

    # Fit on full labeled training set for final model
    final_model.fit(X_train, y_train)

    if isinstance(final_model, xgb.XGBClassifier):
        shap_plots(final_model, X_train, feature_cols)

    # Score ALL companies (including unlabeled ones)
    df_all["deal_probability"] = final_model.predict_proba(X_all)[:, 1]
    df_scored = compute_deal_score(df_all)

    # Save artifacts
    with open("models/classifier.pkl", "wb") as f: pickle.dump(final_model, f)
    with open("models/feature_cols.pkl", "wb") as f: pickle.dump(feature_cols, f)
    df_scored.to_csv("data/scored_companies.csv", index=False)

    log.info(f"\nSaved models/classifier.pkl")
    log.info(f"Saved data/scored_companies.csv ({len(df_scored)} rows)")

    print("\n" + "=" * 85)
    print("TOP 20 M&A TARGETS BY DEAL SCORE")
    print("=" * 85)
    cols = ["ticker","name","sector","market_cap_bn","pe_ratio","pb_ratio",
            "dividend_yield","deal_probability","deal_score"]
    top = df_scored.sort_values("deal_score", ascending=False).head(20)[cols].copy()
    top["deal_probability"] = (top["deal_probability"] * 100).round(1)
    for c in ["market_cap_bn","pe_ratio","pb_ratio","dividend_yield"]:
        top[c] = top[c].round(2)
    print(top.to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"FINAL MODEL: {best_name}")
    if final_cv_auc:
        print(f"  Tuned CV AUC:  {final_cv_auc:.3f}")
    print(f"  Holdout AUC:   {holdout_auc:.3f}")
    print(f"{'=' * 60}")
    print("\nNext: streamlit run app/dashboard.py")
