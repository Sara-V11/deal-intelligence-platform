"""
dashboard.py  —  Streamlit dashboard with AI memo generator + backtest.

Run from project ROOT:
    streamlit run app/dashboard.py
"""

import os, sys
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Make src.memo_generator importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Deal Intelligence Platform",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(PROJECT_ROOT, "data", "scored_companies.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_backtest_results():
    path = os.path.join(PROJECT_ROOT, "reports", "backtest_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


df = load_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Deal Intelligence Platform")
st.caption("M&A target screener for S&P 500 · ML + backtesting + LLM memos (Groq)")
st.divider()


if df is None:
    st.warning("No scored data found. Run the full pipeline first:")
    st.code("""python src/data_loader.py
python src/ma_deals.py
python notebooks/02_feature_engineering.py
python notebooks/03_train_models.py
python notebooks/04_backtest.py""", language="bash")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Analyst view")
view_mode = st.sidebar.radio(
    "Perspective",
    ["Investment Banking", "Private Equity", "All targets"],
    index=0,
)

st.sidebar.divider()
st.sidebar.header("Filters")

sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
sel_sector = st.sidebar.selectbox("Sector", sectors)

tiers = ["All", "Mid-cap", "Large-cap", "Mega-cap"]
sel_tier = st.sidebar.selectbox("Market cap tier", tiers)

min_score = st.sidebar.slider("Min deal score",           0, 100, 30)
min_prob  = st.sidebar.slider("Min acq. probability (%)", 0, 100, 10)
max_pe    = st.sidebar.slider("Max P/E ratio",            0, 100, 50)
max_pb    = st.sidebar.slider("Max P/B ratio",            0,  20, 10)


# ── Filter ────────────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_sector != "All":
    filtered = filtered[filtered["sector"] == sel_sector]
if sel_tier != "All":
    filtered = filtered[filtered["market_cap_tier"] == sel_tier]
filtered = filtered[filtered["deal_score"]           >= min_score]
filtered = filtered[filtered["deal_probability"]*100 >= min_prob]
filtered = filtered[filtered["pe_ratio"]             <= max_pe]
filtered = filtered[filtered["pb_ratio"]             <= max_pb]

if view_mode == "Investment Banking":
    filtered = filtered[filtered["market_cap_bn"] >= 5]
elif view_mode == "Private Equity":
    filtered = filtered[filtered["dividend_yield"] >= 1.0]

filtered = filtered.sort_values("deal_score", ascending=False)


# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe size",        f"{len(df):,}")
c2.metric("Targets after filter", f"{len(filtered):,}")
if len(filtered) > 0:
    c3.metric("Avg acq. probability", f"{filtered['deal_probability'].mean()*100:.1f}%")
    c4.metric("Median P/E",           f"{filtered['pe_ratio'].median():.1f}x")

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_list, tab_sector, tab_dive, tab_memo, tab_backtest, tab_model = st.tabs([
    "Target list", "Sector analysis", "Company deep-dive",
    "AI deal memo", "Backtest results", "Model insights",
])


# ═════════════════════════════ TAB 1: Target list ═════════════════════════════
with tab_list:
    st.subheader(f"Ranked M&A targets — {view_mode}")

    if len(filtered) == 0:
        st.warning("No companies match the current filters. Try loosening the criteria.")
    else:
        # NEW: Top-10 deal score chart
        top10 = filtered.head(10).sort_values("deal_score")
        fig_top = px.bar(
            top10, x="deal_score", y="ticker", orientation="h",
            color="deal_probability", color_continuous_scale="Blues",
            title="Top 10 acquisition targets",
            labels={"deal_score": "Deal score", "ticker": "",
                    "deal_probability": "ML acq. probability"},
            hover_data={"name": True, "sector": True,
                        "market_cap_bn": ":.1f",
                        "pe_ratio": ":.1f"},
            text="deal_score",
        )
        fig_top.update_traces(textposition="outside")
        fig_top.update_layout(height=420)
        st.plotly_chart(fig_top, use_container_width=True)

        # NEW: Distribution chart — deal score distribution colored by sector
        c1, c2 = st.columns(2)
        with c1:
            fig_dist = px.histogram(
                filtered, x="deal_score", color="sector",
                nbins=20, title="Deal score distribution (filtered targets)",
                labels={"deal_score": "Deal score", "count": "# companies"},
            )
            fig_dist.update_layout(height=380, bargap=0.05)
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            fig_scatter = px.scatter(
                filtered, x="pe_ratio", y="deal_probability",
                size="market_cap_bn", color="sector",
                hover_name="ticker",
                hover_data={"name": True, "deal_score": ":.0f"},
                title="P/E vs acquisition probability (size = market cap)",
                labels={"pe_ratio": "P/E ratio",
                        "deal_probability": "Acq. probability"},
                size_max=40,
            )
            fig_scatter.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Original table
    st.markdown("#### Ranked target list")
    cols_map = {
        "ticker": "Ticker", "name": "Company", "sector": "Sector",
        "market_cap_bn": "Mkt cap ($B)",
        "pe_ratio": "P/E", "pb_ratio": "P/B", "ps_ratio": "P/S",
        "dividend_yield": "Div yield %",
        "deal_probability": "Acq. prob.",
        "deal_score": "Deal score",
    }
    cols = {k: v for k, v in cols_map.items() if k in filtered.columns}
    view = filtered[list(cols.keys())].rename(columns=cols).head(50).copy()
    if "Acq. prob." in view.columns:
        view["Acq. prob."] = (view["Acq. prob."] * 100).round(1).astype(str) + "%"
    if "Mkt cap ($B)" in view.columns:
        view["Mkt cap ($B)"] = view["Mkt cap ($B)"].round(1)
    for c in ["P/E", "P/B", "P/S", "Div yield %"]:
        if c in view.columns:
            view[c] = view[c].round(2)
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered list (CSV)",
        filtered.to_csv(index=False),
        "deal_targets.csv",
        "text/csv",
    )

# ═══════════════════════════ TAB 2: Sector analysis ═══════════════════════════
with tab_sector:
    st.subheader("Sector-level M&A attractiveness")

    sec = df.groupby("sector").agg(
        companies=("ticker", "count"),
        acquired_hist=("acquired", "sum"),
        avg_score=("deal_score", "mean"),
        avg_prob=("deal_probability", "mean"),
        median_pe=("pe_ratio", "median"),
        median_pb=("pb_ratio", "median"),
        median_ps=("ps_ratio", "median"),
        median_yield=("dividend_yield", "median"),
        total_mcap_bn=("market_cap_bn", "sum"),
    ).reset_index()
    sec["avg_score"]     = sec["avg_score"].round(1)
    sec["avg_prob"]      = (sec["avg_prob"] * 100).round(1)
    sec["median_pe"]     = sec["median_pe"].round(1)
    sec["median_pb"]     = sec["median_pb"].round(2)
    sec["median_ps"]     = sec["median_ps"].round(2)
    sec["median_yield"]  = sec["median_yield"].round(2)
    sec["total_mcap_bn"] = sec["total_mcap_bn"].round(0)
    sec = sec.sort_values("avg_score", ascending=False)

    # Row 1 — Deal score bar + Valuation scatter
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            sec, x="avg_score", y="sector", orientation="h",
            color="avg_score", color_continuous_scale="Blues",
            title="Average deal score by sector",
            labels={"avg_score": "Avg deal score", "sector": ""},
            text="avg_score",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            sec, x="median_pe", y="avg_prob",
            size="companies", color="sector", hover_name="sector",
            title="Valuation vs acquisition probability",
            labels={"median_pe": "Median P/E (x)",
                    "avg_prob": "Avg acq. probability (%)"},
            size_max=50,
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — NEW: Sector heatmap of valuation multiples
    st.markdown("#### Sector valuation heatmap")
    heat_data = sec[["sector", "median_pe", "median_pb", "median_ps", "median_yield"]].set_index("sector")
    heat_data.columns = ["P/E", "P/B", "P/S", "Yield %"]
    # Normalize each column 0-1 so colors are comparable
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_norm.values,
        x=heat_norm.columns,
        y=heat_norm.index,
        text=heat_data.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale="RdYlGn_r",
        showscale=False,
    ))
    fig_heat.update_layout(height=400,
                            title="Lower = cheaper / more attractive (red); higher = expensive (green)")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Row 3 — NEW: Sector treemap by market cap
    st.markdown("#### Sector market cap treemap (size = $B, color = avg deal score)")
    fig_tree = px.treemap(
        sec, path=["sector"], values="total_mcap_bn",
        color="avg_score", color_continuous_scale="Blues",
        hover_data={"avg_score": ":.1f", "companies": True, "total_mcap_bn": ":,.0f"},
    )
    fig_tree.update_layout(height=450)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("#### Sector summary table")
    st.dataframe(sec, use_container_width=True, hide_index=True)


# ══════════════════════════ TAB 3: Company deep-dive ══════════════════════════
with tab_dive:
    st.subheader("Company deep-dive")
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    sel = st.selectbox("Select a ticker", tickers, key="deepdive_ticker")
    row = df[df["ticker"] == sel]

    if row.empty:
        st.info("No data for this ticker.")
    else:
        row = row.iloc[0]
        st.markdown(f"### {row['name']} ({sel})")
        st.caption(f"{row['sector']} · {row['market_cap_tier']} · ${row['market_cap_bn']:.1f}B market cap")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deal score",       f"{row['deal_score']:.0f}/100")
        c2.metric("Acq. probability", f"{row['deal_probability']*100:.1f}%")
        c3.metric("P/E ratio",        f"{row['pe_ratio']:.1f}x")
        c4.metric("P/B ratio",        f"{row['pb_ratio']:.2f}x")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("P/S ratio",        f"{row['ps_ratio']:.2f}x")
        c6.metric("EV/EBITDA proxy",  f"{row['ev_ebitda_proxy']:.1f}x")
        c7.metric("Dividend yield",   f"{row['dividend_yield']:.2f}%")
        c8.metric("EBITDA",           f"${row['ebitda_bn']:.1f}B")

        # Row 1 — Gauge + Radar
        c1, c2 = st.columns(2)
        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row["deal_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": "Deal attractiveness score",
                    "font": {"size": 16},
                },
                number={
                    "font": {"size": 44},
                    "suffix": "<span style='font-size:18px;color:gray'>/100</span>",
                    "valueformat": ".0f",
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "rgba(255,255,255,0.3)",
                    },
                    "bar": {"color": "#185FA5", "thickness": 0.7},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40],  "color": "rgba(237, 242, 247, 0.15)"},
                        {"range": [40, 70],  "color": "rgba(181, 212, 244, 0.3)"},
                        {"range": [70, 100], "color": "rgba(55, 138, 221, 0.5)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": 70,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=320,
                margin=dict(t=60, b=20, l=40, r=40),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )            

        with c2:
            # NEW: Radar chart — company vs sector median across 5 dimensions
            sector_peers = df[df["sector"] == row["sector"]]
            metrics = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda_proxy", "dividend_yield"]
            metric_labels = ["P/E (lower better)", "P/B (lower better)", "P/S (lower better)",
                              "EV/EBITDA (lower better)", "Div yield (higher better)"]

            # Normalize so all metrics show on same 0-100 scale (cheap = high score)
            sec_min = sector_peers[metrics].min()
            sec_max = sector_peers[metrics].max()

            company_scores = []
            sector_scores  = []
            for m in metrics:
                cmin, cmax = sec_min[m], sec_max[m]
                if cmax == cmin:
                    company_scores.append(50)
                    sector_scores.append(50)
                    continue
                # For valuation multiples, lower is better — invert
                if m == "dividend_yield":
                    company_scores.append((row[m] - cmin) / (cmax - cmin) * 100)
                    sector_scores.append((sector_peers[m].median() - cmin) / (cmax - cmin) * 100)
                else:
                    company_scores.append((cmax - row[m]) / (cmax - cmin) * 100)
                    sector_scores.append((cmax - sector_peers[m].median()) / (cmax - cmin) * 100)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=company_scores + [company_scores[0]],
                theta=metric_labels + [metric_labels[0]],
                fill="toself", name=sel,
                line_color="#185FA5",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=sector_scores + [sector_scores[0]],
                theta=metric_labels + [metric_labels[0]],
                fill="toself", name="Sector median",
                line_color="#A0AEC0", opacity=0.6,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Valuation profile vs sector",
                height=320,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Row 2 — NEW: Peer comparison box plots
        st.markdown(f"#### How {sel} compares to {row['sector']} peers")
        sector_peers = df[df["sector"] == row["sector"]]

        c1, c2 = st.columns(2)
        with c1:
            fig_pe = go.Figure()
            fig_pe.add_trace(go.Box(
                y=sector_peers["pe_ratio"], name="Sector",
                boxpoints="all", jitter=0.3, marker_color="#A0AEC0",
            ))
            fig_pe.add_trace(go.Scatter(
                y=[row["pe_ratio"]], x=["Sector"],
                mode="markers", marker=dict(color="#E53E3E", size=18, symbol="star"),
                name=sel,
            ))
            fig_pe.update_layout(title=f"P/E ratio — {sel} vs sector",
                                  yaxis_title="P/E", height=350, showlegend=True)
            st.plotly_chart(fig_pe, use_container_width=True)

        with c2:
            fig_pb = go.Figure()
            fig_pb.add_trace(go.Box(
                y=sector_peers["pb_ratio"], name="Sector",
                boxpoints="all", jitter=0.3, marker_color="#A0AEC0",
            ))
            fig_pb.add_trace(go.Scatter(
                y=[row["pb_ratio"]], x=["Sector"],
                mode="markers", marker=dict(color="#E53E3E", size=18, symbol="star"),
                name=sel,
            ))
            fig_pb.update_layout(title=f"P/B ratio — {sel} vs sector",
                                  yaxis_title="P/B", height=350, showlegend=True)
            st.plotly_chart(fig_pb, use_container_width=True)

        # Row 3 — NEW: Peer scatter showing where this company sits
        st.markdown(f"#### {sel}'s position in the {row['sector']} valuation map")
        peer_chart = sector_peers.copy()
        peer_chart["is_target"] = peer_chart["ticker"] == sel
        fig_peer = px.scatter(
            peer_chart, x="pe_ratio", y="deal_probability",
            size="market_cap_bn", color="is_target",
            color_discrete_map={True: "#E53E3E", False: "#A0AEC0"},
            hover_name="ticker",
            hover_data={"name": True, "deal_score": ":.0f", "is_target": False},
            labels={"pe_ratio": "P/E ratio", "deal_probability": "Acq. probability"},
            size_max=40,
        )
        fig_peer.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_peer, use_container_width=True)

# ═══════════════════════════ TAB 4: AI Deal Memo ══════════════════════════════
with tab_memo:
    st.subheader("AI-generated investment memo")
    st.caption("Llama 3.3 70B drafts a banker-style M&A memo in ~3 seconds via Groq's free API.")

    api_key_set = bool(os.getenv("GROQ_API_KEY"))
    if not api_key_set:
        st.warning(
            "**GROQ_API_KEY not found.** Get a FREE key (no credit card) at "
            "[console.groq.com/keys](https://console.groq.com/keys) "
            "and add it to your `.env`:\n\n"
            "```\nGROQ_API_KEY=gsk_your_key_here\n```"
        )

    c1, c2 = st.columns([3, 1])
    with c1:
        memo_ticker = st.selectbox(
            "Select target company",
            sorted(df["ticker"].dropna().unique().tolist()),
            key="memo_ticker",
        )
    with c2:
        model_choice = st.selectbox(
            "Model",
            ["Llama 3.3 70B (best)", "Llama 3.1 8B (fastest)", "Mixtral 8x7B", "Gemma 2 9B"],
            key="memo_model",
        )

    target = df[df["ticker"] == memo_ticker].iloc[0]
    st.caption(
        f"**{target['name']}** · "
        f"Deal score {target['deal_score']:.0f}/100 · "
        f"Acq. prob {target['deal_probability']*100:.1f}%"
    )

    if st.button("Generate memo", disabled=not api_key_set, type="primary"):
        with st.spinner("Groq LLM is drafting the memo... (usually 2-4 seconds)"):
            try:
                from src.memo_generator import generate_memo, compute_peer_stats
                peer_stats = compute_peer_stats(df, target["sector"])
                if "70B" in model_choice:    model_key = "llama-70b"
                elif "8B" in model_choice:   model_key = "llama-8b"
                elif "Mixtral" in model_choice: model_key = "mixtral"
                else:                        model_key = "gemma"
                memo, tokens = generate_memo(target.to_dict(), peer_stats, model=model_key)
                st.session_state["current_memo"] = memo
                st.session_state["current_memo_ticker"] = memo_ticker
                st.session_state["current_memo_tokens"] = tokens
            except Exception as e:
                st.error(f"Memo generation failed: {e}")

    # ── Display the memo (only if one exists for the selected ticker) ─────
    if ("current_memo" in st.session_state
            and st.session_state.get("current_memo_ticker") == memo_ticker):
        st.divider()

        # KPI strip
        st.markdown(f"### {target['name']} ({memo_ticker})")
        st.caption(f"{target['sector']} · {target['market_cap_tier']} · "
                   f"Generated {datetime.now().strftime('%H:%M')}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Market Cap", f"${target['market_cap_bn']:.1f}B")
        k2.metric("EBITDA",     f"${target['ebitda_bn']:.1f}B")
        k3.metric("P/E Ratio",  f"{target['pe_ratio']:.1f}x")
        k4.metric("Div Yield",  f"{target['dividend_yield']:.2f}%")
        k5.metric("Deal Score", f"{target['deal_score']:.0f}/100")

        k6, k7, k8, k9, k10 = st.columns(5)
        k6.metric("P/B Ratio",  f"{target['pb_ratio']:.2f}x")
        k7.metric("P/S Ratio",  f"{target['ps_ratio']:.2f}x")
        k8.metric("EV/EBITDA",  f"{target['ev_ebitda_proxy']:.1f}x")
        k9.metric("Acq. Prob.", f"{target['deal_probability']*100:.1f}%")

        sector_peers = df[df["sector"] == target["sector"]]
        sector_pe    = sector_peers["pe_ratio"].median()
        pe_gap       = (target["pe_ratio"] / sector_pe - 1) * 100
        k10.metric("P/E vs Sector", f"{pe_gap:+.0f}%", delta_color="inverse")

        st.divider()

        # Pre-built financial metrics table (data, not LLM)
        st.markdown("### :bar_chart: Key Financial Metrics")
        from src.memo_generator import compute_peer_stats
        peer_stats = compute_peer_stats(df, target["sector"])

        def gap_str(target_val, sector_val):
            if pd.isna(target_val) or pd.isna(sector_val) or sector_val == 0:
                return "—"
            diff = (target_val / sector_val - 1) * 100
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.0f}%"

        def verdict(target_val, sector_val, lower_is_cheap=True):
            if pd.isna(target_val) or pd.isna(sector_val):
                return "—"
            diff = (target_val / sector_val - 1) * 100
            if lower_is_cheap:
                if diff < -10: return "CHEAP"
                if diff > 10:  return "EXPENSIVE"
                return "in line"
            else:
                if diff > 10:  return "ABOVE peers"
                if diff < -10: return "BELOW peers"
                return "in line"

        # Defensive lookups — works even if memo_generator wasn't updated yet
        med_ev = peer_stats.get("median_ev_ebitda")
        if med_ev is None:
            med_ev = sector_peers["ev_ebitda_proxy"].median()

        metrics_df = pd.DataFrame([
            {"Metric": "Market Cap",
            "Company": f"${target['market_cap_bn']:.1f}B",
            "Sector Median": f"${peer_stats['median_mcap_bn']:.1f}B",
            "Gap": gap_str(target["market_cap_bn"], peer_stats["median_mcap_bn"]),
            "Verdict": "DIGESTIBLE" if target["market_cap_bn"] < peer_stats["median_mcap_bn"]
                        else " LARGE TARGET"},
            {"Metric": "EBITDA",
             "Company": f"${target['ebitda_bn']:.1f}B",
             "Sector Median": "—", "Gap": "—", "Verdict": "—"},
            {"Metric": "P/E Ratio",
             "Company": f"{target['pe_ratio']:.1f}x",
             "Sector Median": f"{peer_stats['median_pe']:.1f}x",
             "Gap": gap_str(target["pe_ratio"], peer_stats["median_pe"]),
             "Verdict": verdict(target["pe_ratio"], peer_stats["median_pe"])},
            {"Metric": "P/B Ratio",
             "Company": f"{target['pb_ratio']:.2f}x",
             "Sector Median": f"{peer_stats['median_pb']:.2f}x",
             "Gap": gap_str(target["pb_ratio"], peer_stats["median_pb"]),
             "Verdict": verdict(target["pb_ratio"], peer_stats["median_pb"])},
            {"Metric": "P/S Ratio",
             "Company": f"{target['ps_ratio']:.2f}x",
             "Sector Median": f"{peer_stats['median_ps']:.2f}x",
             "Gap": gap_str(target["ps_ratio"], peer_stats["median_ps"]),
             "Verdict": verdict(target["ps_ratio"], peer_stats["median_ps"])},
            {"Metric": "EV/EBITDA",
             "Company": f"{target['ev_ebitda_proxy']:.1f}x",
             "Sector Median": f"{med_ev:.1f}x",
             "Gap": gap_str(target["ev_ebitda_proxy"], med_ev),
             "Verdict": verdict(target["ev_ebitda_proxy"], med_ev)},
            {"Metric": "Dividend Yield",
             "Company": f"{target['dividend_yield']:.2f}%",
             "Sector Median": f"{peer_stats['median_yield']:.2f}%",
             "Gap": gap_str(target["dividend_yield"], peer_stats["median_yield"]),
             "Verdict": verdict(target["dividend_yield"], peer_stats["median_yield"], False)},
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption("Computed from your data — not generated by the LLM.")
        st.divider()

        # LLM memo split into expandable sections
        memo_text = st.session_state["current_memo"]
        sections = {}
        current_section = None
        current_content = []
        for line in memo_text.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line.replace("## ", "").strip()
                current_content = []
            else:
                current_content.append(line)
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        section_icons = {
            "Executive Summary":          "🎯",
            "Strategic Rationale":        "💡",
            "Valuation Analysis":         "💰",
            "Key Risks":                  "⚠️",
            "Likely Strategic Acquirers": "🤝",
            "Precedent Transactions":     "📚",
        }

        for section_name, content in sections.items():
            display_name = section_name.split(". ", 1)[-1] if ". " in section_name else section_name
            icon = section_icons.get(display_name, "📝")
            with st.expander(f"{icon}  {display_name}", expanded=True):
                st.markdown(content)

        st.divider()

        t = st.session_state["current_memo_tokens"]
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"Model: {t['model']} · {t['input']} input + {t['output']} output tokens")
        with c2:
            st.download_button(
                "Download memo (Markdown)",
                f"# M&A Target Memo: {target['name']} ({memo_ticker})\n\n" + memo_text,
                f"{memo_ticker}_memo.md",
                "text/markdown",
                use_container_width=True,
            )

# ═══════════════════════════ TAB 5: Backtest ═════════════════════════════════
with tab_backtest:
    st.subheader("Temporal backtest — does the model predict real deals?")
    st.caption("Walk-forward validation: train on deals up to year X, test on deals after.")

    bt = load_backtest_results()
    if bt is None:
        st.warning("Run `python notebooks/04_backtest.py` to generate backtest results.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg AUC",                 f"{bt['auc'].mean():.3f}")
        c2.metric("Avg precision @ top-20",  f"{bt['precision_at_20'].mean():.1%}")
        c3.metric("Avg recall @ top-50",     f"{bt['recall_at_50'].mean():.1%}")
        c4.metric("Avg lift @ top-20",       f"{bt['lift_at_20'].mean():.1f}x")

        st.markdown("#### Walk-forward results")
        st.dataframe(bt, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            path = os.path.join(PROJECT_ROOT, "reports", "backtest_precision_at_k.png")
            if os.path.exists(path):
                st.image(path, caption="Precision @ top-K across time windows")
        with c2:
            path = os.path.join(PROJECT_ROOT, "reports", "backtest_lift_curve.png")
            if os.path.exists(path):
                st.image(path, caption="Cumulative lift curve vs random baseline")

        summary_path = os.path.join(PROJECT_ROOT, "reports", "backtest_summary.md")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                st.markdown(f.read())


# ══════════════════════════ TAB 6: Model insights ═════════════════════════════
with tab_model:
    st.subheader("Model explainability — what drives predictions")
    c1, c2 = st.columns(2)
    with c1:
        path = os.path.join(PROJECT_ROOT, "reports", "shap_bar.png")
        if os.path.exists(path):
            st.image(path, caption="Top features driving acquisition probability")
        else:
            st.info("Run `python notebooks/03_train_models.py` to generate SHAP plots.")
    with c2:
        path = os.path.join(PROJECT_ROOT, "reports", "shap_summary.png")
        if os.path.exists(path):
            st.image(path, caption="SHAP value distribution by feature")
        path = os.path.join(PROJECT_ROOT, "reports", "roc_curve.png")
        if os.path.exists(path):
            st.image(path, caption="Model discrimination — ROC curve")

    st.markdown("""
#### How to read these charts

The **bar chart** shows which financial metrics most influence acquisition predictions.
Longer bar = more influence.

The **beeswarm plot** shows directional influence — red dots are high values, blue dots
are low values. If low P/E (blue) pushes predictions right, cheap stocks are more likely targets.
""")


st.divider()
st.caption("Deal Intelligence Platform · Python · XGBoost · SHAP · Groq LLM · Streamlit")
