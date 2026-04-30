"""
memo_generator.py  —  LLM-powered M&A investment memo generator (v3).

Section 2 (Key Financial Metrics) is now rendered by the dashboard in Python.
The LLM produces 6 narrative sections only.
"""

import os, sys, argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY", "")

MODELS = {
    "llama-70b":  "llama-3.3-70b-versatile",
    "llama-8b":   "llama-3.1-8b-instant",
    "mixtral":    "mixtral-8x7b-32768",
    "gemma":      "gemma2-9b-it",
}
DEFAULT_MODEL = "llama-70b"


def fmt_num(val, fmt="{:.1f}", default="n/a"):
    if val is None: return default
    try:
        if isinstance(val, float) and (val != val): return default
        return fmt.format(float(val))
    except Exception:
        return default


def fmt_currency(val, default="n/a"):
    if val is None: return default
    try:
        if isinstance(val, float) and (val != val): return default
        return f"${float(val):.1f}B"
    except Exception:
        return default


def fmt_pct(val, default="n/a"):
    if val is None: return default
    try:
        if isinstance(val, float) and (val != val): return default
        return f"{float(val):.2f}%"
    except Exception:
        return default


def compute_peer_stats(df, sector):
    """Sector benchmarks. Includes median_ev_ebitda."""
    peers = df[df["sector"] == sector]
    return {
        "peer_count":       len(peers),
        "median_pe":        peers["pe_ratio"].median(),
        "median_pb":        peers["pb_ratio"].median(),
        "median_ps":        peers["ps_ratio"].median(),
        "median_ev_ebitda": peers["ev_ebitda_proxy"].median(),
        "median_mcap_bn":   peers["market_cap_bn"].median(),
        "median_yield":     peers["dividend_yield"].median(),
    }


def build_context(company, peer_stats):
    """Pre-compute valuation gaps. The LLM uses these verbatim."""

    def gap(t, s):
        if t is None or s is None or s == 0: return "n/a"
        try:
            d = (float(t) / float(s) - 1) * 100
            return f"{'+' if d > 0 else ''}{d:.0f}%"
        except Exception:
            return "n/a"

    def label(t, s, lower_is_cheap=True):
        if t is None or s is None: return "n/a"
        try:
            d = (float(t) / float(s) - 1) * 100
        except Exception:
            return "n/a"
        if lower_is_cheap:
            if d < -10: return "CHEAP vs peers"
            if d > 10:  return "EXPENSIVE vs peers"
            return "in line with peers"
        else:
            if d > 10:  return "ABOVE peers"
            if d < -10: return "BELOW peers"
            return "in line with peers"

    pe_gap = gap(company.get("pe_ratio"),        peer_stats.get("median_pe"))
    pb_gap = gap(company.get("pb_ratio"),        peer_stats.get("median_pb"))
    ps_gap = gap(company.get("ps_ratio"),        peer_stats.get("median_ps"))
    ev_gap = gap(company.get("ev_ebitda_proxy"), peer_stats.get("median_ev_ebitda"))
    yld_gap = gap(company.get("dividend_yield"), peer_stats.get("median_yield"))
    mcap_gap = gap(company.get("market_cap_bn"), peer_stats.get("median_mcap_bn"))

    pe_v = label(company.get("pe_ratio"),        peer_stats.get("median_pe"))
    pb_v = label(company.get("pb_ratio"),        peer_stats.get("median_pb"))
    ps_v = label(company.get("ps_ratio"),        peer_stats.get("median_ps"))
    ev_v = label(company.get("ev_ebitda_proxy"), peer_stats.get("median_ev_ebitda"))

    return f"""TARGET COMPANY
Ticker:                 {company['ticker']}
Name:                   {company['name']}
Sector:                 {company['sector']}
Sub-sector / industry:  {company.get('industry', 'n/a')}
Market cap:             {fmt_currency(company.get('market_cap_bn'))}
Size tier:              {company.get('market_cap_tier', 'n/a')}
EBITDA:                 {fmt_currency(company.get('ebitda_bn'))}

PRE-COMPUTED VALUATION GAPS — USE THESE EXACT % VALUES IN THE MEMO
                        Value          vs Sector Median    Verdict
P/E ratio:              {fmt_num(company.get('pe_ratio'), '{:.1f}')}x         {pe_gap:>10}            {pe_v}
P/B ratio:              {fmt_num(company.get('pb_ratio'), '{:.2f}')}x        {pb_gap:>10}            {pb_v}
P/S ratio:              {fmt_num(company.get('ps_ratio'), '{:.2f}')}x        {ps_gap:>10}            {ps_v}
EV/EBITDA (proxy):      {fmt_num(company.get('ev_ebitda_proxy'), '{:.1f}')}x         {ev_gap:>10}            {ev_v}
Dividend yield:         {fmt_pct(company.get('dividend_yield'))}        {yld_gap:>10}            (higher = mature business)

ML MODEL OUTPUT
Acquisition probability: {fmt_num(company.get('deal_probability', 0)*100, '{:.1f}')}%
Composite deal score:    {fmt_num(company.get('deal_score', 0), '{:.1f}')}/100

SECTOR BENCHMARKS ({company['sector']})
Peer count:             {peer_stats.get('peer_count', 0)}
Median P/E:             {fmt_num(peer_stats.get('median_pe'),         '{:.1f}')}x
Median P/B:             {fmt_num(peer_stats.get('median_pb'),         '{:.2f}')}x
Median P/S:             {fmt_num(peer_stats.get('median_ps'),         '{:.2f}')}x
Median EV/EBITDA:       {fmt_num(peer_stats.get('median_ev_ebitda'), '{:.1f}')}x
Median market cap:      {fmt_currency(peer_stats.get('median_mcap_bn'))}
Median yield:           {fmt_pct(peer_stats.get('median_yield'))}
Market cap percentile:  {mcap_gap}
"""


def build_prompt(context_block):
    return f"""You are a senior investment banking associate drafting an M&A target memo.

ABSOLUTE RULES — failure to follow these means the memo is rejected:
1. The "vs Sector Median" gaps are PRE-COMPUTED. Quote them VERBATIM in the memo (e.g., "trades at +38% premium to sector P/E"). NEVER write "+0%", "at sector parity", or invent your own numbers.
2. NEVER fabricate specific dollar amounts (e.g., "$100M synergies"). Speak qualitatively.
3. For Precedent Transactions: ONLY include deals you are 100% certain happened. If unsure for this sub-sector, write a single row: "No directly comparable precedent — see adjacent deals" and skip the table. NEVER invent fake deals or wrong directions.
4. For Strategic Acquirers: name companies that operate in the SAME sub-sector. Do not list random large-cap names.
5. DO NOT output a "Key Financial Metrics" section. That section is rendered separately. Skip it entirely.

{context_block}

Output the memo in EXACTLY this 6-section structure. Use bullet points written as COMPLETE SENTENCES with 12-25 words each. NEVER use 2-word fragments. Each bullet must include a specific finding, not just a category label.

For example:
BAD: "Cost savings"
GOOD: "Cost synergies likely from R&D consolidation and shared distribution infrastructure across overlapping product lines."

BAD: "Antitrust concerns"
GOOD: "FTC review likely given combined market share in life sciences instrumentation, but precedent suggests approval with divestitures."

## 1. Executive Summary
- Headline thesis citing the ML acquisition probability and the LARGEST valuation gap (use exact %)
- Most likely acquirer profile (sub-sector and size)
- Estimated takeout premium range
- Key downside risk in one line

## 2. Strategic Rationale
Each bullet must be a 15-25 word sentence describing a specific synergy or strategic angle.
- Horizontal/vertical fit specific to this sub-sector — name the segment
- Cost synergy with the type identified (e.g. "R&D consolidation", "manufacturing scale", "SG&A overlap")
- Revenue synergy or platform expansion — name what specifically expands
- Defensive/competitive rationale citing a specific competitor threat or trend

## 3. Valuation Analysis
- P/E vs sector: cite exact gap from data + verdict (CHEAP/EXPENSIVE/in line)
- P/B vs sector: cite exact gap + verdict
- P/S vs sector: cite exact gap + verdict
- EV/EBITDA vs sector: cite exact gap + verdict
- **Implied takeout premium: choose a specific range like 25-35% or 30-50%. Never write "undetermined" — make a reasonable estimate based on standard M&A premiums (typical range: 20-50%).**

## 4. Key Risks
- Antitrust risk specific to this combination
- Integration risk specific to this business model
- Financial / leverage / financing risk

## 5. Likely Strategic Acquirers
| Acquirer | Strategic Fit | Capacity |
|---|---|---|
| <real company in same sub-sector> | <specific reason> | strong/moderate |
| <real company in same sub-sector> | <specific reason> | strong/moderate |
| <real company in same sub-sector> | <specific reason> | strong/moderate |

## 6. Precedent Transactions
If you know real deals in this sub-sector, list 2-3:
| Year | Target | Acquirer | Deal Value | Multiple |
|---|---|---|---|---|
| <year> | <target name> | <acquirer name> | $XB | XXx EBITDA |

If unsure, output ONLY this single line and skip the table:
"No directly comparable precedent identified for this sub-sector at this size."

CRITICAL: Output ONLY these 6 sections (1 through 6). Do NOT include a "Key Financial Metrics" section. Output ONLY the memo, no preamble."""


def generate_memo(company_dict, peer_stats, model=DEFAULT_MODEL):
    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set in .env. Get free key at console.groq.com/keys"
        )
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("Run: pip install groq")

    model_id = MODELS.get(model, model)
    client   = Groq(api_key=API_KEY)
    context  = build_context(company_dict, peer_stats)
    prompt   = build_prompt(context)

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=1200,
        temperature=0.2,   # lower temperature → less hallucination
        messages=[{"role": "user", "content": prompt}],
    )

    memo = response.choices[0].message.content

    # Strip any preamble before "## 1."
    if "## 1." in memo:
        memo = "## 1." + memo.split("## 1.", 1)[1]

    tokens = {
        "input":  response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "model":  model_id,
    }
    return memo, tokens


def save_memo(ticker, company_name, memo, tokens):
    os.makedirs("reports/memos", exist_ok=True)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"reports/memos/{ticker}_{date}.md"
    with open(path, "w") as f:
        f.write(f"# M&A Target Memo: {company_name} ({ticker})\n")
        f.write(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
                f"Model: {tokens['model']}*\n\n---\n\n")
        f.write(memo)
    return path


if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()))
    args = parser.parse_args()

    if not os.path.exists("data/scored_companies.csv"):
        print("Run notebooks/03_train_models.py first"); sys.exit(1)

    df  = pd.read_csv("data/scored_companies.csv")
    row = df[df["ticker"] == args.ticker.upper()]
    if row.empty:
        print(f"Ticker not found"); sys.exit(1)

    company    = row.iloc[0].to_dict()
    peer_stats = compute_peer_stats(df, company["sector"])
    memo, tokens = generate_memo(company, peer_stats, model=args.model)
    path = save_memo(args.ticker.upper(), company["name"], memo, tokens)

    print("=" * 70)
    print(memo)
    print("=" * 70)
    print(f"\nSaved: {path}")