# file: app.py
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Dict, List, Tuple, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# API client (read-only)
# BASE = "http://concept.alkimiya.io/api"
BASE = "http://54.179.82.59/api"
st.caption(f"API_BASE_URL: {BASE}")

# ================================
# Constants
# ================================
DEFAULT_DECIMAL_PRECISION = 2
MAX_SHARES = 5_000_000
STARTING_BALANCE = 10_000.0
BINARY_OUTCOME_TOKENS = ["YES", "NO"]
MARKET_ID_DEFAULT = 1

# Points system
TRADE_POINTS_PER_USD = 10.0
# MINT_POINTS_PER_SHARE = 10.0
PNL_POINTS_PER_USD   = 5.0
EARLY_QUANTITY_POINT = 5_000_000
MID_QUANTITY_POINT   = 2_500_000
PHASE_MULTIPLIERS = {"early": 1.5, "mid": 1.25, "late": 1.0}

# ================================
# HTTP Session
# ================================
@st.cache_resource
def http_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"Accept": "application/json"})
    return s

def _get(path: str, timeout: float = 8.0, session: Optional[requests.Session] = None):
    s = session or http_session()
    url = f"{BASE.rstrip('/')}{path}"
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ================================
# Streamlit Page
# ================================
st.set_page_config(page_title="42: Cross-Market Analytics", layout="wide")
st.title("42: Dynamic Pari-mutuel — Cross-Market Analytics")

# ================================
# Utilities
# ================================
def _extract_tokens(market: dict) -> list[str]:
    outs = market.get("outcomes") or []
    if outs and isinstance(outs[0], dict):
        toks = [(o.get("token") or o.get("display") or o.get("name") or "").strip() for o in outs]
    else:
        toks = [str(o).strip() for o in outs]
    return [t.upper() for t in toks if t]

def interpret_market(srv: dict) -> Tuple[bool, str]:
    try:
        now = datetime.utcnow()
        end_ts = datetime.fromisoformat(srv.get("end_ts"))
        resolved = int(srv.get("resolved") or 0)
    except Exception:
        return False, "unknown"
    if resolved:
        return False, "resolved"
    elif now >= end_ts:
        return False, "market ended"
    else:
        return True, "active"

def st_display_market_status(active: bool, reason: str = ""):
    if active:
        st.badge("Active", icon=":material/check:", color="green")
    else:
        lab = "Inactive"
        if reason:
            lab = f"{lab} ({reason})"
        st.markdown(f":gray-badge[{lab}]")

def format_usdc_compact(value: float) -> str:
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for i, (scale, suf) in enumerate(units):
        if v >= scale:
            q = v / scale
            dec = 2 if q < 10 else (1 if q < 100 else 0)
            s = f"{q:.{dec}f}"
            if float(s) >= 1000 and i > 0:
                next_scale, next_suf = units[i - 1]
                q = v / next_scale
                dec = 2 if q < 10 else (1 if q < 100 else 0)
                s = f"{q:.{dec}f}"
                suf = next_suf
            s = s.rstrip("0").rstrip(".")
            return f"{sign}${s}{suf}"
    return f"{sign}${v:,.2f}"

# ================================
# Curves
# ================================
def buy_curve(x: float) -> float:
    return (float(x) ** (1.0 / 3.0)) / 1000.0 + 0.1

def buy_delta(x: float) -> float:
    return (3.0 / 4000.0) * (x ** (4.0 / 3.0)) + x / 10.0

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def sale_tax_rate(q: int, C: int) -> float:
    if C <= 0 or q <= 0:
        return 0.0
    X = _clamp01(float(q) / float(C))
    base = 1.15 - 1.3 / (1.0 + math.e ** (4.0 * X - 2.0))
    return _clamp01(base)

def current_marginal_sell_price_after_tax(reserve: int) -> float:
    if reserve <= 0:
        return 0.0
    gross1 = buy_delta(reserve) - buy_delta(reserve - 1)
    tax1 = sale_tax_rate(1, reserve)
    return max(0.0, float(gross1 * (1.0 - tax1)))

_sale_tax_rate_vec = np.vectorize(sale_tax_rate, otypes=[float])

def buy_curve_np(x: np.ndarray) -> np.ndarray:
    return np.cbrt(x) / 1000.0 + 0.1

def sell_marginal_net_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(int)
    bd = np.vectorize(buy_delta,  otypes=[float])(x)
    bd_prev = np.vectorize(buy_delta,  otypes=[float])(np.maximum(0, x - 1))
    gross_1 = bd - bd_prev
    tax_1 = _sale_tax_rate_vec(1, x)
    net_1 = gross_1 * (1.0 - tax_1)
    net_1[x <= 0] = 0.0
    return net_1

def _curve_samples(max_shares: int, reserve: int, dense_pts: int = 1500, sparse_pts: int = 600) -> np.ndarray:
    if max_shares <= 1:
        return np.array([1], dtype=int)
    half = min(int(0.03 * max_shares), 50_000)
    lo = max(1, reserve - half)
    hi = min(max_shares, reserve + half)
    dense = np.linspace(lo, hi, num=max(2, dense_pts), dtype=int)
    left = np.array([], dtype=int)
    if lo > 1:
        left = np.unique(np.logspace(0, np.log10(lo), num=max(2, sparse_pts // 2), base=10.0)).astype(int)
        left = left[(left >= 1) & (left < lo)]
    right = np.array([], dtype=int)
    if hi < max_shares:
        right = np.unique(np.logspace(np.log10(hi + 1), np.log10(max_shares), num=max(2, sparse_pts // 2), base=10.0)).astype(int)
        right = right[(right > hi) & (right <= max_shares)]
    xs = np.unique(np.concatenate([left, dense, right]))
    return xs

@st.cache_data(show_spinner=False)
def get_curve_series(max_shares: int, reserve: int, dense_pts: int = 1500, sparse_pts: int = 600):
    xs = _curve_samples(max_shares, reserve, dense_pts=dense_pts, sparse_pts=sparse_pts)
    return xs, buy_curve_np(xs), sell_marginal_net_np(xs)

# ================================
# Data fetchers
# ================================
@st.cache_data(ttl=5, show_spinner=False)
def fetch_all_tx() -> pd.DataFrame:
    try:
        return pd.DataFrame(_get("/tx"))
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=5, show_spinner=False)
def fetch_users() -> pd.DataFrame:
    try:
        return pd.DataFrame(_get("/users"))
    except Exception:
        return pd.DataFrame(columns=["id", "username", "balance"])

@st.cache_data(ttl=5, show_spinner=False)
def fetch_market(market_id: int) -> dict:
    try:
        return _get(f"/market/{market_id}", 8.0)
    except Exception:
        return {"id": market_id, "reserves": [], "outcomes": []}

@st.cache_data(ttl=10, show_spinner=False)
def discover_markets() -> List[dict]:
    try:
        markets = _get("/markets", 8.0)
        if isinstance(markets, dict) and "results" in markets:
            markets = markets["results"]
        if isinstance(markets, list) and markets:
            return markets
    except Exception:
        pass
    tx = fetch_all_tx()
    ids = set(pd.to_numeric(tx.get("market_id", pd.Series(dtype="int64")), errors="coerce").dropna().astype(int).tolist())
    if not ids:
        ids = {MARKET_ID_DEFAULT}
    return [fetch_market(mid) for mid in sorted(ids)]

def load_users_df(users_raw: pd.DataFrame) -> pd.DataFrame:
    df = users_raw.copy()
    if "id" not in df.columns and "user_id" in df.columns:
        df = df.rename(columns={"user_id": "id"})
    if "username" not in df.columns:
        for alt in ("user", "user_name", "name", "User", "Username"):
            if alt in df.columns:
                df = df.rename(columns={alt: "username"})
                break
    if "balance" not in df.columns:
        for alt in ("Balance", "cash", "usdc", "wallet", "Balance After"):
            if alt in df.columns:
                df = df.rename(columns={alt: "balance"})
                break
    for col, dtype in (("id", "int64"), ("username", "object"), ("balance", "float64")):
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype)
    return df[["id", "username", "balance"]]

# -------- Robust tx normalization helpers --------
RENAME_MAP = {
    "ts": "Time",
    "user": "User",
    "user_name": "User",
    "username": "User",
    "user_id": "user_id",
    "action": "Action",
    "token": "Outcome",
    "qty": "Quantity",
    "buy_delta": "BuyAmt_Delta",
    "sell_delta": "SellAmt_Delta",
    "buy_price": "Buy Price",
    "sell_price": "Sell Price",
    "balance_after": "Balance After",
}

def normalize_tx_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    cols_to_rename = {c: RENAME_MAP[c] for c in RENAME_MAP.keys() if c in out.columns}
    out = out.rename(columns=cols_to_rename)
    if "Time" in out.columns:
        out["Time"] = pd.to_datetime(out["Time"], format="ISO8601", utc=True, errors="coerce")
    if "Quantity" in out.columns:
        out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0).astype(int)
    for c in ["BuyAmt_Delta", "SellAmt_Delta", "Buy Price", "Sell Price", "Balance After", "market_id", "user_id"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def ensure_user_column(df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "User" in out.columns:
        return out
    for alt in ("username", "user_name", "User "):
        if alt in out.columns:
            out["User"] = out[alt].astype(str)
            return out
    if "user_id" in out.columns and not users_df.empty:
        mapper = dict(zip(users_df["id"].astype(int), users_df["username"].astype(str)))
        out["User"] = out["user_id"].map(lambda x: mapper.get(int(x), str(int(x)) if pd.notnull(x) else "unknown"))
        return out
    if "user_id" in out.columns:
        out["User"] = out["user_id"].map(lambda x: str(int(x)) if pd.notnull(x) else "unknown")
    else:
        out["User"] = "unknown"
    return out

# ================================
# Points helpers
# ================================
def compute_user_points(tx_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    id_to_user = dict(zip(users_df["id"], users_df["username"]))
    reserves_state = {}  # (market_id, token) -> shares
    vol_points = {uid: 0.0 for uid in users_df["id"]}

    if tx_df.empty:
        return pd.DataFrame(columns=["user_id", "User", "VolumePoints"])

    tdf = normalize_tx_columns(tx_df)
    tdf = ensure_user_column(tdf, users_df)
    tdf = tdf.sort_values("Time")

    for _, r in tdf.iterrows():
        uid = int(r.get("user_id") or 0)
        act = r.get("Action")
        tkn = r.get("Outcome")
        qty = int(r.get("Quantity") or 0)
        mid = int(r.get("market_id") or MARKET_ID_DEFAULT)
        key = (mid, tkn)
        if key not in reserves_state:
            reserves_state[key] = 0
        if act == "Buy":
            buy_usd = float(r.get("BuyAmt_Delta") or 0.0)
            r_before = reserves_state[key]
            mult = PHASE_MULTIPLIERS["early"] if r_before < EARLY_QUANTITY_POINT else (
                   PHASE_MULTIPLIERS["mid"] if r_before < MID_QUANTITY_POINT else PHASE_MULTIPLIERS["late"])
            vol_points[uid] += buy_usd * TRADE_POINTS_PER_USD * mult
            reserves_state[key] = r_before + qty
        elif act == "Sell":
            sell_usd = float(r.get("SellAmt_Delta") or 0.0)
            vol_points[uid] += sell_usd * TRADE_POINTS_PER_USD
            reserves_state[key] = max(0, reserves_state[key] - qty)
        elif act == "Resolve":
            for k in list(reserves_state.keys()):
                if k[0] == mid:
                    reserves_state[k] = 0

    out = pd.DataFrame({
        "user_id": list(vol_points.keys()),
        "User":    [id_to_user.get(uid, str(uid)) for uid in vol_points.keys()],
        "VolumePoints": [vol_points[uid] for uid in vol_points.keys()]
    })
    return out

def rebuild_holdings_now(tx_norm: pd.DataFrame, users_df: pd.DataFrame, tokens: List[str]) -> Dict[int, Dict[str, int]]:
    h = {int(uid): {t: 0 for t in tokens} for uid in users_df["id"]}
    if tx_norm.empty:
        return h
    tx_sorted = tx_norm.sort_values("Time")
    for _, r in tx_sorted.iterrows():
        uid = int(r.get("user_id") or 0)
        act = r.get("Action")
        tkn = r.get("Outcome")
        qty = int(r.get("Quantity") or 0)
        if act == "Resolve":
            h = {int(u): {t: 0 for t in tokens} for u in users_df["id"]}
            continue
        if uid not in h or tkn not in tokens:
            continue
        if act == "Buy":
            h[uid][tkn] += qty
        elif act == "Sell":
            h[uid][tkn] -= qty
    return h

# ================================
# Sidebar: Market selection + log scope
# ================================
with st.sidebar:
    st.header("Filters")
    markets = discover_markets()

    def _mk_label(m: dict) -> str:
        mid = m.get("id", "?")
        q = m.get("question") or m.get("MarketQuestion") or ""
        return f"{mid} — {q[:60]}{'…' if q and len(q) > 60 else ''}"

    options = []
    label_to_id = {}
    for m in markets:
        lbl = _mk_label(m)
        options.append(lbl)
        label_to_id[lbl] = int(m.get("id", MARKET_ID_DEFAULT))

    default_label = options[0] if options else f"{MARKET_ID_DEFAULT} — (default)"
    selected_label = st.selectbox("Select Market (scopes most views)", options=options or [default_label], index=0)
    SELECTED_MARKET_ID = label_to_id.get(selected_label, MARKET_ID_DEFAULT)

    tx_scope = st.selectbox("Transaction Log Scope", options=["All markets", "Only selected market"], index=0)

# ================================
# Core data pulls
# ================================
tx_raw = fetch_all_tx()
users_df_all = load_users_df(fetch_users())

# Normalize & ensure 'User'
tx_norm_all = normalize_tx_columns(tx_raw)
tx_norm_all = ensure_user_column(tx_norm_all, users_df_all)

# Scoped tx for selected market
if "market_id" in tx_norm_all.columns:
    tx_norm_selected = tx_norm_all[tx_norm_all["market_id"].eq(SELECTED_MARKET_ID)].copy()
else:
    tx_norm_selected = tx_norm_all.copy()

# Selected market object
m_selected = fetch_market(SELECTED_MARKET_ID)
TOKENS_SELECTED = _extract_tokens(m_selected) or BINARY_OUTCOME_TOKENS

server_active, server_why = interpret_market(m_selected)

# ================================
# Selected market header
# ================================
st.subheader(f"Selected Market: :blue[{m_selected.get('question', 'Unknown')}]")

# ================================
# Overall Market Metrics (selected)
# ================================
reserves_list = m_selected.get("reserves") or []
res_df = pd.DataFrame(reserves_list)

# If backend includes market_id, keep only the selected market
if not res_df.empty and "market_id" in res_df.columns:
    res_df = res_df[pd.to_numeric(res_df["market_id"], errors="coerce").eq(SELECTED_MARKET_ID)].copy()

# --- Normalize to canonical columns: Token, Shares, USDC ---
rename_cols = {"token": "Token", "shares": "Shares", "usdc": "USDC"}
res_df = res_df.rename(columns={c: rename_cols[c] for c in rename_cols if c in res_df.columns})

# Ensure required columns exist
for col in ["Token", "Shares", "USDC"]:
    if col not in res_df.columns:
        res_df[col] = 0 if col != "Token" else ""

# Types
res_df["Token"]  = res_df["Token"].astype(str)
res_df["Shares"] = pd.to_numeric(res_df["Shares"], errors="coerce").fillna(0).astype(int)
res_df["USDC"]   = pd.to_numeric(res_df["USDC"],   errors="coerce").fillna(0.0)

# Canonicalize case to match TOKENS_SELECTED (which are uppercased by _extract_tokens)
TOKENS_SELECTED = _extract_tokens(m_selected) or BINARY_OUTCOME_TOKENS  # already .upper()
res_df["TokenKey"] = res_df["Token"].str.upper()

# Collapse possible duplicates by TokenKey
res_df = (
    res_df.groupby("TokenKey", as_index=False)
          .agg({"Shares": "sum", "USDC": "sum"})
          .rename(columns={"TokenKey": "Token"})
)

# Ensure all expected tokens exist (even if zero)
present = set(res_df["Token"])
missing_tokens = [t for t in TOKENS_SELECTED if t not in present]
if missing_tokens:
    res_df = pd.concat(
        [res_df, pd.DataFrame({"Token": missing_tokens, "Shares": 0, "USDC": 0.0})],
        ignore_index=True
    )

# Order rows to match TOKENS_SELECTED using a categorical sort (no set_index/reindex)
res_df["_order"] = pd.Categorical(res_df["Token"], categories=TOKENS_SELECTED, ordered=True)
res_df = res_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

print("3:", res_df)

# Totals
res_tot_usdc = float(res_df["USDC"].sum())

st.subheader("Overall Market Metrics")
mc_cols = st.columns(3)
with mc_cols[0]:
    st.metric("Market ID", SELECTED_MARKET_ID)
with mc_cols[1]:
    st.write("Market Status")
    st_display_market_status(server_active, server_why)
with mc_cols[2]:
    st.metric("Users", len(users_df_all))
    
# --- helper to chunk tokens into rows of up to 3 ---
def _chunked(seq, size=3):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# === Compute per-token and overall trading volume (from normalized tx) ===
# tx_norm_all exists earlier: normalize_tx_columns(fetch_all_tx()) then ensure_user_column(...)
tx_src = tx_norm_all.copy()

token_volumes = pd.DataFrame(columns=["Token", "Volume"])
if not tx_src.empty:
    # Keep only this market
    if "market_id" in tx_src.columns:
        tx_src = tx_src[pd.to_numeric(tx_src["market_id"], errors="coerce").eq(SELECTED_MARKET_ID)].copy()

    # Make sure the value columns exist
    for c in ["BuyAmt_Delta", "SellAmt_Delta"]:
        if c not in tx_src.columns:
            tx_src[c] = 0.0

    # Outcome -> TokenKey (UPPER) to match res_df["Token"] which is canonical uppercase now
    tx_src["Outcome"] = tx_src["Outcome"].astype(str)
    tx_src["Token"] = tx_src["Outcome"].str.upper()

    # Volume = sum of buy + sell $ for that token
    token_volumes = (
        tx_src.groupby("Token", as_index=False)
              .agg({"BuyAmt_Delta": "sum", "SellAmt_Delta": "sum"})
              .assign(Volume=lambda d: d["BuyAmt_Delta"].fillna(0) + d["SellAmt_Delta"].fillna(0))
              [["Token", "Volume"]]
    )

# Join volume into reserves (res_df already has Token/Shares/USDC with Token uppercased)
res_df = res_df.merge(token_volumes, on="Token", how="left").fillna({"Volume": 0.0})
res_df["Volume"] = res_df["Volume"].astype(float)

# Overall market trading volume
total_market_volume = float(res_df["Volume"].sum())
omc_cols = st.columns(3)
with omc_cols[0]:
    st.metric("Total USDC Reserve", round(res_tot_usdc, 2))
with omc_cols[1]:
    st.metric("Overall Market Trading Volume", f"{total_market_volume:,.2f} USDC")

# === Unified per-token stats (Shares, Price, MCAP, Odds, Volume) — max 3 per row ===
total_market_shares = max(1, int(res_df["Shares"].sum()))

for row_tokens in _chunked(TOKENS_SELECTED, 2):
    cols = st.columns(len(row_tokens))
    for i, token in enumerate(row_tokens):
        with cols[i]:
            st.subheader(f'Outcome :blue[{token}]')

            row_q = res_df.loc[res_df['Token'] == token]
            row = {"Shares": 0, "USDC": 0.0, "Volume": 0.0} if row_q.empty else row_q.iloc[0]

            reserve = int(row["Shares"])
            price   = round(buy_curve(reserve), 3)
            mcap    = round(float(row["USDC"]), 2)
            volume  = float(row["Volume"])

            # Odds = inverse share fraction
            s = reserve
            odds_val = '-' if s == 0 else round(1 / (s / total_market_shares), 2)

            # Compact 4-up metrics row
            with st.container(border=True):
                sub_cols = st.columns(4)
                with sub_cols[0]:
                    st.metric("Total Shares", reserve)
                with sub_cols[1]:
                    st.metric("Price", f"${price}")
                with sub_cols[2]:
                    st.metric("MCAP", format_usdc_compact(mcap))
                with sub_cols[3]:
                    st.metric("Odds", f"{odds_val}x" if odds_val != '-' else "-")

            # Secondary: token volume
            st.caption(f"**Trading Volume:** {volume:,.2f} USDC")
            
st.divider()

# ================================
# Transaction Log
# ================================
st.subheader("Transaction Log")
tx_display = tx_norm_all.copy() if tx_scope == "All markets" else tx_norm_selected.copy()
if tx_display.empty:
    st.info("No transactions to display.")
else:
    float_cols = tx_display.select_dtypes(include=["float64", "float32", "float"]).columns
    for col in float_cols:
        tx_display[col] = tx_display[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
    st.dataframe(tx_display)
st.divider()

# ================================
# Payout/Share Trend (selected)
# ================================
st.subheader("📈 Payout/Share Trend (Selected Market)")

if tx_norm_selected.empty:
    st.info("No share movement yet to plot payout/share for the selected market.")
else:
    df = tx_norm_selected.copy()

    # Canonicalize outcome tokens to match TOKENS_SELECTED (which are uppercase)
    df["Outcome"] = df["Outcome"].astype(str)
    df["TokenKey"] = df["Outcome"].str.upper()

    # Keep only rows whose token is in the selected market's outcomes
    token_set = set(TOKENS_SELECTED)  # already uppercased
    df = df[df["TokenKey"].isin(token_set)].copy()

    # Safety: ensure required fields exist / are typed
    df["Time"] = pd.to_datetime(df["Time"], format="ISO8601", utc=True, errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)

    # Order by time to reconstruct properly
    df = df.dropna(subset=["Time"]).sort_values("Time")

    if df.empty:
        st.info("No share movement observed.")
    else:
        # Running per-token share supply (by canonical key)
        running = {t: 0 for t in TOKENS_SELECTED}
        shares_timeline = []

        for _, r in df.iterrows():
            tkn = r["TokenKey"]
            act = r.get("Action")
            qty = int(r.get("Quantity") or 0)
            tstamp = r["Time"]

            if act == "Buy":
                running[tkn] += qty
            elif act == "Sell":
                running[tkn] -= qty
            # We intentionally ignore "Resolve" here (trend is about share movement)

            total = sum(running.values())
            denom = running[tkn] if running[tkn] > 0 else 1
            payout = (total if total > 0 else 1) / denom

            shares_timeline.append({
                "Time": tstamp,
                "Outcome": tkn,           # keep canonical label
                "Payout/Share": payout
            })

        ps_df = pd.DataFrame(shares_timeline)
        if ps_df.empty:
            st.info("No share movement observed.")
        else:
            fig = px.line(
                ps_df,
                x="Time",
                y="Payout/Share",
                color="Outcome",
                markers=True,
                title="Payout/Share Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()


# ================================
# Bonding Curves (selected)
# ================================
st.header("🔁 Bonding Curves by Outcome (Selected Market)")
reserves = m_selected.get("reserves") or []
latest_reserves = {r["token"]: int(r.get("shares", 0)) for r in reserves}
TOKENS_CUR = [r["token"] for r in reserves] or TOKENS_SELECTED

token_tabs = st.tabs(TOKENS_CUR)
for token, token_tab in zip(TOKENS_CUR, token_tabs):
    with token_tab:
        reserve = int(latest_reserves.get(token, 0))
        view = st.radio("View", ["Buy Curve", "Sell Spread", "Effective Sell (Net)"],
                        horizontal=True, key=f"view_{SELECTED_MARKET_ID}_{token}")
        if view == "Buy Curve":
            buy_price_now = float(buy_curve(reserve))
            sell_net_now = float(current_marginal_sell_price_after_tax(reserve))
            xs, buy_vals, sell_net_vals = get_curve_series(MAX_SHARES, reserve)
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scattergl(x=xs, y=buy_vals, mode='lines', name='Buy Curve'))
            fig_curve.add_trace(go.Scatter(x=[reserve], y=[buy_price_now], mode='markers+text',
                                           name=f'{token} Buy Point',
                                           text=[f"Shares: {reserve}<br>Buy: {buy_price_now:.4f}"],
                                           textposition="top right",
                                           marker=dict(size=10),
                                           showlegend=False))
            y0 = max(0.0, min(float(np.nanmin(buy_vals)), buy_price_now, sell_net_now))
            fig_curve.add_trace(go.Scatter(x=[reserve, reserve], y=[y0, buy_price_now],
                                           mode='lines', line=dict(dash='dot'), showlegend=False))
            fig_curve.add_trace(go.Scatter(x=[xs.min(), reserve], y=[buy_price_now, buy_price_now],
                                           mode='lines', line=dict(dash='dot'), showlegend=False))
            fig_curve.update_layout(title=f'{token} — Buy vs Sell (net, 1 share)',
                                    xaxis_title='Shares (reserve)',
                                    yaxis_title='Price',
                                    hovermode="x unified")
            st.plotly_chart(fig_curve, use_container_width=True)

        elif view == "Sell Spread":
            if reserve <= 0:
                st.info("No circulating shares yet — spread curve will show once there is supply.")
            else:
                steps = 200
                X = np.linspace(0.0, 1.0, steps + 1)
                q_grid = (X * reserve).astype(int)
                tax_y = _sale_tax_rate_vec(q_grid, reserve)
                fig_tax = go.Figure()
                fig_tax.add_trace(go.Scattergl(x=X * 100.0, y=tax_y, mode='lines', name='Spread Rate'))
                fig_tax.update_layout(title='Sale Tax vs % of Supply Sold (per order)',
                                      xaxis_title='% of Current Supply Sold in Order',
                                      yaxis_title='Tax Rate',
                                      hovermode="x unified")
                st.plotly_chart(fig_tax, use_container_width=True)
        else:
            C = reserve
            if C <= 0:
                st.info("No circulating shares.")
            else:
                q_max = max(1, C // 10)
                q_axis = np.linspace(1, q_max, 200).astype(int)
                bd_C = np.vectorize(buy_delta,  otypes=[float])(np.full_like(q_axis, C))
                bd_C_minQ = np.vectorize(buy_delta,  otypes=[float])(C - q_axis)
                gross = bd_C - bd_C_minQ
                tax = _sale_tax_rate_vec(q_axis, np.full_like(q_axis, C))
                net = np.maximum(0.0, gross * (1.0 - tax))
                avg_net = net / np.maximum(1, q_axis)
                fig_eff = go.Figure()
                fig_eff.add_trace(go.Scattergl(x=q_axis, y=avg_net, mode='lines', name=f'{token} Avg Net Sell Price'))
                fig_eff.update_layout(title=f'Effective Avg Net Sell Price vs Quantity — {token}',
                                      xaxis_title='Quantity sold in a single order',
                                      yaxis_title='Avg Net Sell Price (USDC/share)',
                                      hovermode="x unified")
                st.plotly_chart(fig_eff, use_container_width=True)

st.divider()

# ================================
# History tabs
# ================================
st.subheader("📊 History")

txp = tx_norm_selected.copy()
users_df = users_df_all[["id", "username"]].copy()

tab1, tab2, tab3 = st.tabs([
    "💼 Portfolio Value (Selected Market)",
    "⭐ Points (Selected Market)",
    "👤 User Holdings by Market (All)"
])

with tab1:
    if txp.empty:
        st.info("No transactions available for this market.")
    else:
        reserves_state = {t: 0 for t in TOKENS_SELECTED}
        user_state = {
            int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS_SELECTED}}
            for _, r in users_df.iterrows()
        }
        records = []
        for _, r in txp.iterrows():
            uid = int(r.get("user_id")) if pd.notnull(r.get("user_id")) else None
            act = r.get("Action")
            tkn = r.get("Outcome")
            qty = int(r.get("Quantity") or 0)
            if uid is None or tkn not in TOKENS_SELECTED:
                continue
            if act == "Buy":
                delta = float(r.get("BuyAmt_Delta") or 0.0)
                user_state[uid]["balance"] -= delta
                user_state[uid]["holdings"][tkn] += qty
                reserves_state[tkn] += qty
            elif act == "Sell":
                delta = float(r.get("SellAmt_Delta") or 0.0)
                user_state[uid]["balance"] += delta
                user_state[uid]["holdings"][tkn] -= qty
                reserves_state[tkn] -= qty
            elif act == "Resolve":
                payout = float(r.get("SellAmt_Delta") or 0.0)
                user_state[uid]["balance"] += payout
                for u_id in user_state:
                    user_state[u_id]["holdings"] = {t: 0 for t in TOKENS_SELECTED}
                reserves_state = {t: 0 for t in TOKENS_SELECTED}
            prices = {t: 0.0 for t in TOKENS_SELECTED} if act == "Resolve" else {t: buy_curve(reserves_state[t]) for t in TOKENS_SELECTED}
            for u_id, s in user_state.items():
                pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS_SELECTED)
                pnl = pv - STARTING_BALANCE
                records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})
        port_df = pd.DataFrame(records)
        if port_df.empty:
            st.info("No portfolio history to display.")
        else:
            fig_port = px.line(port_df, x="Time", y="PortfolioValue", color="User",
                               title=f"Portfolio Value Over Time — Market {SELECTED_MARKET_ID}")
            st.plotly_chart(fig_port, use_container_width=True)

with tab2:
    if txp.empty:
        st.info("No transactions available for points.")
    else:
        df = txp.copy()
        reserves_state = {t: 0 for t in TOKENS_SELECTED}
        user_state = {
            int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS_SELECTED}}
            for _, r in users_df.iterrows()
        }
        vol_points = {int(r.id): 0.0 for _, r in users_df.iterrows()}
        rows = []
        def phase_mult(res_before: int) -> float:
            if res_before < EARLY_QUANTITY_POINT: return PHASE_MULTIPLIERS["early"]
            if res_before < MID_QUANTITY_POINT:   return PHASE_MULTIPLIERS["mid"]
            return PHASE_MULTIPLIERS["late"]
        for _, r in df.iterrows():
            uid = int(r.get("user_id")) if pd.notnull(r.get("user_id")) else None
            act = r.get("Action")
            tkn = r.get("Outcome")
            qty = int(r.get("Quantity") or 0)
            if uid is None or tkn not in TOKENS_SELECTED:
                continue
            add_points = 0.0
            if act == "Buy":
                buy_usd = float(r.get("BuyAmt_Delta") or 0.0)
                add_points = buy_usd * TRADE_POINTS_PER_USD * phase_mult(reserves_state[tkn])
            elif act == "Sell":
                sell_usd = float(r.get("SellAmt_Delta") or 0.0)
                add_points = sell_usd * TRADE_POINTS_PER_USD
            vol_points[uid] += float(add_points)
            if act == "Buy":
                delta = float(r.get("BuyAmt_Delta") or 0.0)
                user_state[uid]["balance"] -= delta
                user_state[uid]["holdings"][tkn] += qty
                reserves_state[tkn] += qty
            elif act == "Sell":
                delta = float(r.get("SellAmt_Delta") or 0.0)
                user_state[uid]["balance"] += delta
                user_state[uid]["holdings"][tkn] -= qty
                reserves_state[tkn] -= qty
            elif act == "Resolve":
                payout = float(r.get("SellAmt_Delta") or 0.0)
                user_state[uid]["balance"] += payout
                for u_id in user_state:
                    user_state[u_id]["holdings"] = {t: 0 for t in TOKENS_SELECTED}
                reserves_state = {t: 0 for t in TOKENS_SELECTED}
            prices = {t: 0.0 for t in TOKENS_SELECTED} if act == "Resolve" else {t: buy_curve(reserves_state[t]) for t in TOKENS_SELECTED}
            ts = r["Time"]
            for u_id, s in user_state.items():
                pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS_SELECTED)
                pnl_points = max(pv - STARTING_BALANCE, 0.0) * PNL_POINTS_PER_USD
                total_points = vol_points[u_id] + pnl_points
                rows.append({"Time": ts, "User": s["username"], "TotalPoints": total_points})
        points_plot_df = pd.DataFrame(rows)
        if points_plot_df.empty:
            st.info("No points history to display.")
        else:
            points_plot_df = points_plot_df[points_plot_df["User"].str.lower() != "admin"]
            fig_pts = px.line(points_plot_df, x="Time", y="TotalPoints", color="User",
                              title=f"Total Points Over Time — Market {SELECTED_MARKET_ID}",
                              line_group="User")
            st.plotly_chart(fig_pts, use_container_width=True)

with tab3:
    if tx_norm_all.empty:
        st.info("No transactions available across markets.")
    else:
        # Ensure 'User' exists here too
        df = ensure_user_column(tx_norm_all, users_df_all)
        df = df.dropna(subset=["Time"])
        df["market_id"] = pd.to_numeric(df.get("market_id"), errors="coerce")
        df = df.dropna(subset=["market_id"])
        df["market_id"] = df["market_id"].astype(int)

        # STRICTLY single-user selection (no 'All users' option)
        usernames = sorted(users_df_all["username"].astype(str).tolist())
        if not usernames:
            st.info("No users found.")
        else:
            pick_user = st.selectbox("Choose User", usernames, index=0)
            df = df[df["User"] == pick_user]

            if df.empty:
                st.info(f"No holdings history for user **{pick_user}**.")
            else:
                # Prepare running holdings per (market_id, Outcome) for this single user
                df = df.sort_values("Time")
                records = []
                running = {}
                # seed keys for user's outcomes across markets
                seed = df[["market_id", "Outcome"]].dropna().drop_duplicates()
                for _, row in seed.iterrows():
                    running[(int(row["market_id"]), str(row["Outcome"]))] = 0

                for _, r in df.iterrows():
                    mid = int(r.get("market_id")) if pd.notnull(r.get("market_id")) else None
                    tok = str(r.get("Outcome")) if pd.notnull(r.get("Outcome")) else None
                    act = r.get("Action")
                    qty = int(r.get("Quantity") or 0)
                    ts  = r.get("Time")
                    if mid is None or tok is None or pd.isna(ts):
                        continue
                    key = (mid, tok)
                    if key not in running:
                        running[key] = 0
                    if act == "Buy":
                        running[key] += qty
                    elif act == "Sell":
                        running[key] -= qty
                    elif act == "Resolve":
                        # zero all holdings for this market
                        for k in list(running.keys()):
                            if k[0] == mid:
                                running[k] = 0
                    records.append({"Time": ts, "Market": mid, "Outcome": tok, "Shares": running[key]})

                hold_df = pd.DataFrame(records)
                if hold_df.empty:
                    st.info(f"No holdings history for user **{pick_user}**.")
                else:
                    # Compute dynamic height: ~250px per row (2 markets per row)
                    n_markets = hold_df["Market"].nunique()
                    n_rows = math.ceil(n_markets / 2)
                    height = max(350, 250 * n_rows)
                    # ... after building hold_df and computing height ...

                    # Build the main holdings figure (faceted, 2 markets per row)
                    fig_h = px.line(
                        hold_df,
                        x="Time",
                        y="Shares",
                        color="Outcome",
                        facet_col="Market",
                        facet_col_wrap=2,   # <= keep 2 markets per row
                        title=f"Holdings for {pick_user} — Shares Over Time by Market",
                        height=height
                    )

                    # === Add a bold, dashed zero-baseline to every facet ===
                    # Create a tiny baseline frame spanning each market's time range
                    extent = (
                        hold_df.groupby("Market")["Time"]
                            .agg(["min", "max"])
                            .reset_index()
                            .rename(columns={"min": "tmin", "max": "tmax"})
                    )

                    baseline_rows = []
                    for _, r0 in extent.iterrows():
                        m = r0["Market"]
                        baseline_rows.append({"Market": m, "Time": r0["tmin"], "Shares": 0, "Outcome": "__BASELINE__"})
                        baseline_rows.append({"Market": m, "Time": r0["tmax"], "Shares": 0, "Outcome": "__BASELINE__"})

                    baseline_df = pd.DataFrame(baseline_rows)

                    # Build a separate PX figure for the baseline (so facets align),
                    # then copy its traces into the main figure with custom styling.
                    base_fig = px.line(
                        baseline_df,
                        x="Time",
                        y="Shares",
                        color="Outcome",
                        facet_col="Market",
                        facet_col_wrap=2
                    )

                    # Copy over only baseline traces and style them clearly
                    first_legend = True
                    for tr in base_fig.data:
                        # These traces are only for "__BASELINE__"; rename & restyle
                        tr.name = "Baseline (0)"
                        tr.showlegend = first_legend  # show legend only once
                        first_legend = False
                        tr.line.update(color="red", width=3, dash="dash")
                        tr.hovertemplate = "Baseline: %{y}"

                        fig_h.add_trace(tr)

                    # Make sure the baseline doesn’t get a color scale slot
                    # (keep original Outcome colors untouched in legend order)
                    fig_h.for_each_trace(
                        lambda t: t.update(legendgroup="Baseline") if t.name == "Baseline (0)" else ()
                    )

                    st.plotly_chart(fig_h, use_container_width=True)


                    # fig_h = px.line(
                    #     hold_df,
                    #     x="Time",
                    #     y="Shares",
                    #     color="Outcome",
                    #     facet_col="Market",
                    #     facet_col_wrap=2,   # <= max 2 markets in a single row
                    #     title=f"Holdings for {pick_user} — Shares Over Time by Market",
                    #     height=height
                    # )
                    # st.plotly_chart(fig_h, use_container_width=True)

st.divider()

# ================================
# Leaderboard (selected)
# ================================
st.subheader("🏆 Leaderboard (Selected Market)")

tx_lead = tx_norm_selected.copy()
if tx_lead.empty:
    st.info("No transactions in this market yet.")
else:
    # Prices now
    reserves_latest = m_selected.get('reserves') or []
    res_map_shares = {r["token"]: int(r["shares"]) for r in reserves_latest}
    tokens_for_lead = _extract_tokens(m_selected) or BINARY_OUTCOME_TOKENS
    price_now = {t: float(buy_curve(res_map_shares.get(t, 0))) for t in tokens_for_lead}

    # Minimal view for holdings rebuild
    txx = tx_lead[["Time","user_id","Action","Outcome","Quantity"]].copy()
    users_df_simple = users_df_all[["id","username","balance"]].copy()
    holdings_now = rebuild_holdings_now(txx, users_df_simple, tokens_for_lead)

    lb_rows = []
    for _, u in users_df_simple.iterrows():
        uid = int(u["id"])
        bal_now = float(u.get("balance", 0.0))
        shares_val = sum(holdings_now.get(uid, {}).get(t, 0) * price_now[t] for t in tokens_for_lead)
        pv_now = bal_now + shares_val
        lb_rows.append({"User": u["username"], "PortfolioValue": pv_now})
    latest = pd.DataFrame(lb_rows)
    if not latest.empty and "User" in latest.columns:
        latest = latest[latest["User"].str.lower() != "admin"]
    latest["PnL"] = latest["PortfolioValue"] - STARTING_BALANCE

    payouts_df = (
        tx_lead.loc[tx_lead["Action"] == "Resolve", ["user_id", "SellAmt_Delta"]]
        .groupby("user_id", as_index=False)
        .agg(Payout=("SellAmt_Delta", "sum"))
    )
    if not payouts_df.empty:
        payouts_df = payouts_df.merge(
            users_df_all[["id","username"]].rename(columns={"id":"user_id","username":"User"}),
            on="user_id", how="left"
        )[["User","Payout"]]
    else:
        payouts_df = pd.DataFrame(columns=["User","Payout"])
    payouts_df["Payout"] = pd.to_numeric(payouts_df["Payout"], errors="coerce").fillna(0.0)
    latest = latest.merge(payouts_df, on="User", how="left")
    latest["Payout"] = latest["Payout"].fillna(0.0)

    points_vol = compute_user_points(tx_lead, users_df_all[["id","username"]])
    pnl_points = latest[["User","PnL"]].copy()
    pnl_points["PnLPoints"] = pnl_points["PnL"].clip(lower=0.0) * PNL_POINTS_PER_USD
    pts = points_vol.merge(pnl_points[["User","PnLPoints"]], on="User", how="left")
    pts["PnLPoints"] = pts["PnLPoints"].fillna(0.0)
    pts["TotalPoints"] = pts["VolumePoints"] + pts["PnLPoints"]

    latest = latest.merge(
        pts[["User","VolumePoints","PnLPoints","TotalPoints"]],
        on="User", how="left"
    ).fillna({"VolumePoints":0.0, "PnLPoints":0.0, "TotalPoints":0.0})

    metric_choice = st.radio("Leaderboard metric (sort by):",
                             ["Portfolio Value", "PnL"],
                             horizontal=True, key="lb_metric")
    sort_key = {"Portfolio Value": "PortfolioValue", "PnL": "PnL"}[metric_choice]
    latest = latest.sort_values(sort_key, ascending=False, kind="mergesort")

    if latest.empty:
        st.info("No eligible users to display yet.")
    else:
        top_cols = st.columns(min(3, len(latest)))
        for i, (_, row) in enumerate(latest.head(3).iterrows()):
            delta_val = f"${row['PnL']:,.2f}" if row['PnL'] >= 0 else f"-${abs(row['PnL']):,.2f}"
            with top_cols[i]:
                st.metric(label=f"#{i+1} {row['User']}",
                          value=f"${row['PortfolioValue']:,.2f}",
                          delta=delta_val,
                          border=True)
                st.caption(f"Payout: ${row['Payout']:,.2f}")
                st.caption(f"Points: {row['TotalPoints']:,.0f}")
        st.dataframe(latest[["User","Payout","PortfolioValue","PnL","VolumePoints","PnLPoints","TotalPoints"]])
