# file: app.py
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, List

from api import api_get, api_post, APIError
from pathlib import Path
import os
from dotenv import load_dotenv

# ROOT = Path(__file__).resolve().parents[1]
# load_dotenv(ROOT / ".env")   # make variables visible to os.getenv
BASE = "http://concept.alkimiya.io/api"

st.caption(f"API_BASE_URL: {BASE}")



# ===========================================================
# Constants
DEFAULT_DECIMAL_PRECISION = 2
BASE_EPSILON = 1e-4
MARKET_DURATION_DAYS = 5
END_TS = "2025-09-07 00:00"
MAX_SHARES = 5000000 #10M
STARTING_BALANCE = 10000.0

MARKET_ID = 1

MARKET_QUESTION = "Will the NASA TOMEX+ sounding rocket mission successfully launch from the Wallops Range by Sunday, August 24, 2025?" # @To-do: Replace
RESOLUTION_NOTE = (
'This market will resolve to "YES" if the NASA TOMEX+ sounding rocket is successfully launched from the Wallops Range in New Mexico by 11:59 PM EDT on Sunday, August 24, 2025.' 
'"Successfully launched" means the rocket leaves the launch pad as intended, regardless of the outcome of the mission itself.'
'The market will resolve to "NO" if the launch is scrubbed, delayed past the deadline, or if there is no successful launch attempt by the end of the day on Sunday. Resolution will be based on official announcements from NASA and live coverage from reliable news sources.'
) # @To-do: Replace
# TOKENS = ["<4200", "4200-4600", ">4600"]
TOKENS = ["YES", "NO"]

# Whitelisted usernames and admin reset control
WHITELIST = {"admin", "rui", "haoye", "leo", "steve", "wenbo", "sam", "sharmaine", "mariam", "henry", "guard", "victor", "toby"}

# Inflection Points
EARLY_QUANTITY_POINT = 270
MID_QUANTITY_POINT = 630

# ==== Points System (tunable) ====
TRADE_POINTS_PER_USD = 10.0   # Buy & Sell volume → 10 pts per $1 traded
PNL_POINTS_PER_USD   = 5.0    # Only for positive PnL → 5 pts per $1 profit
EARLY_MULTIPLIER     = 1.5    # Applies to Buy $ when reserve < EARLY_QUANTITY_POINT
MID_MULTIPLIER       = 1.25   # Applies to Buy $ when reserve < MID_QUANTITY_POINT
LATE_MULTIPLIER      = 1.0    # Applies to Buy $ when reserve >= MID_QUANTITY_POINT

PHASE_MULTIPLIERS = {
    "early": EARLY_MULTIPLIER,
    "mid": MID_MULTIPLIER,
    "late": LATE_MULTIPLIER,
}
# ===========================================================




@st.cache_resource
def _session():
    s = requests.Session()             # keep-alive
    s.headers["Accept-Encoding"] = "gzip, deflate"
    return s

# If you can, update api.py to use the session above; otherwise:
def _get(path: str, timeout: float = 5.0):
    url = f"{BASE.rstrip('/')}{path}"
    r = _session().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5)
def fetch_snapshot(market_id: int, user_id: int | None):
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_market   = ex.submit(_get, f"/market/{market_id}")
        fut_tx       = ex.submit(_get, "/tx")
        # fut_user     = ex.submit(_get, f"/users/{user_id}") if user_id else None
        fut_users    = ex.submit(_get, "/users")
        fut_holdings = ex.submit(_get, f"/holdings/{user_id}") if user_id else None

    # print("See", fut_user, fut_users)
    return {
        "market": fut_market.result(),
        "tx": fut_tx.result(),
        "users": fut_users.result() if fut_users else [],
        # 'user': fut_user.result() if fut_users else None,
        "holdings": fut_holdings.result() if fut_holdings else []
    }

# Use it ONCE
print(st.session_state.get("user_id"))
snap = fetch_snapshot(MARKET_ID, st.session_state.get("user_id"))
m = snap["market"]
tx_raw = pd.DataFrame(snap["tx"])
users_df_all = pd.DataFrame(snap["users"])

user_holdings_list = snap["holdings"]

# if "user_id" in st.session_state:
#     user_id = st.session_state.user_ud
#     user_state = api_get(f"/users/{user_id}")

def interpret_market(srv: dict):
    """
    srv: dict containing keys like start_ts, end_ts, resolved, winner_token, resolved_ts
    
    Returns:
        server_active (bool)
        server_why (str)
    """
    now = datetime.utcnow()
    end_ts = datetime.fromisoformat(srv.get("end_ts"))
    resolved = int(srv.get("resolved") or 0)

    if resolved:
        return False, "resolved"
    elif now >= end_ts:
        return False, "market ended"
    else:
        return True, "active"

def rebuild_holdings_now(tx_norm: pd.DataFrame, users_df: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    # initialize all users x tokens = 0
    h = {int(uid): {t: 0 for t in TOKENS} for uid in users_df["id"]}

    if tx_norm.empty:
        return h

    tx_sorted = tx_norm.sort_values("Time")
    for _, r in tx_sorted.iterrows():
        uid = int(r.get("user_id") or 0)
        act = r.get("Action")
        tkn = r.get("Outcome")
        qty = int(r.get("Quantity") or 0)

        if act == "Resolve":
            # all holdings wiped
            h = {int(u): {t: 0 for t in TOKENS} for u in users_df["id"]}
            continue

        # ignore malformed rows
        if uid not in h or tkn not in TOKENS:
            continue

        if act == "Buy":
            h[uid][tkn] += qty
        elif act == "Sell":
            h[uid][tkn] -= qty

    return h  

# Streamlit Setup
st.set_page_config(page_title="42: Simulatoooor (Global)", layout="wide")
st.title("42: Dynamic Pari-mutuel — Global PVP")

st.subheader(f":blue[{MARKET_QUESTION}]")


server_active, server_why = interpret_market(m)
# Use server_active for button disabled
disabled = ('user_id' not in st.session_state) or (not server_active)

def st_display_market_status(active, reason=''):
    if active:
        st.badge("Active", icon=":material/check:", color="green")
    else:
        st.markdown(f":gray-badge[Inactive] ({server_why})")

# ===========================================================
# # Math Helpers (curves)
# def buy_curve(x: float) -> float:
#     return x**(1/4) + x / 400

# def sell_curve(x: float) -> float:
#     return ((x - 500)/40) / ((8 + ((x - 500)/80)**2)**0.5) + (x - 500)/300 + 3.6

def buy_curve(x: float) -> float:
    """price to buy an infinitesimal share at reserve x"""
    return (float(x) ** (1.0 / 3.0)) / 1000.0 + 0.1

# def sell_curve(x: float) -> float:
#     """price to sell an infinitesimal share at reserve x (clamped >= 0)"""
#     t = (float(x) - 500_000.0) / 1_000_000.0
#     p = 1.0 / (4.0 * (0.8 + math.exp(-t))) - 0.05
#     return max(p, 0.0)

def buy_delta(x: float) -> float:
    """
    ∫ buy_curve dx = (3/4000) * x^(4/3) + x/10   (C=0)
    Total USDC to move reserve from 0 → x.
    """
    x = float(x)
    return (3.0 / 4000.0) * (x ** (4.0 / 3.0)) + x / 10.0

# def sell_delta(x: float) -> float:
#     """
#     ∫ sell_curve dx = 312500 * ln(1 + 0.8 * e^{(x-500000)/1e6}) - 0.05*x   (C=0)
#     Use log1p for stability.
#     """
#     x = float(x)
#     t = (x - 500_000.0) / 1_000_000.0
#     return 312_500.0 * math.log1p(0.8 * math.exp(t)) - 0.05 * x


# ---------------------------
# Curve formulas (vectorized + viz-only)
# ---------------------------
def viz_buy_curve_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return np.cbrt(x) / 1000.0 + 0.1

def viz_sell_curve_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    t = (x - 500_000.0) / 1_000_000.0
    p = 1.0 / (4.0 * (0.8 + np.exp(-t))) - 0.05
    return np.maximum(p, 0.0)

def viz_buy_curve(x: int | float) -> float:
    return (float(x) ** (1.0 / 3.0)) / 1000.0 + 0.1

def viz_sell_curve(x: int | float) -> float:
    t = (float(x) - 500_000.0) / 1_000_000.0
    p = 1.0 / (4.0 * (0.8 + math.exp(-t))) - 0.05
    return max(p, 0.0)

# ===== Sale Tax Helpers (new) =====
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def sale_tax_rate(q: int, C: int) -> float:
    """
    q: number of shares sold in *this order*
    C: total circulating shares (reserve) before the sale

    Tax = min(1, (1.15 - 1.3/(1 + e^(4*(q/C) - 2))))

    We clamp to [0,1] to keep it sane even if the scaling makes it big.
    """
    if C <= 0 or q <= 0:
        return 0.0
    X = q / float(C)  # fraction of supply this order is selling
    base = 1.15 - 1.3 / (1.0 + math.e ** (4.0 * X - 2.0))
    # scale = C * math.e ** ((C / 100000.0 - 1.0) / 10000.0)
    # tax = base * scale
    tax = base
    return _clamp01(tax)

def sell_proceeds_net(reserve: int, q: int) -> float:
    """Net USDC user receives after the order-level sale tax."""
    if q <= 0 or reserve <= 0:
        return 0.0
    q = min(q, reserve)
    gross = buy_delta(reserve) - buy_delta(reserve - q)
    tax = sale_tax_rate(q, reserve)
    net = gross * (1.0 - tax)
    return max(0.0, float(net))

def current_marginal_sell_price_after_tax(reserve: int) -> float:
    """
    A single-share 'instant' effective price (for UI display).
    Uses q=1 to estimate marginal price after tax.
    """
    if reserve <= 0:
        return 0.0
    gross1 = buy_delta(reserve) - buy_delta(reserve - 1)
    tax1 = sale_tax_rate(1, reserve)
    return max(0.0, float(gross1 * (1.0 - tax1)))

def sell_gross_from_bonding(reserve: int, q: int) -> float:
    """Bonding-curve gross proceeds (no tax), using your buy integral as the primitive."""
    if q <= 0 or reserve <= 0:
        return 0.0
    q = min(q, reserve)
    return float(buy_delta(reserve) - buy_delta(reserve - q))

# vectorized tax (for charts)
_sale_tax_rate_vec = np.vectorize(sale_tax_rate, otypes=[float])

def metrics_from_qty(x: int, q: int):
    """
    Returns (in order):
      buy_price         = buy spot after adding q
      sell_price        = marginal 1-share sell price after tax (display only)
      buy_amt_delta     = USDC to buy q
      sell_amt_delta    = USDC received to sell q after tax
      sell_tax_rate_used= order-level tax applied for selling q out of reserve x (0..1)
    """
    q = int(max(0, q))
    new_x = x + q

    buy_price = buy_curve(new_x)
    sell_price = current_marginal_sell_price_after_tax(x)

    buy_amt_delta = buy_delta(new_x) - buy_delta(x)

    # Clamp sell qty to circulating reserve for proceeds + tax computation
    q_eff = min(q, x if x > 0 else 0)
    if q_eff > 0 and x > 0:
        tax_used = sale_tax_rate(q_eff, x)
        sell_amt_delta = sell_proceeds_net(x, q_eff)
    else:
        tax_used = 0.0
        sell_amt_delta = 0.0

    return buy_price, sell_price, buy_amt_delta, sell_amt_delta, float(tax_used)


def qty_from_buy_usdc(reserve: int, usd: float) -> int:
    if usd <= 0:
        return 0
    # initial guess: linear approximation
    q = usd / max(buy_curve(reserve), 1e-9)
    q = max(0.0, min(q, MAX_SHARES - reserve))

    for _ in range(12):
        f  = (buy_delta(reserve + q) - buy_delta(reserve)) - usd
        fp = max(buy_curve(reserve + q), 1e-9)  # df/dq
        step = f / fp
        q -= step
        # clamp
        if q < 0.0: q = 0.0
        if q > (MAX_SHARES - reserve): q = float(MAX_SHARES - reserve)
        if abs(step) < 1e-6:
            break
    return int(q)

def qty_from_sell_usdc(reserve: int, usd: float) -> int:
    if usd <= 0.0 or reserve <= 0:
        return 0

    # initial guess using current *net* marginal price
    p0_net = max(current_marginal_sell_price_after_tax(reserve), 1e-12)
    q_guess = min(reserve, usd / p0_net)

    # robust binary search on net proceeds
    lo, hi = 0, int(reserve)
    # tighten bounds around the guess to speed up
    lo = max(0, int(q_guess * 0.25))
    hi = min(int(reserve), max(lo, int(q_guess * 1.75)))

    while lo < hi:
        mid = (lo + hi + 1) // 2
        net = sell_proceeds_net(reserve, mid)
        if net <= usd:
            lo = mid
        else:
            hi = mid - 1
    return int(lo)

def format_usdc_compact(value: float) -> str:
    """Compact money: $999.99, $1.2K, $12.3M, $4B, $5T (no trailing .0)."""
    sign = "-" if value < 0 else ""
    v = abs(float(value))

    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for i, (scale, suf) in enumerate(units):
        if v >= scale:
            q = v / scale
            # adaptive decimals
            dec = 2 if q < 10 else (1 if q < 100 else 0)
            s = f"{q:.{dec}f}"

            # If rounding hit 1000 of current unit, bump to the next unit
            if float(s) >= 1000 and i > 0:
                next_scale, next_suf = units[i - 1]
                q = v / next_scale
                dec = 2 if q < 10 else (1 if q < 100 else 0)
                s = f"{q:.{dec}f}"
                suf = next_suf

            # strip trailing zeros / dot
            s = s.rstrip("0").rstrip(".")
            return f"{sign}${s}{suf}"

    # < 1,000 — show standard money
    return f"{sign}${v:,.2f}"

def load_users_df() -> pd.DataFrame:
    try:
        data = snap["users"]
    except Exception:
        return pd.DataFrame(columns=["id", "username", "balance"])

    # Coerce to list
    if data is None:
        data = []
    elif isinstance(data, dict):
        # handle wrappers or single objects
        if isinstance(data.get("users"), list):
            data = data["users"]
        elif isinstance(data.get("results"), list):
            data = data["results"]
        else:
            data = [data]

    df = pd.DataFrame(data)

    # Create/rename to expected columns
    if "id" not in df.columns and "user_id" in df.columns:
        df = df.rename(columns={"user_id": "id"})
    if "username" not in df.columns:
        for alt in ("user", "name", "User", "Username"):
            if alt in df.columns:
                df = df.rename(columns={alt: "username"})
                break
    if "balance" not in df.columns:
        for alt in ("Balance", "cash", "usdc", "wallet", "Balance After"):
            if alt in df.columns:
                df = df.rename(columns={alt: "balance"})
                break

    # Ensure all three exist even if empty
    for col, dtype in (("id", "int64"), ("username", "object"), ("balance", "float64")):
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype)

    return df[["id", "username", "balance"]]

# Binary searches

# def qty_from_buy_usdc(reserve: int, usdc_amount: float) -> int:
#     low, high = 0.0, 10000.0
#     eps = BASE_EPSILON
#     while high - low > eps:
#         mid = (low + high) / 2
#         delta = buy_delta(reserve + mid) - buy_delta(reserve)
#         if delta < usdc_amount:
#             low = mid
#         else:
#             high = mid
#     return max(0, math.floor(low))


# def qty_from_sell_usdc(reserve: int, usdc_amount: float) -> int:
#     low, high = 0.0, float(reserve)
#     eps = BASE_EPSILON
#     while high - low > eps:
#         mid = (low + high) / 2
#         delta = sell_delta(reserve) - sell_delta(reserve - mid)
#         if delta < usdc_amount:
#             low = mid
#         else:
#             high = mid
#     return max(0, math.floor(low))

# ===========================================================
# Double check
# Option A: tiny adapter (keep your original compute_cost_basis signature)
def compute_cost_basis_from_df(df: pd.DataFrame) -> dict:
    basis = {t: {'shares': 0, 'cost': 0.0, 'avg_cost': 0.0} for t in TOKENS}
    if df.empty:
        return basis
    df = df.sort_values("ts")
    for _, r in df.iterrows():
        act = r["action"]; t = r["token"]; q = int(r["qty"] or 0)
        if act == "Buy":
            add_cost = float(r.get("buy_delta") or 0.0)
            basis[t]['cost'] += add_cost
            basis[t]['shares'] += q
        elif act == "Sell":
            sh = basis[t]['shares']
            if sh > 0 and q > 0:
                avg = basis[t]['cost'] / sh
                used = min(q, sh)
                basis[t]['cost'] -= avg * used
                basis[t]['shares'] -= used
    for t in TOKENS:
        sh = basis[t]['shares']
        basis[t]['avg_cost'] = (basis[t]['cost'] / sh) if sh > 0 else 0.0
    return basis


# For points computation
def compute_user_points(tx_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      User, VolumePoints, PnLPoints, TotalPoints
    Volume points:
      - 10 pts per $1 traded (Buy or Sell)
      - Buy $ gets a phase multiplier based on *reserve before* the Buy:
          < EARLY_QUANTITY_POINT → 1.5x
          < MID_QUANTITY_POINT   → 1.25x
          >= MID_QUANTITY_POINT  → 1.0x
    PnL points:
      - 5 pts per $1 profit (only positive PnL), using the latest portfolio PnL snapshot
        we compute below when we build the leaderboard.
    """
    # Map id -> username
    id_to_user = dict(zip(users_df["id"], users_df["username"]))

    # Running reserves per token to know the *phase at time of each buy*
    reserves_state = {t: 0 for t in TOKENS}

    # Per-user accumulators
    vol_points = {uid: 0.0 for uid in users_df["id"]}
    # We'll add PnL points later (when we know latest PnL)

    # Process chronologically
    tdf = tx_df.copy()
    tdf["Time"] = pd.to_datetime(tdf["Time"], format="ISO8601", utc=True, errors="coerce")
    tdf = tdf.sort_values("Time")

    for _, r in tdf.iterrows():
        uid  = int(r["user_id"])
        act  = r["Action"]
        tkn  = r["Outcome"]
        qty  = int(r["Quantity"])

        if act == "Buy":
            # Dollars traded for this buy
            buy_usd = float(r["BuyAmt_Delta"] or 0.0)

            # Determine phase by reserve BEFORE adding qty
            r_before = reserves_state.get(tkn, 0)
            if r_before < EARLY_QUANTITY_POINT:
                mult = EARLY_MULTIPLIER
            elif r_before < MID_QUANTITY_POINT:
                mult = MID_MULTIPLIER
            else:
                mult = LATE_MULTIPLIER

            vol_points[uid] += buy_usd * TRADE_POINTS_PER_USD * mult

            # Update state
            reserves_state[tkn] = r_before + qty

        elif act == "Sell":
            sell_usd = float(r["SellAmt_Delta"] or 0.0)
            vol_points[uid] += sell_usd * TRADE_POINTS_PER_USD

            # Update state (post-sell)
            reserves_state[tkn] = reserves_state.get(tkn, 0) - qty

        # "Resolve" does not change volume points

    # Build dataframe (PnL points will be joined later)
    out = pd.DataFrame({
        "user_id": list(vol_points.keys()),
        "User":    [id_to_user[uid] for uid in vol_points.keys()],
        "VolumePoints": [vol_points[uid] for uid in vol_points.keys()]
    })

    return out


def compute_points_timeline(txp: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a timeline of points per user at each transaction timestamp.
    TotalPoints(t) = CumulativeVolumePoints(t) + max(PnL(t), 0) * PNL_POINTS_PER_USD
    PnL(t) is instantaneous (not cumulative) based on reconstructed portfolio valuation.

    Phase rules (for BUY only), decided on reserve *before* applying the buy:
        reserve < EARLY_QUANTITY_POINT -> early
        EARLY_QUANTITY_POINT <= reserve < MID_QUANTITY_POINT -> mid
        reserve >= MID_QUANTITY_POINT -> late
    """

    # Defensive copy & types
    df = txp.copy()
    if df.empty:
        return pd.DataFrame(columns=["Time","User","VolumePointsCum","PnLPointsInstant","TotalPoints"])

    df["Time"] = pd.to_datetime(df["Time"], format="ISO8601", utc=True, errors="coerce")

    # States to reconstruct reserves + user portfolios
    reserves_state = {t: 0 for t in TOKENS}
    user_state = {
        int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS}}
        for _, r in users_df.iterrows()
    }

    # Cumulative volume points per user
    vol_points = {int(r.id): 0.0 for _, r in users_df.iterrows()}

    rows = []

    for _, r in df.iterrows():
        uid = int(r["user_id"])
        act = r["Action"]
        tkn = r["Outcome"]
        qty = int(r["Quantity"])

        # --- Volume points increment (if any) ---
        add_points = 0.0
        if act == "Buy":
            buy_usd = float(r["BuyAmt_Delta"] or 0.0)
            # Phase by reserve BEFORE buy
            reserve_before = reserves_state[tkn]
            if reserve_before < EARLY_QUANTITY_POINT:
                mult = PHASE_MULTIPLIERS["early"]
            elif reserve_before < MID_QUANTITY_POINT:
                mult = PHASE_MULTIPLIERS["mid"]
            else:
                mult = PHASE_MULTIPLIERS["late"]
            add_points = buy_usd * TRADE_POINTS_PER_USD * mult

        elif act == "Sell":
            sell_usd = float(r["SellAmt_Delta"] or 0.0)
            # Sells: volume points = 10 pts / $1 (no multiplier)
            add_points = sell_usd * TRADE_POINTS_PER_USD

        # accumulate
        vol_points[uid] += float(add_points)

        # --- Apply the transaction to portfolio/reserves reconstruction ---
        if act == "Buy":
            delta = float(r["BuyAmt_Delta"] or 0.0)
            user_state[uid]["balance"] -= delta
            user_state[uid]["holdings"][tkn] += qty
            reserves_state[tkn] += qty

        elif act == "Sell":
            delta = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += delta
            user_state[uid]["holdings"][tkn] -= qty
            reserves_state[tkn] -= qty

        elif act == "Resolve":
            # credit this user's payout (logged as SellAmt_Delta)
            payout = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += payout

            # wipe all holdings and reserves
            for u_id in user_state:
                user_state[u_id]["holdings"] = {t: 0 for t in TOKENS}
            reserves_state = {t: 0 for t in TOKENS}

        # --- Price snapshot (use Buy curve for valuation consistency) ---
        if act == "Resolve":
            prices = {t: 0.0 for t in TOKENS}
        else:
            prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}

        # --- Build a row per user at this timestamp ---
        ts = r["Time"]
        for u_id, s in user_state.items():
            pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
            pnl_instant = max(pv - STARTING_BALANCE, 0.0)
            pnl_points = pnl_instant * PNL_POINTS_PER_USD
            total_points = vol_points[u_id] + pnl_points

            rows.append({
                "Time": ts,
                "user_id": u_id,
                "User": s["username"],
                "VolumePointsCum": vol_points[u_id],
                "PnLPointsInstant": pnl_points,
                "TotalPoints": total_points,
            })

    return pd.DataFrame(rows)

                   # ← Fetch Market
market_start = pd.to_datetime(m["start_ts"])
market_end_db = pd.to_datetime(m["end_ts"])
winner_token = m.get('winner_token')
resolved_flag = int(m.get("resolved") or 0)
resolved_ts = m.get('resolved_ts')

# Winner banner (now safe)
if resolved_flag == 1 and m.get("winner_token"):
    st.success(f"✅ Resolved — Winner: {m['winner_token']}")

# ===========================================================
# Sidebar user auth with whitelist (persist to session, rerun)


with st.sidebar:
    st.header("User")
    username_input = st.text_input("Enter username", key="username_input")
    if username_input and username_input not in WHITELIST:
        st.error("Username not whitelisted.")
    join_disabled = bool(username_input) and (username_input not in WHITELIST)

    # if st.button("Join / Load", disabled=join_disabled):
    #     if not username_input:
    #         st.warning("Please enter a username.")
    #     else:
    #         with closing(get_conn()) as conn, conn:
    #             c = conn.cursor()
    #             # enforce max 10 users at create time
    #             row = c.execute("SELECT id, balance FROM users WHERE username=?", (username_input,)).fetchone()
    #             if not row:
    #                 cnt = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    #                 if cnt >= 20:
    #                     st.error("Max 20 users reached. Try another time.")
    #                     st.stop()
    #                 c.execute(
    #                     "INSERT INTO users(username,balance,created_at) VALUES(?,?,?)",
    #                     (username_input, STARTING_BALANCE, datetime.utcnow().isoformat()),
    #                 )
    #                 user_id = c.lastrowid
    #                 for t in TOKENS:
    #                     c.execute("INSERT OR IGNORE INTO holdings(user_id,token,shares) VALUES(?,?,0)", (user_id, t))
    #             else:
    #                 user_id = row["id"]
    #         st.session_state.user_id = int(user_id)
    #         st.session_state.username = username_input
    #         st.rerun()
    if st.button("Join / Load", disabled=join_disabled):
        if not username_input:
            st.warning("Please enter a username.")
        else:
            try:
                resp = api_post("/users", json={"username": username_input})
                st.session_state.user_id = int(resp["id"])
                st.session_state.username = resp["username"]
                st.rerun()
            except APIError as e:
                st.error(str(e))

    if "user_id" in st.session_state and st.button("Logout", type="primary"):
        for k in ("user_id", "username"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# --- helpers: keep the session valid even after admin resets ---
def ensure_user_session():
    """If user_id is in session, make sure it still exists on the server.
    If not, try to recreate via username; otherwise clear the session cleanly."""
    if "user_id" not in st.session_state:
        return None

    try:
        u = api_get(f"/users/{st.session_state.user_id}")
        # keep a fresh balance copy
        st.session_state.balance = float(u.get("balance", 0.0))
        return u
    except APIError:
        # User id is gone (likely admin wiped users). Try to re-create using the saved username.
        uname = st.session_state.get("username")
        if uname:
            try:
                resp = api_post("/users", json={"username": uname})
                st.session_state.user_id = int(resp["id"])
                st.session_state.balance = float(resp.get("balance", 0.0))
                return resp
            except APIError:
                pass  # fall through to clearing the session

        # Could not recover -> clear session and prompt to join again
        for k in ("user_id", "username", "balance"):
            st.session_state.pop(k, None)
        st.warning("Your account session was reset. Please Join/Load again.")
        st.stop()
        return None  # not reached

# call this once early in your script (after handling Join/Load button):
ensure_user_session()


# Refresh session balance snapshot
if "user_id" in st.session_state:
    try:
        u = api_get(f"/users/{st.session_state.user_id}")
        st.session_state.balance = float(u["balance"])
    except APIError as e:
        st.warning(f"Could not refresh balance: {e}")



# ===========================================================
# --- Admin controls (toggle Start vs Reset based on current state) ---
if "user_id" in st.session_state and st.session_state.get("username") == "admin":
    st.sidebar.subheader("Admin Controls")

    # Admin password (kept in session only)
    admin_pwd = st.sidebar.text_input("Admin password", type="password", key="admin_pwd")
    pw_missing = not bool(admin_pwd)

    if server_active:
 

        st.sidebar.markdown("### Resolve Market")
        winner = st.sidebar.selectbox("Winning outcome", TOKENS, key="winner_select")
        if st.sidebar.button("Resolve Now", disabled=(not winner) or pw_missing, type="secondary", key="btn_resolve"):
            if pw_missing:
                st.error("Admin password required.")
            else:
                try:
                    resp = api_post(
                        f"/admin/{MARKET_ID}/resolve",
                        json={"password": admin_pwd, "winner_token": winner},
                    )
                    st.success(
                        f"Market resolved. Winner: {winner}. "
                        f"Payout pool: ${float(resp.get('payout_pool', 0.0)):,.2f}"
                    )
                    st.rerun()
                except APIError as e:
                    st.error(str(e))

    else:
               # Market running → show RESET + Resolve
        wipe_users = st.sidebar.checkbox(
            "Reset all users (wipe accounts)", value=False, key="wipe_users_reset"
        )

        if st.sidebar.button("Reset Market", key="btn_reset_market", disabled=pw_missing):
            if pw_missing:
                st.error("Admin password required.")
            else:
                try:
                    api_post(
                        f"/admin/{MARKET_ID}/reset",
                        json={"password": admin_pwd, "wipe_users": bool(wipe_users)},
                    )
                    st.success("Market reset. It is now inactive.")
                    if wipe_users:
                        for k in ("user_id", "username", "balance"):
                            st.session_state.pop(k, None)
                    st.rerun()
                except APIError as e:
                    st.error(str(e))

 
now = datetime.utcnow()
hard_end = datetime.fromisoformat(END_TS)   # <-- enforce constant cutoff

# Market is active only if:
# 1) within DB market window
# 2) before hard END_TS cutoff
# 3) not resolved
active = (market_start <= now <= market_end_db) and (now <= hard_end) and (resolved_flag == 0)

    # Sidebar market status row
with st.sidebar:
    status_cols = st.columns(2)
    with status_cols[0]:
        st.write("**Market Status:**")
    with status_cols[1]:
       st_display_market_status(active)


st.sidebar.markdown(f"Start (UTC): {market_start:%Y-%m-%d %H:%M}")
# st.sidebar.markdown(f"End (UTC): {market_end:%Y-%m-%d %H:%M}")
st.sidebar.markdown(f"End (UTC): {END_TS}")

if 'user_id' not in st.session_state:
    st.warning("Join with a username on the left to trade.")

# ===========================================================
# Trading UI
    if resolved_flag == 1 and winner_token:
        # Market resolved - show outcome highlight
        st.markdown(
            f"""
            <div style="
                background-color: orange;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #b2f2bb;
                margin-bottom: 1rem;
            ">
                <h4 style="margin-top: 0;">✅ Market Resolved</h4>
                <p>
                    The winning outcome is <b style="color: green;">{winner_token}</b>.
                    Settlement occurred on <b>{pd.to_datetime(resolved_ts).strftime('%Y-%m-%d %H:%M UTC')}</b>.
                </p>
                <p>All holdings in this outcome have been paid out proportionally from the total USDC pool.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background-color: grey;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #a5b4fc;
                margin-bottom: 1rem;
            ">
                <h4 style="margin-top: 0;">📜 Resolution Rules</h4>
                <p>
                    The market will resolve at <b>{END_TS}</b>.
                    The winning outcome will receive the <b>entire USDC pool</b>,
                    distributed <i>pro-rata</i> to holders based on their share count.
                </p>
                <p>
                    <b>Resolution Note:</b> {RESOLUTION_NOTE}
                </p>
                <p>
                    After resolution, all holdings will be cleared and balances updated automatically.
                </p>
                <h5>🔗 Resolution Sources/Resources:</h5>
                <ul>
                    <li><a href="https://www.nasa.gov/blogs/wallops/2025/08/15/turbulence-at-edge-of-space/" target="_blank">
                        NASA Blog: Turbulence at Edge of Space (Aug 15, 2025)</a></li>
                    <li><a href="https://www.nasa.gov/blogs/wallops-range/" target="_blank">
                        NASA Wallops Range Blog</a></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()
        
if 'user_id' in st.session_state:
    # --- live snapshots via API ---
    # u = user_state
    bal = float(u["balance"])

    # user_holdings_list = user_holdings_list
    user_holdings = {h["token"]: int(h["shares"]) for h in user_holdings_list}

    reserves_list = m.get('reserves')
    global_reserves = {r["token"]: int(r["shares"]) for r in reserves_list}

    # Prices from current reserves (buy curve)
    prices = {t: float(buy_curve(global_reserves.get(t, 0))) for t in TOKENS}
    holding_values = {t: user_holdings.get(t, 0) * prices[t] for t in TOKENS}

    holdings_total_value = sum(holding_values.values())
    portfolio_value = bal + holdings_total_value

    # --- Styles ---
    st.markdown("""
    <style>
      .pill { padding:6px 10px;border-radius:8px;font-size:12px;border:1px solid rgba(0,0,0,0.1);
              display:inline-block;margin-bottom:6px;background:rgba(0,0,0,0.02);}
      .pill strong { font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Your Account")
        st.metric("Username", st.session_state.get("username",""))

        with st.container():
            overall_cols = st.columns(2)

            # --- Portfolio card ---
            with overall_cols[0]:
                with st.container(border=True):
                    st.subheader("Portfolio")
                    ucols = st.columns(3)
                    with ucols[0]:
                        st.metric("Overall Value (USD)", f"{portfolio_value:,.2f}")
                    with ucols[1]:
                        st.metric("Balance (USDC)", f"{bal:,.2f}")
                    with ucols[2]:
                        st.metric("Shares Holdings Value (USD)", f"{holdings_total_value:,.2f}")

            # --- Points card (Volume + PnL points) ---
            with overall_cols[1]:
                # Pull all tx from API and normalize columns for compute_user_points()
                try:
                    txp_pts = tx_raw
                except APIError as e:
                    st.error(f"Could not load transactions: {e}")
                    txp_pts = pd.DataFrame()

                my_vol_points = 0.0
                if not txp_pts.empty:
                    # Normalize names to the ones your helper expects
                    colmap = {
                        "ts": "Time",
                        "user_id": "user_id",
                        "user": "User",
                        "action": "Action",
                        "token": "Outcome",
                        "qty": "Quantity",
                        "buy_delta": "BuyAmt_Delta",
                        "sell_delta": "SellAmt_Delta",
                    }
                    txp_pts = txp_pts.rename(columns=colmap)

                    # Types / defaults (in case backend omits a field)
                    for c in ["BuyAmt_Delta", "SellAmt_Delta", "Quantity"]:
                        if c not in txp_pts.columns:
                            txp_pts[c] = 0.0
                    if "Time" in txp_pts.columns:
                        txp_pts["Time"] = pd.to_datetime(txp_pts["Time"], format="ISO8601", utc=True, errors="coerce")

                    # Users list (prefer API; fallback from tx)
                    try:
                        users_list = users_df_all  # if implemented
                        users_df_pts = pd.DataFrame(users_list)[["id", "username"]]
                        
                    except APIError:
                        # Fallback: derive from tx
                        users_df_pts = (
                            txp_pts[["user_id", "User"]]
                            .dropna()
                            .drop_duplicates()
                            .rename(columns={"user_id": "id", "User": "username"})
                        )

                    if not users_df_pts.empty:
                        points_live = compute_user_points(txp_pts, users_df_pts)
                        me_row = points_live[points_live["user_id"] == st.session_state["user_id"]]
                        if not me_row.empty:
                            my_vol_points = float(me_row["VolumePoints"].iloc[0])

                # PnL points from current snapshot
                my_pnl_now = max(portfolio_value - STARTING_BALANCE, 0.0)
                my_pnl_points = my_pnl_now * PNL_POINTS_PER_USD
                my_total_points = my_vol_points + my_pnl_points

                with st.container(border=True):
                    st.subheader("Points")
                    pcols = st.columns(3)
                    with pcols[0]:
                        st.metric("Volume Points", f"{my_vol_points:,.0f}")
                    with pcols[1]:
                        st.metric("PnL Points", f"{my_pnl_points:,.0f}")
                    with pcols[2]:
                        st.metric("Total Points", f"{my_total_points:,.0f}")

        # --- Per-token breakdown ---
        with st.container(border=True):
            st.subheader("Share Holdings")
            br_cols = st.columns(len(TOKENS))  # match the number of tokens
            for i, token in enumerate(TOKENS):
                shares = user_holdings.get(token, 0)
                price = prices[token]
                val = holding_values[token]
                with br_cols[i]:
                    st.caption(f"Outcome :blue[{token}]")
                    st.text(f"Shares: {shares}")
                    st.text(f"Price: {price:.4f}")
                    st.text(f"Value: ${val:,.2f}")

# === Cost basis & per‑token PnL table ===
user_id = st.session_state.get("user_id")
if user_id is not None:
    # 1) Pull this user's transactions from API and map columns to what the cost-basis helper expects
    tx_user = pd.DataFrame(api_get(f"/tx?user_id={user_id}"))

    # If the API returns the "display" schema, remap to raw-style names:
    if not tx_user.empty:
        tx_user_raw = tx_user.rename(columns={
            "Time": "ts",
            "Action": "action",
            "Outcome": "token",
            "Quantity": "qty",
            "BuyAmt_Delta": "buy_delta",
            "SellAmt_Delta": "sell_delta",
        })
    else:
        tx_user_raw = tx_user

    basis = compute_cost_basis_from_df(tx_user_raw)

    # 2) Pull reserves from API (not sqlite)
    reserves_list = m.get('reserves')
    reserves_now = {r["token"]: {"shares": int(r["shares"]), "usdc": float(r["usdc"])} for r in reserves_list}
    total_pool_now = float(sum(r["usdc"] for r in reserves_now.values()))

    # 3) Build the PnL table (prices dict already computed earlier from reserves_map)
    pnl_rows = []
    for t in TOKENS:
        sh = int(basis[t]['shares'])
        avg = float(basis[t]['avg_cost'])
        pos_cost = float(basis[t]['cost'])
        price_now_token = float(prices[t])
        mv = sh * price_now_token
        unreal = mv - pos_cost
        pnl_pct = (unreal / pos_cost * 100.0) if pos_cost > 0 else 0.0

        win_shares_now = max(1, int(reserves_now.get(t, {}).get('shares', 0)))
        est_payout_if_win = 0.0
        if total_pool_now > 0 and win_shares_now > 0 and sh > 0:
            est_payout_if_win = total_pool_now * (sh / win_shares_now)

        pnl_rows.append({
            "Token": t,
            "Shares": sh,
            "Avg Cost/Share": round(avg, DEFAULT_DECIMAL_PRECISION),
            "Position Cost": round(pos_cost, DEFAULT_DECIMAL_PRECISION),
            "Price Now": round(price_now_token, DEFAULT_DECIMAL_PRECISION),
            "Market Value": round(mv, DEFAULT_DECIMAL_PRECISION),
            "Unrealized PnL": round(unreal, DEFAULT_DECIMAL_PRECISION),
            "PnL %": round(pnl_pct, DEFAULT_DECIMAL_PRECISION),
            "Est. Payout if Wins": round(est_payout_if_win, DEFAULT_DECIMAL_PRECISION),
            "Est. Win PnL": round(est_payout_if_win - pos_cost, DEFAULT_DECIMAL_PRECISION)
        })

    pnl_df = pd.DataFrame(pnl_rows)
    st.markdown("**Per-Token Position & PnL**")
    st.dataframe(pnl_df, use_container_width=True)
    st.divider()


# Common Trading UI
input_mode = st.radio("Select Input Mode", ["Quantity", "USDC"], horizontal=True)

quantity = 0
usdc_input = 0.0
if input_mode == "Quantity":
    quantity = st.number_input("Enter Quantity", min_value=1, step=1)
else:
    usdc_str = st.text_input("Enter USDC Amount", key="usdc_input_raw", placeholder="0.00")
    try:
        usdc_input = float(usdc_str) if usdc_str.strip() else 0.0
        if usdc_input < 0:
            st.warning("USDC must be ≥ 0.")
            usdc_input = 0.0
    except ValueError:
        st.warning("Enter a valid number, e.g. 123.45")
        usdc_input = 0.0

st.subheader("Buy/Sell Controls")
cols = st.columns(4)

# Fetch current reserves (DB)
# Reserves from market API
reserves_list = m.get("reserves")
reserves_map = {r["token"]: {"shares": int(r["shares"]), "usdc": float(r["usdc"])} for r in reserves_list}

# Holdings for logged-in user
user_holdings_list = user_holdings_list
user_holdings = {h["token"]: int(h["shares"]) for h in user_holdings_list}

# Prices (unchanged — uses your curves)
prices = {t: float(buy_curve(reserves_map.get(t, {}).get("shares", 0))) for t in TOKENS}

for i, token in enumerate(TOKENS):

    
    with cols[i]:
        st.markdown(f"### Outcome :blue[{token}]")

        reserve = int(reserves_map[token]["shares"])  # global shares for token
        price_now = round(buy_curve(reserve), 2)
        mcap_now = round(reserves_map[token]["usdc"], 2)
        st.text(f"Current Price: ${price_now}")
        st.text(f"MCAP: {format_usdc_compact(mcap_now)}")

         # --- Preview section (live estimate) ---
        # derive est quantities from current inputs (without executing)
        if input_mode == "Quantity":
            est_q_buy = quantity
            est_q_sell = quantity
        else:
            est_q_buy = qty_from_buy_usdc(reserve, usdc_input)
            est_q_sell = qty_from_sell_usdc(reserve, usdc_input)

        # compute deltas for estimates (clamp to non-negative)
        est_q_buy = max(0, int(est_q_buy))
        est_q_sell = max(0, int(est_q_sell))

        # buy estimate
        if est_q_buy > 0:
            _, _, est_buy_cost, _, _ = metrics_from_qty(reserve, est_q_buy)
        else:
            est_buy_cost = 0.0

        # sell estimate + tax %
        if est_q_sell > 0:
            _, _, _, est_sell_proceeds, est_tax_rate = metrics_from_qty(reserve, est_q_sell)
        else:
            est_sell_proceeds = 0.0
            est_tax_rate = 0.0

        st.caption(
            f"Est. Buy Cost ({est_q_buy}x sh): **{est_buy_cost:,.2f} USDC**  \n"
            f"Est. Sell Proceeds ({est_q_sell}x sh): **{est_sell_proceeds:,.2f} USDC**  \n"
            f"Order Tax on Sell: **{est_tax_rate*100:.2f}%**"
        )

        buy_col, sell_col = st.columns(2)


        # --- BUY ---
        with buy_col:
            disabled = ('user_id' not in st.session_state) or (not active)
            if st.button(f"Buy {token}", disabled=disabled):
                if 'user_id' not in st.session_state:
                    st.stop()
                # derive quantity if USDC mode
                q = quantity
                if input_mode == "USDC":
                    q = qty_from_buy_usdc(reserve, usdc_input)
                if q <= 0:
                    st.warning("Quantity computed as 0.")
                else:
                    bp, _, bdelta, _, _ = metrics_from_qty(reserve, q)
                    # check balance
                    payload = {
                        "user_id": st.session_state.user_id,
                        "token": token,
                        "side": "buy",   # or "sell"
                        "mode": "qty" if input_mode == "Quantity" else "usdc",
                        "amount": int(quantity) if input_mode == "Quantity" else float(usdc_input),
                    }
                    try:
                        api_post("/trade", json=payload)
                        st.rerun()
                    except APIError as e:
                        st.error(str(e))
   

        # --- SELL ---
        with sell_col:
            disabled = ('user_id' not in st.session_state) or (not active)
            if st.button(f"Sell {token}", disabled=disabled):
                if 'user_id' not in st.session_state:
                    st.stop()
                q = quantity
                if input_mode == "USDC":
                    q = qty_from_sell_usdc(reserve, usdc_input)
                if q <= 0:
                    st.warning("Quantity computed as 0.")
                else:
                    payload = {
                        "user_id": st.session_state.user_id,
                        "token": token,
                        "side": "sell",   # or "sell"
                        "mode": "qty" if input_mode == "Quantity" else "usdc",
                        "amount": int(quantity) if input_mode == "Quantity" else float(usdc_input),
                    }
                    try:
                        api_post("/trade", json=payload)
                        st.rerun()
                    except APIError as e:
                        st.error(str(e))
  

# ================================
# Dashboards (global)
# ================================
# Pull reserves from the Market API
reserves_list = m.get("reserves")
res_df = pd.DataFrame(reserves_list)

# Normalize to PrettyCaps schema used by the app
if res_df.empty:
    res_df = pd.DataFrame(columns=["Token", "Shares", "USDC"])
else:
    res_df = res_df.rename(columns={"token": "Token", "shares": "Shares", "usdc": "USDC"})

# Ensure numeric types
if "Shares" in res_df.columns:
    res_df["Shares"] = pd.to_numeric(res_df["Shares"], errors="coerce").fillna(0).astype(int)
if "USDC" in res_df.columns:
    res_df["USDC"] = pd.to_numeric(res_df["USDC"], errors="coerce").fillna(0.0).astype(float)

# Guarantee all TOKENS exist in the table (even if 0)
existing_tokens = set(res_df["Token"]) if "Token" in res_df.columns else set()
missing_tokens = [t for t in TOKENS if t not in existing_tokens]
if missing_tokens:
    res_df = pd.concat([
        res_df,
        pd.DataFrame({"Token": missing_tokens, "Shares": 0, "USDC": 0.0})
    ], ignore_index=True)

# Order rows by TOKENS for consistent UI
if not res_df.empty:
    res_df = res_df.set_index("Token").reindex(TOKENS).fillna({"Shares": 0, "USDC": 0.0}).reset_index()

# Totals
res_tot_shares = int(res_df["Shares"].sum()) if "Shares" in res_df else 0
res_tot_usdc   = float(res_df["USDC"].sum())  if "USDC"  in res_df else 0.0

# Users count (works whether you expose /users/list or not)
try:
    users_df_all = pd.DataFrame(api_get("/users/list"))  # preferred
    users_count = len(users_df_all)
except Exception:
    try:
        stats = api_get("/stats")  # optional fallback if you added a stats endpoint
        users_count = int(stats.get("users", 0))
    except Exception:
        users_count = 0

st.subheader("Overall Market Metrics")
mc_cols = st.columns(3)
with mc_cols[0]:
    st.metric("Total USDC Reserve", round(res_tot_usdc, 2))
with mc_cols[1]:
    st.text("Market Status")
    st_display_market_status(active)
with mc_cols[2]:
    st.metric("Users", users_count)

# Per-token metrics row
tok_cols = st.columns(len(TOKENS))
for i, token in enumerate(TOKENS):
    with tok_cols[i]:
        st.subheader(f'Outcome :blue[{token}]')
        row = res_df.loc[res_df['Token'] == token].iloc[0] if not res_df.empty else {"Shares": 0, "USDC": 0.0}
        reserve = int(row["Shares"])
        price = round(buy_curve(reserve), 3)
        mcap = round(float(row["USDC"]), 2)
        sub_cols = st.columns(3)
        with sub_cols[0]:
            st.metric(f"Total Shares", reserve)
        with sub_cols[1]:
            st.metric(f"Price", f"${price}")
        with sub_cols[2]:
            st.metric(f"MCAP", format_usdc_compact(mcap))

# Odds based on circulating shares
total_market_shares = max(1, int(res_df["Shares"].sum())) if "Shares" in res_df else 1
sub_cols_2 = st.columns(len(TOKENS))
for i, token in enumerate(TOKENS):
    # safe lookup
    s_series = res_df.loc[res_df["Token"] == token, "Shares"]
    s = int(s_series.iloc[0]) if not s_series.empty else 0
    odds_val = '-' if s == 0 else round(1 / (s / total_market_shares), 2)
    with sub_cols_2[i]:
        st.metric(f"Odds {token}", f"{odds_val}x" if odds_val != '-' else "-", border=True)

# ================================
# Logs & Charts (Transactions)
# ================================
# Fetch and normalize /tx to match PrettyCaps schema used by charts
# expect keys like ts, user, user_id, action, token, qty, ...
rename_map = {
    "ts": "Time",
    "user": "User",
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
tx = tx_raw.rename(columns=rename_map)

# Ensure required columns exist (handles empty or partial payloads)
required_cols = ["Time", "User", "user_id", "Action", "Outcome", "Quantity", "BuyAmt_Delta", "SellAmt_Delta"]
for c in required_cols:
    if c not in tx.columns:
        tx[c] = pd.Series(dtype="float" if c in ["Quantity","BuyAmt_Delta","SellAmt_Delta"] else "object")

# Coerce types
if not tx.empty:
    tx["Time"] = pd.to_datetime(tx["Time"], format="ISO8601", utc=True, errors="coerce")
    tx["Quantity"] = pd.to_numeric(tx["Quantity"], errors="coerce").fillna(0).astype(int)
    for c in ["BuyAmt_Delta", "SellAmt_Delta", "Buy Price", "Sell Price", "Balance After"]:
        if c in tx.columns:
            tx[c] = pd.to_numeric(tx[c], errors="coerce")

# Drop rows without timestamp for charting
tx = tx.dropna(subset=["Time"]) if "Time" in tx.columns else tx

# Pretty display copy
tx_display = tx.copy()
float_cols = tx_display.select_dtypes(include=["float64", "float32", "float"]).columns
for col in float_cols:
    tx_display[col] = tx_display[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

if tx.empty:
    st.info("No transactions yet to compute history.")
else:
    st.subheader("Transaction Log")
    st.dataframe(tx_display, use_container_width=True)
    st.divider()

    # 📈 Payout/Share Trend (reconstructed)
    st.subheader("📈 Payout/Share Trend")
    shares_timeline = []
    running = {t: 0 for t in TOKENS}

    # only process rows that have required fields
    for _, r in tx.iterrows():
        tkn = r.get("Outcome")
        act = r.get("Action")
        qty = int(r.get("Quantity") or 0)
        tstamp = r.get("Time")

        if tkn not in running or pd.isna(tstamp):
            continue

        if act == "Buy":
            running[tkn] += qty
        elif act == "Sell":
            running[tkn] -= qty
        # ignore Resolve for shares movement here

        total = sum(running.values())
        denom = running[tkn] if running[tkn] > 0 else 1  # avoid div/0
        payout = (total if total > 0 else 1) / denom

        shares_timeline.append({
            "Time": pd.to_datetime(tstamp),
            "Outcome": tkn,
            "Payout/Share": payout
        })

    ps_df = pd.DataFrame(shares_timeline)
    if ps_df.empty:
        st.info("No share movement yet to plot payout/share.")
    else:
        fig = px.line(
            ps_df, x="Time", y="Payout/Share", color="Outcome", markers=True,
            title="Payout/Share Over Time"
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_payout_share")
        st.divider()

# ===========================================================
st.header("🔁 Bonding Curves by Outcome")


# --- smart sampler: dense around reserve, sparse elsewhere ---
def _curve_samples(max_shares: int, reserve: int, dense_pts: int = 1500, sparse_pts: int = 600) -> np.ndarray:
    if max_shares <= 1:
        return np.array([1], dtype=int)

    # Dense window = ±3% of domain (capped at ±50k)
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

# Vectorized curve evals
def buy_curve_np(x: np.ndarray) -> np.ndarray:
    # y = cbrt(x)/1000 + 0.1
    return np.cbrt(x) / 1000.0 + 0.1

def sell_marginal_net_np(x: np.ndarray) -> np.ndarray:
    # Effective *net* marginal price for selling 1 share at reserve x
    # = [gross (x→x-1)] * (1 - tax(q=1, C=x))
    x = x.astype(int)
    # gross price for 1 share = buy_delta(x) - buy_delta(x-1)
    bd = np.vectorize(buy_delta, otypes=[float])(x)
    bd_prev = np.vectorize(buy_delta, otypes=[float])(np.maximum(0, x - 1))
    gross_1 = bd - bd_prev
    tax_1 = _sale_tax_rate_vec(1, x)  # tax for selling 1 share from reserve x
    net_1 = gross_1 * (1.0 - tax_1)
    net_1[x <= 0] = 0.0
    return net_1

@st.cache_data(show_spinner=False)
def get_curve_series(max_shares: int, reserve: int, dense_pts: int = 1500, sparse_pts: int = 600):
    xs = _curve_samples(max_shares, reserve, dense_pts=dense_pts, sparse_pts=sparse_pts)
    return xs, buy_curve_np(xs), sell_marginal_net_np(xs)

# ---------------------------
# API fetchers
# ---------------------------
@st.cache_data(show_spinner=False, ttl=3)
def fetch_market_with_reserves(market_id: int) -> dict:
    # assuming api_get joins base+path internally or accepts absolute paths
    return api_get(f"/market/{market_id}", timeout=10)

# ---------------------------
# Render
# ---------------------------
market_id = 1
with st.spinner("Loading market & reserves..."):
    try:
        market = fetch_market_with_reserves(market_id)
    except requests.HTTPError as e:
        st.error(f"Failed to load market {market_id}: {e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load market {market_id}: {e}")
        st.stop()

reserves: List[dict] = market.get("reserves", [])
if not reserves:
    st.info("No reserves found for this market.")
    st.stop()

# derive token list dynamically
TOKENS = [r["token"] for r in reserves]

# quick lookup for shares/usdc
latest_reserves = {r["token"]: int(r.get("shares", 0)) for r in reserves}
latest_usdc     = {r["token"]: float(r.get("usdc", 0.0)) for r in reserves}

token_tabs = st.tabs(TOKENS)
for token, token_tab in zip(TOKENS, token_tabs):
    with token_tab:
        reserve = int(latest_reserves.get(token, 0))

        # One radio to switch sub-graphs (no nested tabs)
        view = st.radio(
            "View",
            ["Buy Curve", "Sell Spread", "Effective Sell (Net)"],
            horizontal=True,
            key=f"view_{token}",
        )

        if view == "Buy Curve":
            # point annotations at current reserve
            buy_price_now = float(buy_curve(reserve))
            sell_net_now = float(current_marginal_sell_price_after_tax(reserve))  # if you have this

            # smart-sampled series (buy + sell curves)
            xs, buy_vals, sell_net_vals = get_curve_series(MAX_SHARES, reserve)

            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scattergl(
                x=xs, y=buy_vals, mode='lines', name='Buy Curve'
            ))
            # If you want to show the net sell (1-share) curve too, uncomment:
            # fig_curve.add_trace(go.Scattergl(
            #     x=xs, y=sell_net_vals, mode='lines', name='Sell (net, 1 share)'
            # ))

            # Buy point annotation
            fig_curve.add_trace(go.Scatter(
                x=[reserve], y=[buy_price_now], mode='markers+text',
                name=f'{token} Buy Point',
                text=[f"Shares: {reserve}<br>Buy: {buy_price_now:.4f}"],
                textposition="top right",
                marker=dict(size=10),
                showlegend=False
            ))

            # helper lines
            y0 = max(
                0.0,
                min(
                    float(np.nanmin(buy_vals)),
                    float(np.nanmin(sell_net_vals)),
                    buy_price_now,
                    sell_net_now,
                )
            )
            fig_curve.add_trace(go.Scatter(
                x=[reserve, reserve], y=[y0, buy_price_now],
                mode='lines', line=dict(dash='dot'), showlegend=False
            ))
            fig_curve.add_trace(go.Scatter(
                x=[xs.min(), reserve], y=[buy_price_now, buy_price_now],
                mode='lines', line=dict(dash='dot'), showlegend=False
            ))
            # If you also plot sell_net_vals, you can add matching helper lines:
            # fig_curve.add_trace(go.Scatter(
            #     x=[reserve, reserve], y=[y0, sell_net_now],
            #     mode='lines', line=dict(dash='dot'), showlegend=False
            # ))
            # fig_curve.add_trace(go.Scatter(
            #     x=[xs.min(), reserve], y=[sell_net_now, sell_net_now],
            #     mode='lines', line=dict(dash='dot'), showlegend=False
            # ))

            fig_curve.update_layout(
                title=f'{token} — Buy vs Sell (net, 1 share)',
                xaxis_title='Shares (reserve)',
                yaxis_title='Price',
                hovermode="x unified",
            )
            st.plotly_chart(fig_curve, use_container_width=True, key=f"chart_curve_{token}")

        elif view == "Sell Spread":
            if reserve <= 0:
                st.info("No circulating shares yet — Sell spread curve will show once there is supply.")
            else:
                steps = 200
                X = np.linspace(0.0, 1.0, steps + 1)
                q_grid = (X * reserve).astype(int)

                # vectorized tax rate for selling q out of current reserve
                tax_y = _sale_tax_rate_vec(q_grid, reserve)

                fig_tax = go.Figure()
                fig_tax.add_trace(go.Scattergl(
                    x=X * 100.0, y=tax_y, mode='lines', name='Spread Rate'
                ))
                fig_tax.update_layout(
                    title='Sale Tax vs % of Supply Sold (per order)',
                    xaxis_title='% of Current Supply Sold in Order',
                    yaxis_title='Tax Rate',
                    hovermode="x unified"
                )
                st.plotly_chart(fig_tax, use_container_width=True, key=f"sale_tax_curve_{token}")

        else:  # "Effective Sell (Net)"
            C = reserve
            st.markdown(f"**{token}**")
            if C <= 0:
                st.info("No circulating shares.")
            else:
                # up to 10% of supply (at least 1)
                q_max = max(1, C // 10)
                q_axis = np.linspace(1, q_max, 200).astype(int)

                # gross proceeds via integral difference; then apply order-level tax
                bd_C      = np.vectorize(buy_delta,  otypes=[float])(np.full_like(q_axis, C))
                bd_C_minQ = np.vectorize(buy_delta,  otypes=[float])(C - q_axis)
                gross     = bd_C - bd_C_minQ

                tax = _sale_tax_rate_vec(q_axis, np.full_like(q_axis, C))
                net = np.maximum(0.0, gross * (1.0 - tax))
                avg_net = net / np.maximum(1, q_axis)

                fig_eff = go.Figure()
                fig_eff.add_trace(go.Scattergl(
                    x=q_axis, y=avg_net, mode='lines', name=f'{token} Avg Net Sell Price'
                ))
                fig_eff.update_layout(
                    title=f'Effective Avg Net Sell Price vs Quantity — {token}',
                    xaxis_title='Quantity sold in a single order',
                    yaxis_title='Avg Net Sell Price (USDC/share)',
                    hovermode="x unified"
                )
                st.plotly_chart(fig_eff, use_container_width=True, key=f"effective_sell_{token}")


 # ===========================================================

# Always fetch tx + users once
txp = tx_raw
if txp.empty:
    pass

# Normalize txp for your plotting helpers:
else:
    users_df = pd.DataFrame(api_get("/users"))[["id","username"]]
    txp = txp.rename(columns={
        "ts":"Time","user":"User","action":"Action","token":"Outcome",
        "qty":"Quantity","buy_delta":"BuyAmt_Delta","sell_delta":"SellAmt_Delta","user_id":"user_id"
    })

    # Normalize column names
    txp = txp.rename(columns={
        "ts": "Time",
        "user_id": "user_id",
        "username": "User",
        "action": "Action",
        "token": "Outcome",
        "qty": "Quantity",
        "buy_delta": "BuyAmt_Delta",
        "sell_delta": "SellAmt_Delta"
    })

    txp["Time"] = pd.to_datetime(txp["Time"], format="ISO8601", utc=True, errors="coerce")

    # Historical charts: toggle between Portfolio Value and Points
    st.subheader("📊 History")

    # Tabs for Portfolio vs Points
    tab1, tab2 = st.tabs(["💼 Portfolio Value", "⭐ Points"])
    # Fetch tx for all users
    # txp = pd.DataFrame(api_get("/tx"))

    with tab1:
        # ---- Portfolio chart (Buy Price) ----
        # Reconstruct state over time (same as your current logic)
        reserves_state = {t: 0 for t in TOKENS}
        user_state = {
            int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS}}
            for _, r in users_df.iterrows()
        }
        records = []

        for _, r in txp.iterrows():
            uid = int(r["user_id"])
            act = r["Action"]
            tkn = r["Outcome"]
            qty = int(r["Quantity"])

            if act == "Buy":
                delta = float(r["BuyAmt_Delta"] or 0.0)
                user_state[uid]["balance"] -= delta
                user_state[uid]["holdings"][tkn] += qty
                reserves_state[tkn] += qty

            elif act == "Sell":
                delta = float(r["SellAmt_Delta"] or 0.0)
                user_state[uid]["balance"] += delta
                user_state[uid]["holdings"][tkn] -= qty
                reserves_state[tkn] -= qty

            elif act == "Resolve":
                payout = float(r["SellAmt_Delta"] or 0.0)
                user_state[uid]["balance"] += payout
                for u_id in user_state:
                    user_state[u_id]["holdings"] = {t: 0 for t in TOKENS}
                reserves_state = {t: 0 for t in TOKENS}

            # Pricing snapshot
            if act == "Resolve":
                prices = {t: 0.0 for t in TOKENS}
            else:
                prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}


            # Snapshot every user at this event time
            for u_id, s in user_state.items():
                pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
                pnl = pv - STARTING_BALANCE
                records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})

        port_df = pd.DataFrame(records)
        fig_port = px.line(
            port_df, x="Time", y="PortfolioValue", color="User",
            title=f"Portfolio Value Over Time (Buy Price)"
        )
        st.plotly_chart(fig_port, use_container_width=True, key="portfolio_value_chart")

    with tab2:
        # ---- Points chart ----
        points_tl = compute_points_timeline(txp, users_df)

        if points_tl.empty:
            st.info("No points history to display yet.")
        else:
            # (Optional) keep leaderboard consistency by excluding admin
            points_plot_df = points_tl[points_tl["User"].str.lower() != "admin"].copy()

            fig_pts = px.line(
                points_plot_df,
                x="Time",
                y="TotalPoints",
                color="User",
                title="Total Points Over Time (Cumulative Volume + Instant PnL Points)",
                line_group="User",
            )
            st.plotly_chart(fig_pts, use_container_width=True, key='points_chart')

 # ===========================================================
# Historical portfolio visualization (toggle between buy and sell price)
if not txp.empty:
    reserves_state = {t: 0 for t in TOKENS}
    user_state = {
        int(r.id): {"username": r.username, "balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKENS}}
        for _, r in users_df.iterrows()
    }
    records = []

    txp["Time"] = pd.to_datetime(txp["Time"])

    for _, r in txp.iterrows():
        uid = int(r["user_id"])
        act = r["Action"]
        tkn = r["Outcome"]
        qty = int(r["Quantity"])

        if act == "Buy":
            delta = float(r["BuyAmt_Delta"] or 0.0)
            user_state[uid]["balance"] -= delta
            user_state[uid]["holdings"][tkn] += qty
            reserves_state[tkn] += qty

        elif act == "Sell":
            delta = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += delta
            user_state[uid]["holdings"][tkn] -= qty
            reserves_state[tkn] -= qty

        elif act == "Resolve":
            # Apply this user's payout (logged in SellAmt_Delta)
            payout = float(r["SellAmt_Delta"] or 0.0)
            user_state[uid]["balance"] += payout

            # Zero ALL holdings (winners keep only credited payout; losers go to 0)
            for u_id in user_state:
                user_state[u_id]["holdings"] = {t: 0 for t in TOKENS}
            reserves_state = {t: 0 for t in TOKENS}

        # Pricing for valuation
        if act == "Resolve":
            # After resolution, holdings are worthless; prices treated as 0
            prices = {t: 0.0 for t in TOKENS}
        else:
            prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}

        # Snapshot all users at this event time
        for u_id, s in user_state.items():
            pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
            pnl = pv - STARTING_BALANCE
            records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})
    st.divider()

    # === Leaderboard (PV now, PnL, Payout & Points) ===
    users_df_all = load_users_df()
    if users_df_all.empty:
        st.info("No users found (or /users returned an unexpected shape).")
    else:
        # 1) Latest reserves and mark-to-curve prices
        reserves_latest = m.get('reserves')
        res_map_shares = {r["token"]: int(r["shares"]) for r in reserves_latest}
        price_now = {t: float(buy_curve(res_map_shares.get(t, 0))) for t in TOKENS}

        # 2) CURRENT holdings rebuilt from tx (resets on Resolve)
        holdings_now = rebuild_holdings_now(tx, users_df_all)

        # 3) Build “latest” leaderboard rows (PV=balance + holdings*price_now)
        lb_rows = []
        for _, u in users_df_all.iterrows():
            uid = int(u["id"])
            bal_now = float(u["balance"])  # live cash from API
            shares_val = sum(holdings_now.get(uid, {}).get(t, 0) * price_now[t] for t in TOKENS)
            pv_now = bal_now + shares_val
            lb_rows.append({"User": u["username"], "PortfolioValue": pv_now})
        latest = pd.DataFrame(lb_rows)

        # Exclude admin if you like
        if not latest.empty and "User" in latest.columns:
            latest = latest[latest["User"].str.lower() != "admin"]

        latest["PnL"] = latest["PortfolioValue"] - STARTING_BALANCE

        # ---- NEW: compute payout received at resolution from tx log (API version) ----
        # tx uses normalized columns: 'Action', 'SellAmt_Delta', 'user_id'
        payouts_df = (
            tx.loc[tx["Action"] == "Resolve", ["user_id", "SellAmt_Delta"]]
            .groupby("user_id", as_index=False)
            .agg(Payout=("SellAmt_Delta", "sum"))
        )

        # Map user_id -> username from /users
        if not payouts_df.empty:
            payouts_df = payouts_df.merge(
                users_df_all[["id", "username"]].rename(columns={"id": "user_id", "username": "User"}),
                on="user_id",
                how="left"
            )[["User", "Payout"]]
        else:
            payouts_df = pd.DataFrame(columns=["User", "Payout"])

        payouts_df["Payout"] = pd.to_numeric(payouts_df["Payout"], errors="coerce").fillna(0.0)

        latest = latest.merge(payouts_df, on="User", how="left")
        latest["Payout"] = latest["Payout"].fillna(0.0)

        # === Compute points (volume + PnL points) ===
        # Volume points from full tx + users (your helper already expects tx with normalized columns)
        points_vol = compute_user_points(tx, users_df_all[["id", "username"]])

        pnl_points = latest[["User", "PnL"]].copy()
        pnl_points["PnLPoints"] = pnl_points["PnL"].clip(lower=0.0) * PNL_POINTS_PER_USD
        pts = points_vol.merge(pnl_points[["User", "PnLPoints"]], on="User", how="left")
        pts["PnLPoints"] = pts["PnLPoints"].fillna(0.0)
        pts["TotalPoints"] = pts["VolumePoints"] + pts["PnLPoints"]

        latest = latest.merge(
            pts[["User", "VolumePoints", "PnLPoints", "TotalPoints"]],
            on="User",
            how="left",
        ).fillna({"VolumePoints": 0.0, "PnLPoints": 0.0, "TotalPoints": 0.0})

        # ---- UI: let you sort by Payout to verify payouts after resolution ----
        metric_choice = st.radio(
            "Leaderboard metric (sort by):",
            ["Portfolio Value", "PnL", "Payout"],
            horizontal=True,
            key="lb_metric"
        )
        sort_key = {"Portfolio Value": "PortfolioValue", "PnL": "PnL", "Payout": "Payout"}[metric_choice]
        latest = latest.sort_values(sort_key, ascending=False, kind="mergesort")

        # Optional fairness check after resolution (among those who got > 0)
        payouts_nonzero = payouts_df[payouts_df["Payout"] > 0]
        if resolved_flag == 1 and not payouts_nonzero.empty and payouts_nonzero["Payout"].nunique() == 1:
            st.caption("✅ All positive payouts are equal among winners.")

        # Top cards + table (include Payout column)
        st.subheader("🏆 Leaderboard")
        if latest.empty:
            st.info("No eligible users to display yet.")
        else:
            top_cols = st.columns(min(3, len(latest)))
            for i, (_, row) in enumerate(latest.head(3).iterrows()):
                delta_val = f"${row['PnL']:,.2f}" if row['PnL'] >= 0 else f"-${abs(row['PnL']):,.2f}"
                with top_cols[i]:
                    st.metric(
                        label=f"#{i+1} {row['User']}",
                        value=f"${row['PortfolioValue']:,.2f}",
                        delta=delta_val,
                        border=True
                    )
                    st.caption(f"Payout: ${row['Payout']:,.2f}")
                    st.caption(f"Points: {row['TotalPoints']:,.0f}")

            st.dataframe(
                latest[["User", "Payout", "PortfolioValue", "PnL", "VolumePoints", "PnLPoints", "TotalPoints"]],
                use_container_width=True
            )

