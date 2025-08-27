# file: app.py
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Tuple, Dict, List

from api import api_get, api_post, APIError

import os
st.caption(f"API_BASE_URL: {os.getenv('API_BASE_URL')}")

# ===========================================================
# Constants
DEFAULT_DECIMAL_PRECISION = 2
BASE_EPSILON = 1e-4
MARKET_DURATION_DAYS = 5
END_TS = "2025-08-31 00:00"
DB_PATH = "app.db"
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
st.title("42: Twin Bonding Curve Simulatoooor — Global PVP")

st.subheader(f":blue[{MARKET_QUESTION}]")

srv = api_get("/market/1")
server_active, server_why = interpret_market(srv)

# Use server_active for button disabled
disabled = ('user_id' not in st.session_state) or (not server_active)

# # And show it:
# with st.sidebar:
#     st.write("**Server Active:**", "✅" if server_active else f"❌ ({server_why})")

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

def sell_curve(x: float) -> float:
    """price to sell an infinitesimal share at reserve x (clamped >= 0)"""
    t = (float(x) - 500_000.0) / 1_000_000.0
    p = 1.0 / (4.0 * (0.8 + math.exp(-t))) - 0.05
    return max(p, 0.0)

def buy_delta(x: float) -> float:
    """
    ∫ buy_curve dx = (3/4000) * x^(4/3) + x/10   (C=0)
    Total USDC to move reserve from 0 → x.
    """
    x = float(x)
    return (3.0 / 4000.0) * (x ** (4.0 / 3.0)) + x / 10.0

def sell_delta(x: float) -> float:
    """
    ∫ sell_curve dx = 312500 * ln(1 + 0.8 * e^{(x-500000)/1e6}) - 0.05*x   (C=0)
    Use log1p for stability.
    """
    x = float(x)
    t = (x - 500_000.0) / 1_000_000.0
    return 312_500.0 * math.log1p(0.8 * math.exp(t)) - 0.05 * x


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

def metrics_from_qty(x: int, q: int):
    new_x = x + q
    buy_price = buy_curve(new_x)
    sell_price = sell_curve(x)
    buy_amt_delta = buy_delta(new_x) - buy_delta(x)
    sell_amt_delta = sell_delta(x) - sell_delta(x - q) if x - q >= 0 else 0.0
    return buy_price, sell_price, buy_amt_delta, sell_amt_delta


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
    if usd <= 0:
        return 0
    # initial guess: linear approx using current sell price
    q = usd / max(sell_curve(reserve), 1e-9)
    q = max(0.0, min(q, float(reserve)))

    for _ in range(12):
        f  = (sell_delta(reserve) - sell_delta(reserve - q)) - usd
        fp = max(sell_curve(reserve - q), 1e-9)  # df/dq = price at (reserve - q)
        step = f / fp
        q -= step
        if q < 0.0: q = 0.0
        if q > reserve: q = float(reserve)
        if abs(step) < 1e-6:
            break
    return int(q)

def load_users_df() -> pd.DataFrame:
    try:
        data = api_get("/users")
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

m = api_get(f"/market/{MARKET_ID}")                      # ← Fetch Market
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
# if "user_id" in st.session_state:
#     with closing(get_conn()) as conn:
#         bal_row = conn.execute("SELECT balance FROM users WHERE id=?", (st.session_state.user_id,)).fetchone()
#         if bal_row:
#             st.session_state.balance = float(bal_row["balance"])




# ===========================================================


# Admin-only reset market button (use session username)
# --- Admin controls (toggle Start vs Reset based on current state) ---
if 'user_id' in st.session_state and st.session_state.get("username") == "admin":
    st.sidebar.subheader("Admin Controls")

    if server_active:
        # Market running → show only RESET + (optional) Resolve
        wipe_users = st.sidebar.checkbox("Reset all users (wipe accounts)", value=False, key="wipe_users_reset")
        if st.sidebar.button("Reset Market", key="btn_reset_market"):
            try:
                api_post(f"/admin/{MARKET_ID}/reset", json={"wipe_users": bool(wipe_users)})
                st.success("Market reset. It is now inactive.")
                if wipe_users:
                    for k in ("user_id", "username", "balance"):
                        st.session_state.pop(k, None)
                st.rerun()
            except APIError as e:
                st.error(str(e))

        st.sidebar.markdown("### Resolve Market")
        winner = st.sidebar.selectbox("Winning outcome", TOKENS, key="winner_select")
        if st.sidebar.button("Resolve Now", disabled=not winner, type="secondary", key="btn_resolve"):
            try:
                resp = api_post(f"/admin/{MARKET_ID}/resolve", json={"winner_token": winner})
                st.success(
                    f"Market resolved. Winner: {winner}. "
                    f"Payout pool: ${float(resp.get('payout_pool', 0.0)):,.2f}"
                )
                st.rerun()
            except APIError as e:
                st.error(str(e))

    else:
        # Market inactive/reset → show only START
        st.sidebar.markdown("### Start Market")
        duration_days = st.sidebar.number_input(
            "Duration (days)", min_value=1, value=5, step=1, key="start_duration_days"
        )
        reset_reserves = st.sidebar.checkbox(
            "Zero reserves on start", value=False, key="start_zero_reserves"
        )
        if st.sidebar.button("Start Market", key="btn_start_market"):
            try:
                resp = api_post(
                    f"/admin/{MARKET_ID}/start",
                    json={"duration_days": int(duration_days), "reset_reserves": bool(reset_reserves)},
                )
                st.success(f"Market started. Ends at {resp['end_ts']}")
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
    u = api_get(f"/users/{st.session_state.user_id}")
    bal = float(u["balance"])

    user_holdings_list = api_get(f"/holdings/{st.session_state.user_id}")
    user_holdings = {h["token"]: int(h["shares"]) for h in user_holdings_list}

    reserves_list = api_get(f"/market/{MARKET_ID}").get('reserves')
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
                    txp_pts = pd.DataFrame(api_get("/tx"))
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
                        users_list = api_get("/users")  # if implemented
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

# === Cost basis & per-token PnL table ===
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
    reserves_list = api_get(f"/market/{MARKET_ID}").get('reserves')
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
    usdc_input = st.number_input("Enter USDC Amount", min_value=0.0, step=0.1)

st.subheader("Buy/Sell Controls")
cols = st.columns(4)

# Fetch current reserves (DB)
# Reserves
reserves_list = api_get("/reserves")
reserves_map = {r["token"]: {"shares": int(r["shares"]), "usdc": float(r["usdc"])} for r in reserves_list}

# Holdings for logged-in user
user_holdings_list = api_get(f"/holdings/{st.session_state.user_id}") if "user_id" in st.session_state else []
user_holdings = {h["token"]: int(h["shares"]) for h in user_holdings_list}

# Prices (unchanged — uses your curves)
prices = {t: float(buy_curve(reserves_map.get(t, {}).get("shares", 0))) for t in TOKENS}

for i, token in enumerate(TOKENS):

    
    with cols[i]:
        st.markdown(f"### Outcome :blue[{token}]")

        reserve = int(reserves_map[token]["shares"])  # global shares for token
        price_now = round(buy_curve(reserve), 2)
        mcap_now = round(reserves_map[token]["usdc"], 2)
        st.text(f"Current Price: {price_now}")
        st.text(f"MCAP: {mcap_now}")

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
            _, _, est_buy_cost, _ = metrics_from_qty(reserve, est_q_buy)
        else:
            est_buy_cost = 0.0

        # sell estimate
        if est_q_sell > 0:
            _, _, _, est_sell_proceeds = metrics_from_qty(reserve, est_q_sell)
        else:
            est_sell_proceeds = 0.0

        st.caption(
            f"Est. Buy Cost ({est_q_buy}x sh): **{est_buy_cost:,.2f} USDC**  \n"
            f"Est. Sell Proceeds ({est_q_sell}x sh): **{est_sell_proceeds:,.2f} USDC**"
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
                    bp, _, bdelta, _ = metrics_from_qty(reserve, q)
                    # check balanc
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
# Pull reserves from the API
reserves_list = api_get("/reserves")  # [{"token": "...", "shares": int, "usdc": float}, ...]
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
            st.metric(f"Total Shares {token}", reserve)
        with sub_cols[1]:
            st.metric(f"Price {token}", price)
        with sub_cols[2]:
            st.metric(f"MCAP {token}", mcap)

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
tx_raw = pd.DataFrame(api_get("/tx"))  # expect keys like ts, user, user_id, action, token, qty, ...
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

# ---------------------------
# UI Controls
# ---------------------------
# colA, colB = st.columns(2)
# with colA:
#     market_id = st.number_input("Market ID", min_value=1, value=1, step=1, help="Market to visualize")
# with colB:
#     MAX_SHARES = st.number_input("Max Shares (domain cap)", min_value=1_000, value=2_000_000, step=50_000)




# ---------------------------
# Smart sampler: dense around reserve, sparse elsewhere
# ---------------------------
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
    return xs, viz_buy_curve_np(xs), viz_sell_curve_np(xs)

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
TOKENS = sorted(TOKENS)

# quick lookup for shares/usdc
latest_reserves = {r["token"]: int(r.get("shares", 0)) for r in reserves}
latest_usdc     = {r["token"]: float(r.get("usdc", 0.0)) for r in reserves}

tabs = st.tabs(TOKENS)

for token, tab in zip(TOKENS, tabs):
    with tab:
        reserve = int(latest_reserves.get(token, 0))
        buy_price_now = buy_curve(reserve)
        sell_price_now = sell_curve(reserve)

        xs, buy_vals, sell_vals = get_curve_series(int(MAX_SHARES), reserve)

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=xs, y=buy_vals, mode='lines', name='Buy Curve'
        ))
        fig.add_trace(go.Scattergl(
            x=xs, y=sell_vals, mode='lines', name='Sell Curve'
        ))

        # current points
        fig.add_trace(go.Scatter(
            x=[reserve], y=[buy_price_now], mode='markers+text',
            name=f'{token} Buy @ reserve',
            text=[f"Shares: {reserve}<br>Price: {buy_price_now:.2f}"],
            textposition="top right",
            marker=dict(size=10),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[reserve], y=[sell_price_now], mode='markers+text',
            name=f'{token} Sell @ reserve',
            text=[f"Shares: {reserve}<br>Price: {sell_price_now:.2f}"],
            textposition="bottom right",
            marker=dict(size=10),
            showlegend=False
        ))

        # helper lines
        y_floor = float(min(buy_vals.min(), sell_vals.min(), buy_price_now, sell_price_now, 0.0))
        fig.add_trace(go.Scatter(x=[reserve, reserve], y=[y_floor, buy_price_now],
                                 mode='lines', line=dict(dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter(x=[xs.min(), reserve], y=[buy_price_now, buy_price_now],
                                 mode='lines', line=dict(dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter(x=[reserve, reserve], y=[y_floor, sell_price_now],
                                 mode='lines', line=dict(dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter(x=[xs.min(), reserve], y=[sell_price_now, sell_price_now],
                                 mode='lines', line=dict(dash='dot'), showlegend=False))

        fig.update_layout(
            title=f"{token} — Price vs Shares",
            xaxis_title="Shares",
            yaxis_title="Price",
            hovermode="x unified",
        )

        # header metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Reserve (shares)", f"{reserve:,}")
        c2.metric("Pool USDC", f"{latest_usdc.get(token, 0.0):,.2f}")
        c3.metric("Buy / Sell now", f"{buy_price_now:.2f} / {sell_price_now:.2f}")

        st.plotly_chart(fig, use_container_width=True, key=f"curve_{market_id}_{token}")

st.divider()
 # ===========================================================

# Always fetch tx + users once
txp = pd.DataFrame(api_get("/tx"))
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
        # ---- Portfolio chart (with pricing mode) ----
        price_mode = st.radio("Value holdings at:", ["Buy Price", "Mid Price", "Sell Price"], horizontal=True)

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
                if price_mode == "Buy Price":
                    prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}
                elif price_mode == "Sell Price":
                    prices = {t: sell_curve(reserves_state[t]) for t in TOKENS}
                else: # use average for mid-price
                    prices = {t: 0.5 * (buy_curve(reserves_state[t]) + sell_curve(reserves_state[t])) for t in TOKENS}

            # Snapshot every user at this event time
            for u_id, s in user_state.items():
                pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
                pnl = pv - STARTING_BALANCE
                records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})

        port_df = pd.DataFrame(records)
        fig_port = px.line(
            port_df, x="Time", y="PortfolioValue", color="User",
            title=f"Portfolio Value Over Time ({price_mode})"
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
            if price_mode == "Buy Price":
                prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}
            elif price_mode == "Sell Price":
                prices = {t: sell_curve(reserves_state[t]) for t in TOKENS}
            else:
                prices = {t: buy_curve(reserves_state[t]) - sell_curve(reserves_state[t]) for t in TOKENS}

        # Snapshot all users at this event time
        for u_id, s in user_state.items():
            pv = s["balance"] + sum(s["holdings"][t] * prices[t] for t in TOKENS)
            pnl = pv - STARTING_BALANCE
            records.append({"Time": r["Time"], "User": s["username"], "PortfolioValue": pv, "PnL": pnl})
    st.divider()

    # === Leaderboard (Correct "now" snapshot) ===
# Use current balances from /users, current holdings reconstructed from /tx,
# and current prices from the latest reserves. This reflects:
# PV_now = balance_now + sum(holdings_now[token] * price_now[token]).

# 1) Load users (id, username, balance)
users_df_all = load_users_df()
if users_df_all.empty:
    st.info("No users found (or /users returned an unexpected shape).")

else:
    # 2) Latest reserves and prices (use your APP pricing, not viz)
    reserves_latest = api_get(f"/market/{MARKET_ID}").get('reserves')
    res_map_shares = {r["token"]: int(r["shares"]) for r in reserves_latest}
    price_now = {t: float(buy_curve(res_map_shares.get(t, 0))) for t in TOKENS}

    # 3) Rebuild CURRENT holdings by replaying tx and resetting at every Resolv

    # tx was already normalized earlier (has Time, Action, Outcome, Quantity, user_id)
    holdings_now = rebuild_holdings_now(tx, users_df_all)

    # 4) Build leaderboard rows
    lb_rows = []
    for _, u in users_df_all.iterrows():
        uid = int(u["id"])
        bal_now = float(u["balance"])  # current cash from API
        shares_val = sum(holdings_now.get(uid, {}).get(t, 0) * price_now[t] for t in TOKENS)
        pv_now = bal_now + shares_val
        pnl_now = pv_now - STARTING_BALANCE
        lb_rows.append({"User": u["username"], "PortfolioValue": pv_now, "PnL": pnl_now})

    latest = pd.DataFrame(lb_rows)

    # Exclude admin if desired
    if not latest.empty:
        if "User" in latest.columns:
            latest = latest[~latest["User"].astype(str).str.lower().eq("admin")]
        elif "username" in latest.columns:
            # if the frame uses 'username', align naming and filter
            latest = latest.rename(columns={"username": "User"})
            latest = latest[~latest["User"].astype(str).str.lower().eq("admin")]

    # 5) Points: volume + PnL points (only positive PnL)
    points_vol = compute_user_points(tx, users_df_all[["id","username"]].rename(columns={"username": "username"}))

    # When latest is empty, still keep the expected columns so merges work
    if latest.empty and "User" not in latest.columns:
        latest["User"] = pd.Series(dtype="object")
    if latest.empty and "PnL" not in latest.columns:
        latest["PnL"] = pd.Series(dtype="float")
        
    pnl_points = latest[["User", "PnL"]].copy()
    pnl_points["PnLPoints"] = pnl_points["PnL"].clip(lower=0.0) * PNL_POINTS_PER_USD

    # 6) Merge and present
    pts = points_vol.merge(pnl_points[["User","PnLPoints"]], on="User", how="outer").fillna(0.0)
    latest = latest.merge(pts[["User", "PnLPoints", "VolumePoints"]], on="User", how="left").fillna(0.0)
    latest["TotalPoints"] = latest["VolumePoints"] + latest["PnLPoints"]

    # Sort by PnL (or TotalPoints, your call)
    latest = latest.sort_values("PnL", ascending=False, kind="mergesort")

    st.subheader("🏆 Leaderboard (Portfolio, PnL & Points)")
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
                st.caption(f"Points: {row['TotalPoints']:,.0f}")

        st.dataframe(
            latest[["User", "PortfolioValue", "PnL", "VolumePoints", "PnLPoints", "TotalPoints"]],
            use_container_width=True
        )
