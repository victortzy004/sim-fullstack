import math
from datetime import datetime, timedelta
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import select, func, update, delete
from .models import User, Market, Reserve, Holding, Tx


# To-do:
# 1. replace search function for input size
# 

# ---- Constants (match your Streamlit app) ----
BASE_EPSILON = 1e-4
MARKET_DURATION_DAYS = 5
END_TS = "2025-08-24 00:00"
STARTING_BALANCE = 10000.0
TOKENS = ["YES", "NO"]
MAX_SHARES = 5000000 #5M

# Points
TRADE_POINTS_PER_USD = 10.0
PNL_POINTS_PER_USD   = 5.0
EARLY_QUANTITY_POINT = 270
MID_QUANTITY_POINT   = 630
EARLY_MULTIPLIER     = 1.5
MID_MULTIPLIER       = 1.25
LATE_MULTIPLIER      = 1.0

# Math Helpers (curves)
def buy_curve(x: float) -> float:
    return (x**(1/3)/1000) + 0.1
    # return x**(1/4) + x / 400

def sell_curve(x: float) -> float:
    t = (x - 500000.0) / 1_000_000.0
    p = 1.0 / (4.0 * (0.8 + math.exp(-t))) - 0.05
    return max(p,0.0)
    # return ((x - 500)/40) / ((8 + ((x - 500)/80)**2)**0.5) + (x - 500)/300 + 3.6

def buy_delta(x: float) -> float:
    return (3.0/4000.0) * (x**(4.0/3.0)) + x/10.0
    # return (640 * x**(5/4) + x**2) / 800

def sell_delta(x: float) -> float:
    """
    ∫ y dx = 312500 * ln(1 + 0.8 * e^{(x-500000)/1e6}) - 0.05*x + C, C=0
    """
    t = (x - 500000.0) / 1_000_000.0
    # use log1p for better numerical stability
    return 312_500.0 * math.log1p(0.8 * math.exp(t)) - 0.05 * x
    # return 1.93333 * x + 0.00166667 * x**2 + 2 * math.sqrt(301200 - 1000 * x + x**2)

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

def ensure_seed(db: Session):
    # market
    m = db.scalar(select(Market).where(Market.id==1))
    if not m:
        start = datetime.utcnow()
        end = start + timedelta(days=MARKET_DURATION_DAYS)
        db.add(Market(id=1, start_ts=start.isoformat(), end_ts=end.isoformat(), resolved=0))
    # reserves
    for t in TOKENS:
        if not db.scalar(select(Reserve).where(Reserve.token==t)):
            db.add(Reserve(token=t, shares=0, usdc=0.0))
    db.commit()


def _parse_ts(s: str):
    if not s:
        return None
    # Accept '2025-08-31T12:34:56.123456', '2025-08-31 12:34:56', etc.
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

# Market helper
def is_market_active_with_reason(m) -> tuple[bool, str]:
    if not m:
        return (False, "no-market")
    if int(m.resolved or 0) == 1:
        return (False, "resolved")
    start = _parse_ts(m.start_ts)
    end   = _parse_ts(m.end_ts)
    if not start or not end:
        return (False, "bad-timestamps")
    now = datetime.utcnow()
    if now < start:
        return (False, f"not-started (now<{start.isoformat()})")
    if now > end:
        return (False, f"ended (now>{end.isoformat()})")
    return (True, "ok")

# Main market active fx
def is_market_active(m) -> bool:
    ok, _ = is_market_active_with_reason(m)
    return ok

# ---- Trading endpoints logic ----
def do_trade(db: Session, user_id: int, token: str, side: str, mode: str, amount: float):
    # 0) Market must be active (single source of truth)
    m = db.get(Market, 1)
    ok, why = is_market_active_with_reason(m)
    if not ok:
        raise ValueError(f"Market not active: {why}")

    # 1) Basic input validation
    if token not in TOKENS:
        raise ValueError("Invalid token.")
    side = side.lower().strip()
    mode = mode.lower().strip()
    if side not in ("buy", "sell"):
        raise ValueError("Invalid side.")
    if mode not in ("qty", "usdc"):
        raise ValueError("Invalid mode.")
    try:
        amount = float(amount)
    except Exception:
        raise ValueError("Invalid amount.")

    # 2) Load user
    user = db.get(User, user_id)
    if not user:
        raise ValueError("User not found.")

    # 3) Load or create reserve for this token
    #   NOTE: don't use db.get(Reserve, token) unless token is the PK.
    res = db.scalar(select(Reserve).where(Reserve.token == token))
    if res is None:
        res = Reserve(token=token, shares=0, usdc=0.0)
        db.add(res)
        db.flush()

    # 4) Load or create holding row for (user, token)
    hold = db.scalar(select(Holding).where(Holding.user_id == user_id, Holding.token == token))
    if hold is None:
        hold = Holding(user_id=user_id, token=token, shares=0)
        db.add(hold)
        db.flush()

    reserve = int(res.shares)

    # 5) Derive quantity from input mode
    if mode == "usdc":
        if side == "buy":
            qty = qty_from_buy_usdc(reserve, amount)
        else:
            qty = qty_from_sell_usdc(reserve, amount)
    else:
        qty = int(amount)

    if qty <= 0:
        raise ValueError("Quantity computed as 0.")

    now_iso = datetime.utcnow().replace(microsecond=0).isoformat()

    if side == "buy":
        # price & deltas based on current reserve
        buy_price, _, buy_amt_delta, _ = metrics_from_qty(reserve, qty)
        if user.balance < buy_amt_delta:
            raise ValueError("Insufficient balance.")

        # state updates
        res.shares = int(res.shares) + qty
        res.usdc   = float(res.usdc) + float(buy_amt_delta)
        user.balance = float(user.balance) - float(buy_amt_delta)
        hold.shares = int(hold.shares) + qty

        # tx log
        db.add(Tx(
            ts=now_iso, user_id=user_id, action="Buy", token=token, qty=qty,
            buy_price=float(buy_price), buy_delta=float(buy_amt_delta),
            sell_price=None, sell_delta=None,
            balance_after=float(user.balance),
        ))
        db.commit()
        return {"status": "ok", "qty": qty, "cost": float(buy_amt_delta), "price": float(buy_price)}

    else:  # sell
        if int(hold.shares) < qty:
            raise ValueError("Insufficient shares to sell.")

        # sell metrics use current reserve (before decrement)
        _, sell_price, _, sell_amt_delta = metrics_from_qty(reserve, qty)

        # Optionally guard: pool must have enough USDC (should be true by construction)
        if float(res.usdc) < float(sell_amt_delta) - 1e-9:
            raise ValueError("Insufficient pool liquidity.")

        # state updates
        res.shares = int(res.shares) - qty
        res.usdc   = float(res.usdc) - float(sell_amt_delta)
        user.balance = float(user.balance) + float(sell_amt_delta)
        hold.shares = int(hold.shares) - qty

        # tx log
        db.add(Tx(
            ts=now_iso, user_id=user_id, action="Sell", token=token, qty=qty,
            buy_price=None, buy_delta=None,
            sell_price=float(sell_price), sell_delta=float(sell_amt_delta),
            balance_after=float(user.balance),
        ))
        db.commit()
        return {"status": "ok", "qty": qty, "proceeds": float(sell_amt_delta), "price": float(sell_price)}

# ---- Points + Leaderboard helpers (same math as app) ----
def compute_user_points(db: Session):
    # collect tx + users
    tx = db.execute(select(Tx).order_by(Tx.ts.asc())).scalars().all()
    users = db.execute(select(User.id, User.username)).all()
    id2name = {u.id: u.username for u in db.query(User).all()}

    reserves_state = {t: 0 for t in TOKENS}
    vol_points = {uid: 0.0 for uid in id2name.keys()}

    for r in tx:
        if r.action == "Buy":
            buy_usd = float(r.buy_delta or 0.0)
            reserve_before = reserves_state[r.token]
            mult = EARLY_MULTIPLIER if reserve_before < EARLY_QUANTITY_POINT else (
                   MID_MULTIPLIER if reserve_before < MID_QUANTITY_POINT else LATE_MULTIPLIER)
            vol_points[r.user_id] += buy_usd * TRADE_POINTS_PER_USD * mult
            reserves_state[r.token] += int(r.qty)
        elif r.action == "Sell":
            sell_usd = float(r.sell_delta or 0.0)
            vol_points[r.user_id] += sell_usd * TRADE_POINTS_PER_USD
            reserves_state[r.token] -= int(r.qty)

    # build latest portfolio snapshot (buy-price valuation)
    reserves_state = {t: 0 for t in TOKENS}
    user_state = {uid: {"balance": STARTING_BALANCE, "holdings": {t:0 for t in TOKENS}} for uid in id2name}

    for r in tx:
        if r.action == "Buy":
            reserves_state[r.token] += int(r.qty)
            user_state[r.user_id]["balance"] -= float(r.buy_delta or 0.0)
            user_state[r.user_id]["holdings"][r.token] += int(r.qty)
        elif r.action == "Sell":
            reserves_state[r.token] -= int(r.qty)
            user_state[r.user_id]["balance"] += float(r.sell_delta or 0.0)
            user_state[r.user_id]["holdings"][r.token] -= int(r.qty)
        elif r.action == "Resolve":
            payout = float(r.sell_delta or 0.0)
            user_state[r.user_id]["balance"] += payout
            for u in user_state:
                user_state[u]["holdings"] = {t:0 for t in TOKENS}
            reserves_state = {t:0 for t in TOKENS}

    prices = {t: buy_curve(reserves_state[t]) for t in TOKENS}
    rows = []
    for uid, s in user_state.items():
        pv = s["balance"] + sum(s["holdings"][t]*prices[t] for t in TOKENS)
        pnl = pv - STARTING_BALANCE
        pnl_points = max(pnl, 0.0) * PNL_POINTS_PER_USD
        rows.append({
            "user": id2name[uid],
            "portfolio_value": pv,
            "pnl": pnl,
            "volume_points": vol_points[uid],
            "pnl_points": pnl_points,
            "total_points": vol_points[uid] + pnl_points
        })
    return rows
