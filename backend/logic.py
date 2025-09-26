import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from .models import User, Market, Reserve, Holding, Tx, MarketOutcome


# To-do:
# 1. replace search function for input size
# 

# ---- Constants (match your Streamlit app) ----
BASE_EPSILON = 1e-4
MARKET_DURATION_DAYS = 5
END_TS = "2025-08-15 00:00"
STARTING_BALANCE = 50000.0
TOKENS = ["YES", "NO"]
MAX_SHARES = 5000000 #5M

# Points
MINT_POINTS_PER_SHARE        = 10.0 
REALISED_PNL_POINTS_PER_USD  = 5.0   

# Math Helpers (curves)
def buy_curve(x: float) -> float:
    return (x**(1/3)/1000) + 0.1
    # return x**(1/4) + x / 400


def buy_delta(x: float) -> float:
    return (3.0/4000.0) * (x**(4.0/3.0)) + x/10.0
    # return (640 * x**(5/4) + x**2) / 800


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
    X = _clamp01(float(q) / float(C))  # fraction of supply this order is selling
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
# _sale_tax_rate_vec = np.vectorize(sale_tax_rate, otypes=[float])

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
            db.add(Reserve(token=t, market_id=1, shares=0, usdc=0.0))
    db.commit()


def _parse_ts(s: str):
    if not s:
        return None
    # Accept '2025-08-31T12:34:56.123456', '2025-08-31 12:34:56', etc.
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

def parse_to_naive_utc(dt_str: str) -> str:
    """
    Accepts common ISO-8601 forms (with/without 'Z' or offset),
    returns a naive UTC ISO string (no microseconds), e.g. '2026-01-01T00:00:00'.
    """
    s = dt_str.strip()
    # Allow 'Z'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # datetime.fromisoformat handles '+HH:MM' offsets; if none, assume naive (UTC)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # Treat naive as UTC (your API uses UTC-naive everywhere)
        dt_utc = dt
    else:
        dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt_utc.replace(microsecond=0).isoformat()

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

# Market Outcomes: Return list of outcome tokens
def list_market_outcomes(db: Session, market_id: int) -> list[str]:
    return [
        o.token
        for o in db.query(MarketOutcome)
                   .filter(MarketOutcome.market_id == market_id)
                   .order_by(MarketOutcome.sort_order.asc(), MarketOutcome.token.asc())
                   .all()
    ]


def ensure_user_holdings_for_market(db: Session, user_id: int, market_id: int):
    existing = {
        (h.token)
        for h in db.query(Holding).filter_by(user_id=user_id, market_id=market_id).all()
    }
    for token in list_market_outcomes(db, market_id):
        if token not in existing:
            db.add(Holding(user_id=user_id, market_id=market_id, token=token, shares=0))
    db.flush

def _allowed_tokens_for_market(db: Session, market_id: int) -> set[str]:
    rows = db.execute(
        select(func.upper(MarketOutcome.token)).where(MarketOutcome.market_id == market_id)
    ).scalars().all()
    return set(rows)

# ---- Trading endpoints logic ----
def do_trade(db: Session, user_id: int, token: str, market_id: int, side: str, mode: str, amount: float):
    # 0) Market must be active (single source of truth)
    m = db.get(Market, market_id)
    ok, why = is_market_active_with_reason(m)
    if not ok:
        raise ValueError(f"Market not active: {why}")

    # sanity check
    # ensure_user_holdings_for_market(db, user_id, market_id)

    # 1) Basic input validation
    # normalize token and validate against THIS market's outcomes
    tok = (token or "").strip().upper()
    side = side.lower().strip()
    mode = mode.lower().strip()

    allowed = _allowed_tokens_for_market(db, market_id)
    if not allowed:
        raise ValueError("Market has no configured outcome tokens.")
    if tok not in allowed:
        raise ValueError(f"Invalid token for market {market_id}. Allowed: {sorted(allowed)}")

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

   # 3) Load or create reserve for THIS market + token
    res = db.scalar(
        select(Reserve).where(
            Reserve.market_id == market_id,
            Reserve.token == tok,
        )
    )
    if res is None:
        res = Reserve(market_id=market_id, token=tok, shares=0, usdc=0.0)
        db.add(res)
        db.flush()  # gets res.id

    # 4) Load or create holding row for (user, market, token)
    hold = db.scalar(
        select(Holding).where(
            Holding.user_id == user_id,
            Holding.market_id == market_id,
            Holding.token == tok,
        )
    )
    if hold is None:
        hold = Holding(user_id=user_id, market_id=market_id, token=tok, shares=0)
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
    username = user.username

    if side == "buy":
        # price & deltas based on current reserve
        buy_price, _, buy_amt_delta, _, _ = metrics_from_qty(reserve, qty)
        if user.balance < buy_amt_delta:
            raise ValueError("Insufficient balance.")

        # state updates
        res.shares = int(res.shares) + qty
        res.usdc   = float(res.usdc) + float(buy_amt_delta)
        user.balance = float(user.balance) - float(buy_amt_delta)
        hold.shares = int(hold.shares) + qty

        # tx log
        db.add(Tx(
            ts=now_iso, user_id=user_id,market_id=market_id, user_name=username, action="Buy", token=token, qty=qty,
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
        _, sell_price, _, sell_amt_delta, _ = metrics_from_qty(reserve, qty)

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
            ts=now_iso, user_id=user_id, market_id=market_id, user_name=username, action="Sell", token=token, qty=qty,
            buy_price=None, buy_delta=None,
            sell_price=float(sell_price), sell_delta=float(sell_amt_delta),
            balance_after=float(user.balance),
        ))
        db.commit()
        return {"status": "ok", "qty": qty, "proceeds": float(sell_amt_delta), "price": float(sell_price)}

# Preview trade outcome
def preview_for_side_mode(
    reserve: int,
    *,
    side: str,            # "buy" | "sell"
    mode: str,            # "qty" | "usdc"
    amount: Optional[float],   # required by validator upstream
) -> Dict[str, float | int]:
    """
    Pure computation for preview, no DB access.

    - If mode="qty": treat `amount` as a quantity; round to nearest int and clamp to >= 0.
    - If mode="usdc": treat `amount` as USDC; derive quantity via qty_from_buy_usdc / qty_from_sell_usdc.
    """
    side = (side or "").strip().lower()
    mode = (mode or "").strip().lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if mode not in ("qty", "usdc"):
        raise ValueError("mode must be 'qty' or 'usdc'")

    if mode == "qty":
        q = int(max(0, round(float(amount or 0.0))))
    else:  # mode == "usdc"
        usd = float(amount or 0.0)
        q = qty_from_buy_usdc(reserve, usd) if side == "buy" else qty_from_sell_usdc(reserve, usd)

    # Core preview math (existing semantics)
    buy_price, sell_price, buy_amt, sell_amt, tax_used = metrics_from_qty(reserve, q)

    # Prospective reserve after the action
    new_reserve = reserve + q if side == "buy" else max(0, reserve - min(q, reserve))

    return {
        "quantity": int(q),
        "new_reserve": int(new_reserve),
        "buy_price": float(buy_price),
        "sell_price_marginal_net": float(sell_price),
        "buy_amt_delta": float(buy_amt),
        "sell_amt_delta": float(sell_amt),
        "sell_tax_rate_used": float(tax_used),
    }

# Token ownership breakdown
def ownership_breakdown(
    db: Session,
    market_id: int,
    token: str,
    *,
    min_shares: int = 1,               # exclude 0-share rows by default
    limit: Optional[int] = None,       # top-N holders
    order: str = "desc",               # "desc" (largest first) or "asc"
) -> Dict[str, Any]:
    token = (token or "").strip().upper()

    # 1) Validate token against canonical outcomes for THIS market
    valid = list_market_outcomes(db, market_id)
    if token not in valid:
        # keep it logic-layer agnostic; let the endpoint raise HTTPException
        return {
            "market_id": market_id,
            "token": token,
            "total_shares": 0,
            "holders": [],
            "error": "invalid-token",
            "valid_tokens": valid,
        }

    # 2) Total outstanding shares for this market+token (denominator for pct)
    total_shares = int(db.query(func.coalesce(func.sum(Holding.shares), 0))
                         .filter(Holding.market_id == market_id,
                                 Holding.token == token,
                                 Holding.shares > 0)
                         .scalar() or 0)

    # 3) Pull holders with optional threshold/limit and ordering
    q = (db.query(Holding.user_id, User.username, Holding.shares)
           .join(User, User.id == Holding.user_id)
           .filter(Holding.market_id == market_id,
                   Holding.token == token,
                   Holding.shares >= (min_shares if min_shares is not None else 0)))

    if order == "asc":
        q = q.order_by(Holding.shares.asc(), User.username.asc())
    else:
        q = q.order_by(Holding.shares.desc(), User.username.asc())

    if limit:
        q = q.limit(int(limit))

    rows = q.all()

    holders: List[Dict[str, Any]] = []
    for uid, uname, sh in rows:
        pct = (float(sh) / float(total_shares)) if total_shares > 0 else 0.0
        holders.append({
            "user_id": int(uid),
            "username": uname,
            "shares": int(sh),
            "share_pct": float(pct), # percentage
        })

    return {
        "market_id": market_id,
        "token": token,
        "total_shares": int(total_shares),
        "holders": holders,
    }

# # ---- Points + Leaderboard helpers (same math as app) ----
def compute_user_points(db: Session, market_id: Optional[int] = None):
    """
    New points model:
      - Mint points: 10 pts/share for any shares still held at the instant of market resolution.
        Awarded once when a Resolve occurs (snapshot all users' holdings before wiping).
        If the market is resolved but no Resolve tx rows exist, award at the end using the
        final holdings snapshot.
      - Realised PnL points: 5 pts per $1 of realised PROFIT ONLY (>=0).
          * Sell: proceeds - (avg cost of shares sold)
          * Resolve (winners): payout - (avg cost of remaining winning shares)
    Returns: list of dicts per user: user, portfolio_value, pnl,
             mint_points, realised_pnl_points, total_points.
    """
    # Load tx (optionally filtered) ordered by time
    tx_q = db.query(Tx).order_by(Tx.ts.asc())
    if market_id is not None:
        tx_q = tx_q.filter(Tx.market_id == market_id)
    tx = tx_q.all()

    users = {u.id: u.username for u in db.query(User).all()}

    # Market resolved flag (only meaningful when market_id is provided)
    m = db.get(Market, market_id) if market_id is not None else None
    market_is_resolved = bool(m and int(m.resolved or 0) == 1)

    if not tx:
        # Baseline rows if no activity
        return [
            {
                "user": uname,
                "portfolio_value": STARTING_BALANCE,
                "pnl": 0.0,
                "mint_points": 0.0,
                "realised_pnl_points": 0.0,
                "total_points": 0.0,
            }
            for uid, uname in users.items()
        ]

    # Determine token universe for this scope
    if market_id is not None:
        TOKS = sorted(_token_set_for_market(db, market_id)) or []
    else:
        TOKS = sorted({r.token for r in tx if r.token})  # global union
    if not TOKS:
        TOKS = []

    # ---- State while walking the timeline ----
    reserves_state = {t: 0 for t in TOKS}  # circulating shares
    user_state = {
        uid: {"balance": STARTING_BALANCE, "holdings": {t: 0 for t in TOKS}}
        for uid in users.keys()
    }

    # Running-average cost basis per user+token
    pos = {uid: {t: {"shares": 0, "cost": 0.0} for t in TOKS} for uid in users.keys()}

    # Point tallies
    mint_points = {uid: 0.0 for uid in users.keys()}
    realised_profit_usd = {uid: 0.0 for uid in users.keys()}

    # We award mint points exactly once
    mint_awarded = False

    # ---- Walk chronologically ----
    for r in tx:
        uid = int(r.user_id)
        tkn = r.token
        act = (r.action or "").strip()

        # Defensive: add unseen token on the fly
        if tkn and tkn not in reserves_state:
            reserves_state[tkn] = 0
            for st in user_state.values():
                st["holdings"].setdefault(tkn, 0)
            for p in pos.values():
                p.setdefault(tkn, {"shares": 0, "cost": 0.0})

        if act == "Buy":
            q = int(r.qty or 0)
            usd = float(r.buy_delta or 0.0)
            reserves_state[tkn] += q
            user_state[uid]["balance"] -= usd
            user_state[uid]["holdings"][tkn] += q
            # running-average basis
            pos[uid][tkn]["shares"] += q
            pos[uid][tkn]["cost"]   += usd

        elif act == "Sell":
            q = int(r.qty or 0)
            usd = float(r.sell_delta or 0.0)
            reserves_state[tkn] -= q
            user_state[uid]["balance"] += usd
            user_state[uid]["holdings"][tkn] -= q
            # realise vs avg cost
            s_prev = max(0, pos[uid][tkn]["shares"])
            c_prev = max(0.0, pos[uid][tkn]["cost"])
            if s_prev > 0 and q > 0:
                used = min(q, s_prev)
                avg = c_prev / float(s_prev)
                realised = usd - avg * used
                if realised > 0:
                    realised_profit_usd[uid] += realised
                pos[uid][tkn]["shares"] = s_prev - used
                pos[uid][tkn]["cost"]   = c_prev - avg * used

        elif act == "Resolve":
            # 1) Award mint points ONCE for ALL users at the instant of resolution.
            if not mint_awarded:
                for u2 in users.keys():
                    outstanding = sum(int(user_state[u2]["holdings"].get(t, 0)) for t in TOKS)
                    if outstanding > 0:
                        mint_points[u2] += outstanding * MINT_POINTS_PER_SHARE
                mint_awarded = True

            # 2) Realise winner payout vs remaining winning basis for THIS recipient
            payout = float(r.sell_delta or 0.0)
            winner = tkn  # Resolve rows use the winning token
            if payout > 0.0 and winner:
                s_win = max(0, pos[uid][winner]["shares"])
                c_win = max(0.0, pos[uid][winner]["cost"])
                if s_win > 0:
                    avg_win = c_win / float(s_win)
                    realised = payout - (avg_win * s_win)
                    if realised > 0:
                        realised_profit_usd[uid] += realised

            # 3) Wipe ALL users' holdings, reserves, and bases (market is closed)
            for u2 in users.keys():
                user_state[u2]["holdings"] = {t: 0 for t in TOKS}
                for t in TOKS:
                    pos[u2][t] = {"shares": 0, "cost": 0.0}
            for t in TOKS:
                reserves_state[t] = 0

        # else: ignore unknown actions

    # If market is resolved but we never saw a Resolve row (e.g., zero pool),
    # award mint points from the final holdings snapshot.
    if market_is_resolved and not mint_awarded:
        for u2 in users.keys():
            outstanding = sum(int(user_state[u2]["holdings"].get(t, 0)) for t in TOKS)
            if outstanding > 0:
                mint_points[u2] += outstanding * MINT_POINTS_PER_SHARE
        mint_awarded = True  # (not strictly needed after loop)

    # ---- Final PV/PnL snapshot (buy-price valuation) ----
    prices = {t: float(buy_curve(int(reserves_state.get(t, 0)))) for t in TOKS}
    rows = []
    for uid, uname in users.items():
        bal = float(user_state[uid]["balance"])
        mv  = sum(int(user_state[uid]["holdings"].get(t, 0)) * prices[t] for t in TOKS)
        pv  = bal + mv
        pnl = pv - STARTING_BALANCE

        rp_pts   = max(0.0, realised_profit_usd[uid]) * REALISED_PNL_POINTS_PER_USD
        mint_pts = float(mint_points[uid])
        total    = mint_pts + rp_pts

        rows.append({
            "user": uname,
            "portfolio_value": pv,
            "pnl": pnl,
            "mint_points": mint_pts,
            "realised_pnl_points": rp_pts,
            "total_points": total,
        })

    return rows



# ---- Helpers ----

def _token_set_for_market(db: Session, market_id: int) -> set[str]:
    """Prefer tokens from reserves for this market; fallback to tokens in tx."""
    toks = {
        r.token
        for r in db.query(Reserve).filter(Reserve.market_id == market_id).all()
        if r.token
    }
    if toks:
        return toks
    # fallback: from tx rows
    toks = {
        r.token
        for r in db.query(Tx.token).filter(Tx.market_id == market_id).all()
        if r and r.token
    }
    return toks
