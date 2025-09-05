from fastapi import FastAPI, Depends, HTTPException, Body, status, Path, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Annotated
from datetime import datetime, timedelta
import secrets, os
from textwrap import dedent
from dotenv import load_dotenv, find_dotenv

from .db import Base, engine, get_db
from .models import User, Market, Reserve, Holding, Tx, MarketOutcome
from .schemas import *
from .logic import do_trade, compute_user_points, is_market_active, ownership_breakdown, preview_for_side_mode, list_market_outcomes, STARTING_BALANCE, MARKET_DURATION_DAYS, TOKENS

from fastapi.middleware.cors import CORSMiddleware

DEFAULT_QUESTION = "Price of Ethereum greater than $4500 by 7th Sept?"
DEFAULT_RES_NOTE = ("This market will resolve according to the price chart of the "
                    "Binance Spot Market ETH/USDT until the end of deadline (7th Sept 12:00 UTC).")
DEFAULT_OUTCOMES = ["YES", "NO"]  # or your 3 buckets

# Load .env file for root path
load_dotenv(find_dotenv(), override=False)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not ADMIN_PASSWORD:
    # raise or log clearly so you catch it early
    print("WARNING: ADMIN_PASSWORD is not set; admin endpoints will fail")

def _assert_admin(pw: str):
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=500, detail="Admin password not configured")
    if not pw:
        raise HTTPException(status_code=401, detail="Missing admin password")
    if not secrets.compare_digest(pw, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid admin password")

def _now_iso():
    # no timezone suffix, matches _parse_ts above
    return datetime.utcnow().replace(microsecond=0).isoformat()




OPENAPI_TAGS = [
    {"name": "System",        "description": "Health/liveness and meta endpoints."},
    {"name": "Users",         "description": "Manage user accounts and balances."},
    {"name": "Market",        "description": "Start, reset, resolve, and inspect markets."},
    {"name": "Trading",       "description": "Execute trades and preview bonding-curve metrics."},
    {"name": "Holdings",      "description": "Query user token holdings."},
    {"name": "Transactions",  "description": "Retrieve transaction logs (buys/sells/resolves)."},
    {"name": "Leaderboard",   "description": "Points and rankings across users."},
    {"name": "Admin",         "description": "Privileged market control (start/reset/resolve)."},
]

API_VERSION = "1.1.0"

app = FastAPI(
    title="42 DPM API",
    version=API_VERSION,
    summary="Dynamic Pari-mutuel platform API",
    description=dedent("""
        ## Overview
        This API powers the **42: Dynamic Pari-mutuel** platform.

        Use it to manage **users**, **markets**, **trading**, and **leaderboards**.

        ### Groups
        - **Users** → create/load users, balances
        - **Market** → start/reset/resolve markets, inspect reserves
        - **Trading** → execute trades, preview metrics_from_qty
        - **Holdings** → per-user holdings (by ID or username)
        - **Transactions** → full audit log
        - **Leaderboard** → points & rankings
        - **Admin** → privileged operations

        ### Notes
        - All timestamps are UTC (ISO-8601 without timezone suffix).
        - Bonding-curve maths mirror the Streamlit frontend (cube-root buy, logistic sell tax).
    """).strip(),
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "docExpansion": "list",
        "tryItOutEnabled": True,
    },
)

def _market_to_out(db: Session, m: Market) -> MarketOut:
    tokens = [
        o.token
        for o in db.query(MarketOutcome)
                   .filter(MarketOutcome.market_id == m.id)
                   .order_by(MarketOutcome.sort_order.asc(), MarketOutcome.token.asc())
                   .all()
    ]
    
    return MarketOut(
        id=m.id,
        start_ts=m.start_ts,
        end_ts=m.end_ts,
        winner_token=m.winner_token,
        resolved=int(m.resolved or 0),
        resolved_ts=m.resolved_ts,
        question=m.question or "",
        resolution_note=m.resolution_note or "",
        outcomes=tokens,
    )



def ensure_bootstrap(db: Session):
    market = db.query(Market).get(1)
    if not market:
        market = Market(id=1, start_ts=..., end_ts=...)
        db.add(market)
    for t in TOKENS:
        if not db.query(Reserve).filter_by(market_id=1, token=t).first():
            db.add(Reserve(market_id=1, token=t, shares=0, usdc=0.0))
    db.commit()


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    with next(get_db()) as db:
        m = db.get(Market, 1)
        if not m:
            now_dt = datetime.utcnow().replace(microsecond=0)
            end_dt = now_dt + timedelta(days=5)
            db.add(Market(
                id=1,
                start_ts=now_dt.isoformat(),
                end_ts=end_dt.isoformat(),
                winner_token=None,
                resolved=0,
                resolved_ts=None,
                question=DEFAULT_QUESTION,
                resolution_note=DEFAULT_RES_NOTE
            ))
             # outcomes
            for i, tok in enumerate(DEFAULT_OUTCOMES):
                db.add(MarketOutcome(market_id=1, token=tok, display=tok, sort_order=i))
            for t in TOKENS:
                db.add(Reserve(market_id=1, token=t, shares=0, usdc=0.0))
            db.commit()

# ====== Default ======
@app.get(
    "/health",
    response_model=HealthOut,
    tags=["System"],
    summary="Health check",
    description="Lightweight liveness probe. Returns `'ok'` when the API is up."
)
def health():
    return HealthOut(status="ok")

# ====== User ======
# Fetch all users
@app.get(
    "/users",
    response_model=List[UserOut],
    tags=["Users"],
    summary="List all users",
    description="Fetch all users currently registered in the system."
)
def list_users(db: Session = Depends(get_db)):
    # Either return ORM objects (Pydantic v2 from_attributes)...
    users = db.query(User).order_by(User.id.asc()).all()
    return [UserOut(id=u.id, username=u.username, balance=u.balance) for u in users]
    

# Fetch user by user_id
@app.get(
    "/users/{user_id}",
    response_model=UserOut,
    tags=["Users"],
    summary="Get user by ID",
    description="Returns the user by ID."
)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)  # or: db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Not Found")
    return UserOut(id=user.id, username=user.username, balance=user.balance)

# Fetch user data by username (unchanged)
@app.get(
    "/users/by-username/{username}",
    response_model=UserOut,
    tags=["Users"],
    summary="Get user by username"
)
def get_user_by_username(
    username: str = Path(..., min_length=1),
    db: Session = Depends(get_db),
):
    u = (
        db.query(User)
        .filter(func.lower(User.username) == func.lower(username.strip()))
        .first()
    )
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut(id=u.id, username=u.username, balance=u.balance)

# @app.get("/users/with-points", response_model=List[UserWithPointsOut])
# def list_users_with_points(db: Session = Depends(get_db)):
#     users = db.query(User).order_by(User.id.asc()).all()
#     out = []
#     pts_all = { (r.get("user_id") or r.get("id")): r for r in compute_user_points(db) }
#     for u in users:
#         pr = pts_all.get(u.id, {})
#         out.append(UserWithPointsOut(
#             id=u.id, username=u.username, balance=u.balance,
#             volume_points=float(pr.get("volume_points") or pr.get("VolumePoints") or 0.0),
#             pnl_points=float(pr.get("pnl_points") or pr.get("PnLPoints") or 0.0),
#             total_points=float(pr.get("total_points") or pr.get("TotalPoints") or 0.0),
#         ))
#     print(out)
#     return out

# Update user data via username
@app.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED,
          tags=["Users"], summary="Create or load user")
def join_or_load_user(payload: UserCreate, db: Session = Depends(get_db)):
    uname = payload.username.strip()
    if not uname:
        raise HTTPException(status_code=400, detail="username required")

    # case-insensitive lookup
    user = db.query(User).filter(func.lower(User.username) == func.lower(uname)).first()
    if user:
        # return existing (200 would also be fine; we still return 201 here)
        return UserOut(id=user.id, username=user.username, balance=user.balance)

    # create new
    user = User(username=uname, balance=STARTING_BALANCE, created_at=func.now())
    db.add(user)
    db.flush()  # gets user.id

    # create zero holdings rows for each token
    # for t in TOKENS:
    #     db.add(Holding(user_id=user.id, token=t, shares=0))

    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, username=user.username, balance=user.balance)


# ====== Admin ======
# Start new market
@app.post("/admin/{market_id}/start", response_model=MarketOut,
          tags=["Admin"], summary="Start market",
          description="Starts a market. Requires admin password in body.")
def admin_start(
    market_id: int,
    req: AdminStartRequest = Body(...),
    db: Session = Depends(get_db),
):
    _assert_admin(req.password)

    now = _now_iso()
    end = (datetime.utcnow() + timedelta(days=req.duration_days or MARKET_DURATION_DAYS))\
            .replace(microsecond=0).isoformat()

    m = db.get(Market, market_id)
    if not m:
        m = Market(id=market_id, start_ts=now, end_ts=end, winner_token=None, resolved=0, resolved_ts=None)
        db.add(m)
    else:
        if is_market_active(m):
            raise HTTPException(status_code=400, detail="Market already active.")
        m.start_ts = now
        m.end_ts = end
        m.winner_token = None
        m.resolved = 0
        m.resolved_ts = None

    # Ensure reserves for THIS market
    for t in TOKENS:
        r = db.query(Reserve).filter(Reserve.market_id == market_id, Reserve.token == t).first()
        if not r:
            db.add(Reserve(market_id=market_id, token=t, shares=0, usdc=0.0))
        elif req.reset_reserves:
            r.shares = 0
            r.usdc = 0.0

    db.commit()
    db.refresh(m)
    return _market_to_out(db, m)


# Admin market reset (makes market active)
@app.post(
    "/admin/{market_id}/reset",
    tags=["Admin"],
    summary="Reset one market/users",
    description="Resets a market (and optionally users). Requires admin password in body."
)
def admin_reset(
    market_id: int,
    req: AdminResetRequest = Body(...),
    db: Session = Depends(get_db),
):
    _assert_admin(req.password)

    now_iso = _now_iso()
    end_iso = (datetime.utcnow() + timedelta(days=MARKET_DURATION_DAYS)).replace(microsecond=0).isoformat()

    # Upsert + mark ACTIVE
    m = db.get(Market, market_id)
    if not m:
        m = Market(
            id=market_id,
            start_ts=now_iso,
            end_ts=end_iso,
            winner_token=None,
            resolved=0,          # <-- active
            resolved_ts=None
        )
        db.add(m)
    else:
        m.start_ts = now_iso
        m.end_ts = end_iso
        m.winner_token = None
        m.resolved = 0          # <-- active
        m.resolved_ts = None

    # Reset ONLY this market's reserves (avoid UNIQUE constraint)
    db.query(Reserve).filter(Reserve.market_id == market_id).delete(synchronize_session=False)
    # Fresh zeroed rows
    for t in TOKENS:
        db.add(Reserve(market_id=market_id, token=t, shares=0, usdc=0.0))

    if req.wipe_users:
        db.query(Holding).delete(synchronize_session=False)
        db.query(Tx).delete(synchronize_session=False)
        db.query(User).delete(synchronize_session=False)
    else:
        # keep users; reset balances/holdings; clear tx
        for u in db.query(User).all():
            u.balance = STARTING_BALANCE
        db.query(Holding).update({Holding.shares: 0})
        db.query(Tx).delete(synchronize_session=False)

    db.commit()
    return {"status": "ok", "market_active": True, "start_ts": m.start_ts, "end_ts": m.end_ts}

# Resolve market
@app.post("/admin/{market_id}/resolve",
          tags=["Admin"], summary="Resolve a market",
          description="Resolves a market. Requires admin password in body.")
def admin_resolve(
    market_id: int,
    payload: AdminResolveRequest = Body(...),
    db: Session = Depends(get_db),
):
    print(payload)
    _assert_admin(payload.password)

    winner_token = (payload.winner_token or "").strip().upper()
    if winner_token not in TOKENS:
        raise HTTPException(status_code=400, detail="Invalid winner_token")

    m = db.get(Market, market_id)
    if not m:
        raise HTTPException(status_code=404, detail="Market not found")
    if int(m.resolved or 0) == 1:
        raise HTTPException(status_code=400, detail="Market already resolved")

    # Sum pool for THIS market
    tot_pool = float(
        db.query(func.coalesce(func.sum(Reserve.usdc), 0.0))
          .filter(Reserve.market_id == market_id)
          .scalar() or 0.0
    )

    # Winning shares for THIS market
    win_res = db.query(Reserve).filter(
        Reserve.market_id == market_id, Reserve.token == winner_token
    ).first()
    win_shares = int(win_res.shares if win_res else 0)

    payouts = {}
    if tot_pool > 0 and win_shares > 0:
        # NOTE: If you ever run multiple markets concurrently,
        # you should add market_id to Holding and filter by it here.
        for h in db.query(Holding).filter(Holding.token == winner_token, Holding.market_id == market_id, Holding.shares > 0).all():
            share = h.shares
            payout = tot_pool * (share / win_shares)
            payouts[h.user_id] = payouts.get(h.user_id, 0.0) + float(payout)

    now_iso = datetime.utcnow().isoformat()
    for uid, amt in payouts.items():
        u = db.get(User, uid)
        if not u:
            continue
        u.balance = float(u.balance or 0.0) + float(amt)
        db.add(Tx(
            ts=now_iso, user_id=uid, market_id=market_id, user_name=u.username, action="Resolve", token=winner_token,
            qty=0, buy_price=None, sell_price=None, buy_delta=None, sell_delta=float(amt),
            balance_after=u.balance
        ))

    # Zero out ONLY this market's reserves
    for r in db.query(Reserve).filter(Reserve.market_id == market_id).all():
        r.shares = 0
        r.usdc = 0.0

    # (Optional) Zero holdings if you want a clean slate:
    # for h in db.query(Holding).all(): h.shares = 0

    m.winner_token = winner_token
    m.resolved = 1
    m.resolved_ts = now_iso
    m.end_ts = now_iso

    db.commit()
    return {"status": "ok", "payout_pool": tot_pool, "winner_token": winner_token}

# main.py
@app.post("/admin/{market_id}/configure", tags=["Admin"], summary="Configure market copy & outcomes")
def admin_configure(
    market_id: int,
    req: AdminConfigureMarketRequest,
    db: Session = Depends(get_db),
):
    _assert_admin(req.password)

    m = db.get(Market, market_id)
    if not m:
        raise HTTPException(404, "Market not found")

    if req.question is not None:
        m.question = req.question
    if req.resolution_note is not None:
        m.resolution_note = req.resolution_note

    if req.tokens is not None:
        tokens = [t.strip().upper() for t in req.tokens if t.strip()]
        if not tokens:
            raise HTTPException(400, "tokens must be non-empty")

        # upsert outcomes (preserve order)
        existing = {o.token for o in db.query(MarketOutcome).filter_by(market_id=market_id).all()}
        # add/update
        for i, tok in enumerate(tokens):
            mo = db.query(MarketOutcome).filter_by(market_id=market_id, token=tok).first()
            if mo:
                mo.sort_order = i
                mo.display = mo.display or tok
            else:
                db.add(MarketOutcome(market_id=market_id, token=tok, display=tok, sort_order=i))
        # remove extras
        for extra in existing - set(tokens):
            db.query(MarketOutcome).filter_by(market_id=market_id, token=extra).delete()

        # ensure reserves match outcomes (create missing, delete extras)
        have = {r.token for r in db.query(Reserve).filter_by(market_id=market_id).all()}
        for tok in set(tokens) - have:
            db.add(Reserve(market_id=market_id, token=tok, shares=0, usdc=0.0))
        for extra in have - set(tokens):
            db.query(Reserve).filter_by(market_id=market_id, token=extra).delete()

    db.commit()
    return {"status": "ok"}


# ====== Market ======
@app.get("/markets",
         response_model=List[MarketOut],
         tags=["Market"], summary="List all markets")
def list_markets(db: Session = Depends(get_db)):
    """Return all markets (without reserves for brevity)."""
    ms = db.query(Market).order_by(Market.id.asc()).all()
    return [_market_to_out(db, m) for m in ms]


@app.get("/market/{market_id}",
         response_model=MarketWithReservesOut,
         tags=["Market"], summary="Get one market (with reserves)")
def get_market(market_id: int, db: Session = Depends(get_db)):
    m = db.query(Market).filter(Market.id == market_id).first()
    if not m:
        raise HTTPException(404, "Market not found")

    reserves = db.query(Reserve).filter(Reserve.market_id == market_id).all()
    outcomes = (db.query(MarketOutcome)
                  .filter(MarketOutcome.market_id == market_id)
                  .order_by(MarketOutcome.sort_order.asc(), MarketOutcome.token.asc())
                  .all())
                  
    base = _market_to_out(db, m)  # reuse the basic fields
    return MarketWithReservesOut(
        id=base.id,
        start_ts=m.start_ts,
        end_ts=m.end_ts,
        winner_token=m.winner_token,
        resolved=m.resolved,
        resolved_ts=m.resolved_ts,
        question=m.question,
        resolution_note=m.resolution_note,
        outcomes=[
            OutcomeOut(token=o.token, display=o.display, sort_order=int(o.sort_order or 0))
            for o in outcomes
        ],
        reserves=reserves,
    )

# Token user ownership breakdown
@app.get(
    "/market/{market_id}/ownership/{token}",
    response_model=OwnershipBreakdownOut,
    tags=["Market"],
    summary="Ownership breakdown for a market outcome",
    description="Returns holders (user_id, username, shares, share_pct) for the given market and token.",
)
def get_ownership_breakdown(
    market_id: int,
    token: str,
    min_shares: int = Query(1, ge=0, description="Minimum shares to be included (default 1, set 0 to include all)"),
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Optional top-N holders"),
    order: Literal["desc", "asc"] = Query("desc", description="Sort by shares"),
    db: Session = Depends(get_db),
):
    token = token.strip().upper()

    # Validate token here to return a proper HTTP 400
    valid = list_market_outcomes(db, market_id)
    if token not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid token for market {market_id}. Valid: {valid}")

    data = ownership_breakdown(
        db, market_id, token,
        min_shares=min_shares,
        limit=limit,
        order=order,
    )

    # The logic() returns a plain dict; it already matches the response model shape.
    return data


# ====== User Holdings ======
# Fetch user holdings via user_id
@app.get(
    "/holdings/{user_id}",
    response_model=List[HoldingOut],
    tags=["Holdings"],
    summary="Get holdings by user ID",
    description="Returns token holdings for the user. Optional ?market_id filter; if omitted, returns across all markets."
)
def get_holdings(
    user_id: Annotated[int, Path(..., ge=1, description="Numeric user ID", example=1)],
    market_id: Annotated[int | None, Query(ge=1, description="Optional market filter")] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Holding).filter(Holding.user_id == user_id)
    if market_id is not None:
        q = q.filter(Holding.market_id == market_id)
    return q.order_by(Holding.token.asc()).all()

# Fetch user holdings via username
@app.get(
    "/holdings/by-username/{username}",
    response_model=List[HoldingOut],
    tags=["Holdings"],
    summary="Get holdings by username",
    description="Case-insensitive lookup. Optional ?market_id filter; if omitted, returns across all markets."
)
def get_holdings_by_username(
    username: Annotated[str, Path(..., min_length=1, description="Username (case-insensitive)", example="alice")],
    market_id: Annotated[int | None, Query(ge=1, description="Optional market filter")] = None,
    db: Session = Depends(get_db),
):
    u = (
        db.query(User)
          .filter(func.lower(User.username) == func.lower(username.strip()))
          .first()
    )
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    q = db.query(Holding).filter(Holding.user_id == u.id)
    if market_id is not None:
        q = q.filter(Holding.market_id == market_id)
    return q.order_by(Holding.token.asc()).all()

# ====== Transaction Logs ======
@app.get(
    "/tx",
    response_model=List[TxOut],
    tags=["Transactions"],
    summary="List all transaction logs",
    description="Returns every transaction in the system ordered by timestamp (ascending). "
                "Includes buys, sells, resolves, deltas, and post-trade balances."
)
def get_tx(
    market_id: Annotated[int | None, Query(ge=1, description="Filter by market id")] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Tx).order_by(Tx.ts.asc())
    if market_id is not None:
        q = q.filter(Tx.market_id == market_id)
    return q.all()


# ====== Trading ======
# Post a trade action for a user
@app.post("/trade",
          tags=["Trading"], summary="Execute a trade")
def trade(payload: TradeIn, db: Session = Depends(get_db)):
    try:
        result = do_trade(
            db,
            user_id=payload.user_id,
            token=payload.token,
            market_id=payload.market_id,
            side=payload.side.lower(),
            mode=payload.mode.lower(),  # "qty" or "usdc"
            amount=float(payload.amount),
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))

# ====== Leaderboard / Points ======
@app.get("/leaderboard", response_model=List[LeaderboardRow],
         tags=["Leaderboard"], summary="View leaderboard standings")
def leaderboard(db: Session = Depends(get_db)):
    rows = compute_user_points(db)
    # sort by pnl desc (like your app)
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows

# @app.get("/points_timeline", response_model=List[PointsTimelineRow])
# def points_timeline(db: Session = Depends(get_db)):
#     # quick, coarse timeline based on current tx timestamps
#     rows = compute_user_points(db)
#     # flatten to one “now” row per user (for a true timeline, port your
#     # compute_points_timeline logic to server and build rows at each tx)
#     now = datetime.utcnow().isoformat()
#     return [{"ts": now, "user": r["user"], "total_points": r["total_points"]} for r in rows]


@app.post("/preview/{market_id}/{token}",
          response_model=PreviewResponse,
          tags=["Trading"],
          summary="Preview buy/sell metrics (side+mode+amount) with no state change")
def preview_metrics(
    market_id: Annotated[int, Path(..., ge=1, description="Market ID", example=1)],
    token: Annotated[str, Path(..., description="Outcome token (e.g. 'YES')", example="YES")],
    req: PreviewRequest,
    db: Session = Depends(get_db),
):
    token = token.strip().upper()

    # validate against THIS market's outcomes
    if token not in list_market_outcomes(db, market_id):
        raise HTTPException(status_code=400, detail=f"Invalid token for market {market_id}")

    r = (
        db.query(Reserve)
        .filter(Reserve.market_id == market_id, Reserve.token == token)
        .first()
    )
    if not r:
        raise HTTPException(status_code=404, detail="Reserve not found for this market/token")

    reserve = int(r.shares or 0)

    res = preview_for_side_mode(
        reserve,
        side=req.side,
        mode=req.mode,
        amount=req.amount,
    )

    return PreviewResponse(
        market_id=market_id,
        token=token,
        reserve=reserve,
        quantity=res["quantity"],
        new_reserve=res["new_reserve"],
        buy_price=res["buy_price"],
        sell_price_marginal_net=res["sell_price_marginal_net"],
        buy_amt_delta=res["buy_amt_delta"],
        sell_amt_delta=res["sell_amt_delta"],
        sell_tax_rate_used=res["sell_tax_rate_used"],
    )

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your exact Streamlit origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)