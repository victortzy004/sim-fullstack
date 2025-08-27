from fastapi import FastAPI, Depends, HTTPException, Body, APIRouter, status, Path
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime, timedelta

from .db import Base, engine, get_db
from .models import User, Market, Reserve, Holding, Tx
from .schemas import *
from .logic import ensure_seed, do_trade, compute_user_points, is_market_active, is_market_active_with_reason, STARTING_BALANCE, MARKET_DURATION_DAYS, TOKENS

from fastapi.middleware.cors import CORSMiddleware
    

def _now_iso():
    # no timezone suffix, matches _parse_ts above
    return datetime.utcnow().replace(microsecond=0).isoformat()



app = FastAPI(
    title="42 DPM API",
    description="""
## Overview
This API powers the 42:Dynamic Pari-mutuel platform.  
Use it to manage **users**, **markets**, **trading**, and **leaderboards**.

### Subgroups
- **Users** → manage accounts and balances
- **Market** → start/reset/resolve market rounds
- **Trading** → post trades, get holdings
- **Leaderboard** → fetch points and rankings
- **Admin** → privileged endpoints
    """,
    version="1.0.0",
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
                resolved_ts=None
            ))
            for t in TOKENS:
                db.add(Reserve(market_id=1, token=t, shares=0, usdc=0.0))
            db.commit()


@app.get("/health")
def health():
    return {"status": "ok"}

# ====== User ======
# Fetch all users
@app.get("/users", response_model=List[UserOut],
         tags=["Users"], summary="List all users",
         description="Fetch all users currently registered in the system.")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.id.asc()).all()

# Fetch user by user_id
@app.get("/users/{user_id}", response_model=UserOut,
         tags=["Users"], summary="Get user by ID")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Not Found")
    return UserOut(id=user.id, username=user.username, balance=user.balance)

# Fetch user data by username
@app.get("/users/by-username/{username}", response_model=UserOut,
           tags=["Users"], summary="Get user by username")
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
    for t in TOKENS:
        db.add(Holding(user_id=user.id, token=t, shares=0))

    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, username=user.username, balance=user.balance)


# ====== Admin ======
@app.post("/admin/{market_id}/start", response_model=MarketOut,
          tags=["Admin"], summary="Start market")
def admin_start(market_id: int,
                req: StartRequest = Body(default=StartRequest()),
                db: Session = Depends(get_db)):
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
    return m

# Admin reset (path matches your Streamlit code: /admin/reset)
@app.post("/admin/{market_id}/reset",
          tags=["Admin"], summary="Reset one market/users")
def admin_reset(market_id: int, req: ResetRequest, db: Session = Depends(get_db)):
    now = _now_iso()

    m = db.get(Market, market_id)
    if not m:
        m = Market(id=market_id, start_ts=now, end_ts=now, winner_token=None, resolved=1, resolved_ts=now)
        db.add(m)
    else:
        m.start_ts = now
        m.end_ts = now
        m.winner_token = None
        m.resolved = 0
        m.resolved_ts = None

    # Reset ONLY this market's reserves
    db.query(Reserve).filter(Reserve.market_id == market_id).delete()

    for t in TOKENS:
        db.add(Reserve(market_id=market_id, token=t, shares=0, usdc=0.0))

    if req.wipe_users:
        db.query(Holding).delete()
        db.query(Tx).delete()
        db.query(User).delete()
    else:
        for u in db.query(User).all():
            u.balance = STARTING_BALANCE
        for h in db.query(Holding).all():
            h.shares = 0
        db.query(Tx).delete()

    db.commit()
    return {"status": "ok", "market_active": False}

@app.post("/admin/{market_id}/resolve",
          tags=["Admin"], summary="Resolve a market")
def admin_resolve(market_id: int, payload: dict, db: Session = Depends(get_db)):
    winner_token = (payload.get("winner_token") or "").strip()
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
        for h in db.query(Holding).filter(Holding.token == winner_token, Holding.shares > 0).all():
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
            ts=now_iso, user_id=uid, action="Resolve", token=winner_token,
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

# # Admin resolve (path matches your Streamlit code: /admin/resolve)
# @app.post("/admin/resolve",
#           tags=["Admin"], summary="Resolve on-going market.")
# def admin_resolve(payload: dict, db: Session = Depends(get_db)):
#     winner_token = (payload.get("winner_token") or "").strip()
#     if winner_token not in TOKENS:
#         raise HTTPException(status_code=400, detail="Invalid winner_token")

#     m = db.get(Market, 1)
#     if not m:
#         raise HTTPException(status_code=404, detail="Market not found")
#     if int(m.resolved or 0) == 1:
#         raise HTTPException(status_code=400, detail="Market already resolved")

#     # total pool (sum of USDC across both tokens)
#     tot_pool = float(db.query(func.coalesce(func.sum(Reserve.usdc), 0.0)).scalar() or 0.0)

#     # total winning shares
#     win_res = db.query(Reserve).filter(Reserve.token == winner_token).first()
#     win_shares = int(win_res.shares if win_res else 0)

#     payouts = {}
#     if tot_pool > 0 and win_shares > 0:
#         # all holders for winner token
#         for h in db.query(Holding).filter(Holding.token == winner_token, Holding.shares > 0).all():
#             share = h.shares
#             payout = tot_pool * (share / win_shares)
#             payouts[h.user_id] = payouts.get(h.user_id, 0.0) + float(payout)

#     # credit users + log Tx rows
#     now_iso = datetime.utcnow().isoformat()
#     for uid, amt in payouts.items():
#         u = db.get(User, uid)
#         if not u:
#             continue
#         u.balance = float(u.balance or 0.0) + float(amt)
#         # log a resolve Tx for each paid user
#         tx = Tx(
#             ts=now_iso,
#             user_id=uid,
#             action="Resolve",
#             token=winner_token,
#             qty=0,
#             buy_price=None, sell_price=None,
#             buy_delta=None, sell_delta=float(amt),
#             balance_after=u.balance
#         )
#         db.add(tx)

#     # zero holdings & reserves
#     for h in db.query(Holding).all():
#         h.shares = 0
#     for r in db.query(Reserve).all():
#         r.shares = 0
#         r.usdc = 0.0

#     # mark market resolved
#     m.winner_token = winner_token
#     m.resolved = 1
#     m.resolved_ts = now_iso
#     m.end_ts = now_iso

#     db.commit()
#     return {"status": "ok", "payout_pool": tot_pool, "winner_token": winner_token}


# ====== Market ======
@app.get("/markets",
         response_model=List[MarketOut],
         tags=["Market"], summary="List all markets")
def list_markets(db: Session = Depends(get_db)):
    """Return all markets (without reserves for brevity)."""
    return db.query(Market).order_by(Market.id.asc()).all()


@app.get("/market/{market_id}",
         response_model=MarketWithReservesOut,
         tags=["Market"], summary="Get one market (with reserves)")
def get_market(market_id: int, db: Session = Depends(get_db)):
    m = db.query(Market).filter(Market.id == market_id).first()
    if not m:
        raise HTTPException(404, "Market not found")

    reserves = db.query(Reserve).filter(Reserve.market_id == market_id).all()

    return MarketWithReservesOut(
        id=m.id,
        start_ts=m.start_ts,
        end_ts=m.end_ts,
        winner_token=m.winner_token,
        resolved=m.resolved,
        resolved_ts=m.resolved_ts,
        reserves=reserves,
    )




# @app.get("/market/status")
# def market_status(db: Session = Depends(get_db)):
#     m = db.get(Market, 1)
#     if not m:
#         return {"exists": False}
#     ok, why = is_market_active_with_reason(m)
#     return {
#         "exists": True,
#         "start_ts": m.start_ts,
#         "end_ts": m.end_ts,
#         "resolved": int(m.resolved or 0),
#         "winner_token": m.winner_token,
#         "active": ok,
#         "why": why,
#         "now": _now_iso(),
#     }

@app.post("/market/{market_id}/reset")
def reset_market(
    market_id: int = Path(..., ge=1, description="Market ID to reset"),
    req: ResetRequest = Body(...),
    db: Session = Depends(get_db),
):
    now = datetime.utcnow().replace(microsecond=0)
    end = now + timedelta(days=MARKET_DURATION_DAYS)

    # upsert the market
    m = db.get(Market, market_id)
    if not m:
        m = Market(
            id=market_id,
            start_ts=now.isoformat(),
            end_ts=end.isoformat(),
            winner_token=None,
            resolved=0,
            resolved_ts=None,
        )
        db.add(m)
    else:
        m.start_ts = now.isoformat()
        m.end_ts = end.isoformat()
        m.winner_token = None
        m.resolved = 0
        m.resolved_ts = None

    # reset reserves for THIS market only
    db.query(Reserve).filter(Reserve.market_id == market_id).delete(synchronize_session=False)
    for t in TOKENS:
        db.add(Reserve(market_id=market_id, token=t, shares=0, usdc=0.0))

    if req.wipe_users:
        # global wipe (your models aren't market-scoped)
        db.query(Holding).delete(synchronize_session=False)
        db.query(Tx).delete(synchronize_session=False)
        db.query(User).delete(synchronize_session=False)
    else:
        # keep users; reset balances/holdings; clear tx (global, given current schema)
        for u in db.query(User).all():
            u.balance = STARTING_BALANCE
        for h in db.query(Holding).all():
            h.shares = 0
        db.query(Tx).delete(synchronize_session=False)

    db.commit()
    return {"status": "ok", "market_id": market_id}

# --- Reserves / Holdings / Transactions ---
@app.get("/reserves", response_model=List[ReserveOut])
def get_reserves(db: Session = Depends(get_db)):
    return db.query(Reserve).all()

# Fetch user holdings via user_id
@app.get("/holdings/{user_id}", response_model=List[HoldingOut])
def get_holdings(user_id: int, db: Session = Depends(get_db)):
    return db.query(Holding).filter(Holding.user_id==user_id).all()

# Fetch user holdings via username
@app.get("/holdings/by-username/{username}", response_model=List[HoldingOut])
def get_holdings_by_username(
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
    return db.query(Holding).filter(Holding.user_id == u.id).all()

@app.get("/tx", response_model=List[TxOut])
def get_tx(db: Session = Depends(get_db)):
    return db.query(Tx).order_by(Tx.ts.asc()).all()

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


# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your exact Streamlit origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)