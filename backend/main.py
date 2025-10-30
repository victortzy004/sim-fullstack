from fastapi import FastAPI, Depends, HTTPException, Body, status, Path, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import func, delete
from typing import List, Annotated, Tuple
from datetime import datetime, timedelta
import secrets, os
from textwrap import dedent
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict

from .db import Base, engine, get_db
from .models import User, Market, Reserve, Holding, Tx, MarketOutcome
from .schemas import *
from .logic import do_trade, buy_curve, compute_user_points, is_market_active, parse_to_naive_utc, ownership_breakdown, preview_for_side_mode, list_market_outcomes, user_out_by_filter, STARTING_BALANCE, SPECIAL_STARTING_BALANCE, MARKET_DURATION_DAYS, TOKENS

from fastapi.middleware.cors import CORSMiddleware

DEFAULT_QUESTION = "Price of Ethereum by 15th Sept?"
DEFAULT_RES_NOTE = ("This market will resolve according to the price chart of the "
                    "Binance Spot Market ETH/USDT until the end of deadline (15th Sept 12:00 UTC).")
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


# =========================
# Resolved-holdings helpers
# =========================

def _iso(ts: datetime | str) -> str:
    """Return ISO-8601 string (keeps strings as-is; converts datetimes)."""
    if isinstance(ts, str):
        return ts
    return ts.replace(microsecond=0).isoformat()
# ---------- helpers: generic share-reconstruction at a timestamp ----------
def _shares_at_ts(
    db: Session,
    *,
    user_id: int,
    market_id: int,
    token: str,
    ts_iso: str,
) -> int:
    tok_upper = (token or "").strip().upper()
    print(f"[shares_at_ts] uid={user_id} mid={market_id} tok='{token}' ts<={ts_iso}", flush=True)

    q = (
        db.query(Tx)
          .filter(
              Tx.user_id == user_id,
              Tx.market_id == market_id,
              func.upper(Tx.token) == tok_upper,
              Tx.ts <= ts_iso,
              Tx.action.in_(["Buy", "Sell"]),
          )
          .order_by(Tx.ts.asc(), Tx.id.asc())
    )
    rows = q.all()
    print(f"[shares_at_ts] found {len(rows)} buy/sell rows for {token}", flush=True)

    total = 0
    prev_post_trade_qty = 0
    for r in rows:
        print(f"  ↳ tx id={r.id} act={r.action} qty={r.qty} qty_change={r.qty_change}", flush=True)

        qc = getattr(r, "qty_change", None)

        # use qty_change only if it’s not None *and* non-zero
        if qc not in (None, 0):
            delta = int(qc)
            total += delta
            prev_post_trade_qty += delta
        else:
            # fallback logic based on action
            act = (r.action or "").strip().lower()
            qn = int(r.qty or 0)
            if act == "buy":
                delta = abs(qn)
            elif act == "sell":
                delta = -abs(qn)
            else:
                delta = 0

            total += delta
            prev_post_trade_qty += delta

    print(f"[shares_at_ts] total shares_at={total} for {token}", flush=True)
    return max(0, int(total))


def build_resolved_holdings_for_user(
    db: Session,
    *,
    user_id: int,
    market_id: Optional[int] = None,
) -> List[ResolvedHoldingOut]:
    mq = db.query(Market).filter(Market.resolved == 1)
    if market_id is not None:
        mq = mq.filter(Market.id == market_id)
    markets = mq.all()
    print(f"[resolved_holdings] user={user_id} markets_found={len(markets)}", flush=True)

    out: List[ResolvedHoldingOut] = []
    for m in markets:
        mid = int(m.id)
        resolved_iso = (m.resolved_ts or m.end_ts or m.start_ts)
        print(f"[resolved_holdings] market_id={mid} winner={m.winner_token} ts={resolved_iso}", flush=True)

        # outcomes
        tokens = [
            o.token
            for o in db.query(MarketOutcome)
                       .filter(MarketOutcome.market_id == mid)
                       .order_by(MarketOutcome.sort_order.asc(), MarketOutcome.token.asc())
                       .all()
        ]
        print(f"[resolved_holdings] market_id={mid} tokens={tokens}", flush=True)

        # resolve tx
        resolve_tx = (
            db.query(Tx)
              .filter(Tx.user_id == user_id, Tx.market_id == mid, Tx.action == "Resolve")
              .order_by(Tx.ts.desc(), Tx.id.desc())
              .first()
        )
        print(f"[resolved_holdings] resolve_tx={'YES' if resolve_tx else 'NO'}", flush=True)
        if resolve_tx:
            print(f"  ↳ token={resolve_tx.token} sell_delta={resolve_tx.sell_delta}", flush=True)

        winner_tok = (m.winner_token or "").strip()
        winner_upper = winner_tok.upper()
        winner_sell_delta = 0.0
        if resolve_tx and (resolve_tx.token or "").upper() == winner_upper:
            winner_sell_delta = float(resolve_tx.sell_delta or 0.0)

        for tok in tokens:
            shares_at = _shares_at_ts(
                db=db, user_id=user_id, market_id=mid, token=tok, ts_iso=resolved_iso
            )
            print(f"[resolved_holdings] market={mid} token={tok} shares_at={shares_at}", flush=True)

            if shares_at <= 0: # bug
                continue
            sell_delta = winner_sell_delta if tok.upper() == winner_upper else 0.0
            out.append(
                ResolvedHoldingOut(
                    user_id=user_id,
                    market_id=mid,
                    token=tok,
                    shares=int(shares_at),
                    sell_delta=sell_delta,
                    resolved_ts=resolved_iso,
                )
            )

    print(f"[resolved_holdings] total rows={len(out)}", flush=True)
    return out


def _user_out_by_filter(
    db: Session,
    *,
    user_id: int | None = None,
    username: str | None = None,
    market_id: int | None = None,
) -> UserOut:
    # Find the user by id or username (exactly one must be provided)
    if (user_id is None) == (username is None):
        raise HTTPException(status_code=400, detail="Provide exactly one of user_id or username")

    if user_id is not None:
        user = db.get(User, user_id)
    else:
        user = (
            db.query(User)
              .filter(func.lower(User.username) == func.lower(username.strip()))
              .first()
        )

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    print(f"[user_out] user id={user.id} name='{user.username}' market_filter={market_id}", flush=True)

    # --- Live holdings (optionally filtered by market) ---
    # --- Live holdings (optionally filtered by market) ---
    q = db.query(Holding).filter(Holding.user_id == user.id)

    # exclude resolved markets unless user explicitly filters a specific market_id that is still active
    if market_id is not None:
        # If they asked for a specific market, we’ll still include it if it’s active.
        q = q.join(Market, Market.id == Holding.market_id) \
            .filter(Holding.market_id == market_id, Market.resolved == 0)
    else:
        # No filter → show only active markets
        q = q.join(Market, Market.id == Holding.market_id).filter(Market.resolved == 0)

    holdings_rows = q.order_by(Holding.token.asc()).all()
    print("user hold")
    # Replace ====
    # q = db.query(Holding).filter(Holding.user_id == user.id)
    # if market_id is not None:
    #     q = q.filter(Holding.market_id == market_id)
    # holdings_rows = q.order_by(Holding.token.asc()).all()

    # Prefetch reserves + market status to decide whether to include in holdings
    m_ids = {h.market_id for h in holdings_rows if h.market_id is not None}
    toks = {h.token for h in holdings_rows}
    res_map: Dict[Tuple[int, str], int] = {}
    resolved_markets: set[int] = set()

    if m_ids:
        # Get reserves
        reserves = (
            db.query(Reserve)
            .filter(Reserve.market_id.in_(m_ids), Reserve.token.in_(toks))
            .all()
        )
        res_map = {(r.market_id, r.token): int(r.shares or 0) for r in reserves}

        # Get resolution status for those markets
        mkts = db.query(Market).filter(Market.id.in_(m_ids)).all()
        resolved_markets = {m.id for m in mkts if int(m.resolved or 0) == 1}

    out_holdings: List[HoldingOut] = []
    for h in holdings_rows:
        # Skip resolved markets entirely
        if h.market_id in resolved_markets:
            print(f"[user_out] skip resolved market {h.market_id} for token {h.token}", flush=True)
            continue

        key = (h.market_id, h.token) if h.market_id is not None else None
        reserve_shares = res_map.get(key, 0) if key else 0
        price_now = float(buy_curve(reserve_shares))
        out_holdings.append(
            HoldingOut(
                user_id=h.user_id,
                market_id=h.market_id,
                token=h.token,
                shares=int(h.shares or 0),
                price=price_now,
                value=price_now * int(h.shares or 0),
            )
        )

    # --- Resolved holdings (held-to-resolution) ---
    resolved_holdings = build_resolved_holdings_for_user(
        db=db,
        user_id=user.id,
        market_id=market_id,
    )
    print(f"[user_out] resolved_holdings rows={len(resolved_holdings)}", flush=True)

    return UserOut(
        id=user.id,
        username=user.username,
        balance=user.balance,
        holdings=out_holdings,
        holdings_by_token={h.token: int(h.shares or 0) for h in holdings_rows},
        resolved_holdings=resolved_holdings,   # <-- requires your schema to have this field
    )

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


# ====== Helper Functions ======
def _enrich_holdings_with_prices(
    db: Session,
    holdings_rows: list[Holding],
) -> list[HoldingOut]:
    """Attach price/value (using buy curve on current reserve) to raw Holding rows."""
    if not holdings_rows:
        return []

    # Prefetch reserves for all (market_id, token) pairs we need
    m_ids = {h.market_id for h in holdings_rows if h.market_id is not None}
    toks  = {h.token for h in holdings_rows}
    res_map: dict[tuple[int, str], int] = {}

    if m_ids and toks:
        reserves = (
            db.query(Reserve)
              .filter(Reserve.market_id.in_(m_ids), Reserve.token.in_(toks))
              .all()
        )
        res_map = {(r.market_id, r.token): int(r.shares or 0) for r in reserves}

    # Build enriched outputs
    out: list[HoldingOut] = []
    for h in holdings_rows:
        key = (h.market_id, h.token)
        reserve_shares = res_map.get(key, 0)
        price_now = float(buy_curve(reserve_shares))
        shares = int(h.shares or 0)
        out.append(
            HoldingOut(
                user_id=h.user_id,
                market_id=h.market_id,
                token=h.token,
                shares=shares,
                price=price_now,
                value=price_now * shares,
            )
        )
    return out


def _canonical_token_for_market(db: Session, market_id: int, token_in: str) -> str | None:
    """Return the exact-cased token configured for this market, matching token_in case-insensitively."""
    toks = list_market_outcomes(db, market_id)  # e.g. ["Bilibili Gaming", "Others", ...]
    lut = {t.upper(): t for t in toks}
    return lut.get((token_in or "").strip().upper())


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    with next(get_db()) as db:
        m = db.get(Market, 1)
        if not m:
            now_dt = datetime.utcnow().replace(microsecond=0)
            # end_dt = now_dt + timedelta(days=20)
            end_dt = datetime(2025, 9, 15, 0, 0)  # ← Sept 15 @ 00:00 (UTC, naive)
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
    response_model=List[UserSummaryOut],
    tags=["Users"],
    summary="List all users",
    description="Fetch all users currently registered in the system."
)
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.id.asc()).all()


# Fetch user by user_id
@app.get(
    "/users/{user_id}",
    response_model=UserOut,
    tags=["Users"],
    summary="Get user by ID",
    description="Returns the user by ID."
)
def get_user(
    user_id: int,
    market_id: Annotated[int | None, Query(ge=1, description="Optional market filter")] = None,
    db: Session = Depends(get_db),
):
    return _user_out_by_filter(db, user_id=user_id, market_id=market_id)


# Fetch user data by username (unchanged)
@app.get(
    "/users/by-username/{username}",
    response_model=UserOut,
    tags=["Users"],
    summary="Get user by username"
)
def get_user_by_username(
    username: Annotated[str, Path(..., min_length=1)],
    market_id: Annotated[int | None, Query(ge=1)] = None,
    db: Session = Depends(get_db),
):
    return _user_out_by_filter(db, username=username, market_id=market_id)


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
        return _user_out_by_filter(db, user_id=user.id, market_id=None)
        # return UserOut(id=user.id, username=user.username, balance=user.balance)

    # create new
    balance = SPECIAL_STARTING_BALANCE if uname == 'jack' else STARTING_BALANCE
    user = User(username=uname, balance=balance, created_at=func.now())
    db.add(user)
    db.flush()  # gets user.id

    db.commit()
    db.refresh(user)
    return _user_out_by_filter(db, user_id=user.id, market_id=None)


# ====== Admin ======
@app.post("/admin/{market_id}/start", response_model=MarketOut,
          tags=["Admin"], summary="Start market",
          description="Starts a market. Requires admin password in body.")
def admin_start(
    market_id: int,
    req: AdminStartRequest = Body(...),
    db: Session = Depends(get_db),
):
    _assert_admin(req.password)

    # ---- 1) Resolve desired outcomes (support legacy 'tokens')
    requested = None
    if getattr(req, "outcomes", None):
        requested = req.outcomes
    elif getattr(req, "tokens", None):
        requested = req.tokens

    if requested:
        # unique by UPPER(), preserve first occurrence casing for display
        desired: list[str] = []
        seen_upper: set[str] = set()
        for t in requested:
            if not t:
                continue
            tok = t.strip()
            if not tok:
                continue
            u = tok.upper()
            if u in seen_upper:
                continue
            desired.append(tok)
            seen_upper.add(u)
        if not desired:
            raise HTTPException(status_code=400, detail="At least one outcome required.")
    else:
        desired = None  # decide below

    # ---- 2) Compute start/end timestamps
    now_iso = _now_iso()
    if getattr(req, "end_ts", None):
        try:
            end_iso = parse_to_naive_utc(req.end_ts)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid end_ts (expect ISO-8601, e.g. '2026-06-30T12:00:00Z')",
            )
        if datetime.fromisoformat(end_iso) <= datetime.fromisoformat(now_iso):
            raise HTTPException(status_code=400, detail="end_ts must be strictly after current UTC time")
    else:
        end_iso = (datetime.utcnow() + timedelta(days=(req.duration_days or MARKET_DURATION_DAYS)))\
                    .replace(microsecond=0).isoformat()

    # ---- 3) Upsert market meta & (re)activate
    m = db.get(Market, market_id)
    if not m:
        m = Market(
            id=market_id,
            start_ts=now_iso,
            end_ts=end_iso,
            winner_token=None,
            resolved=0,
            resolved_ts=None,
            question=req.question,
            resolution_note=req.resolution_note,
        )
        db.add(m)
        db.flush()
    else:
        if is_market_active(m):
            raise HTTPException(status_code=400, detail="Market already active.")
        m.start_ts = now_iso
        m.end_ts = end_iso
        m.winner_token = None
        m.resolved = 0
        m.resolved_ts = None
        if req.question is not None:
            m.question = req.question
        if req.resolution_note is not None:
            m.resolution_note = req.resolution_note

    # ---- 4) Determine final desired outcomes (use existing or defaults if none provided)
    existing_mo = db.query(MarketOutcome).filter(MarketOutcome.market_id == market_id).all()
    existing_tokens = [mo.token for mo in existing_mo]

    if desired is None:
        desired = existing_tokens if existing_tokens else list(DEFAULT_OUTCOMES)

    upper_desired  = {t.upper() for t in desired}
    upper_existing = {t.upper() for t in existing_tokens}

    # ---- 5) Remove outcomes no longer desired (and scrub reserves & holdings for those tokens)
    for tok_upper in (upper_existing - upper_desired):
        db.query(MarketOutcome).filter(
            MarketOutcome.market_id == market_id,
            func.upper(MarketOutcome.token) == tok_upper
        ).delete(synchronize_session=False)

        db.query(Reserve).filter(
            Reserve.market_id == market_id,
            func.upper(Reserve.token) == tok_upper
        ).delete(synchronize_session=False)

        db.query(Holding).filter(
            Holding.market_id == market_id,
            func.upper(Holding.token) == tok_upper
        ).delete(synchronize_session=False)

    # ---- 6) Insert/update desired outcomes (preserve order via sort_order)
    for i, tok in enumerate(desired):
        mo = db.query(MarketOutcome).filter(
            MarketOutcome.market_id == market_id,
            func.upper(MarketOutcome.token) == tok.upper()
        ).first()
        if not mo:
            db.add(MarketOutcome(
                market_id=market_id,
                token=tok,
                display=tok,
                sort_order=i
            ))
        else:
            mo.sort_order = i
            if not mo.display:
                mo.display = tok  # normalize display if previously empty

    # ---- 7) Ensure reserves exist for each desired token (case-insensitive) and normalize casing
    for tok in desired:
        r = db.query(Reserve).filter(
            Reserve.market_id == market_id,
            func.upper(Reserve.token) == tok.upper()
        ).first()
        if not r:
            db.add(Reserve(market_id=market_id, token=tok, shares=0, usdc=0.0))
        else:
            # normalize canonical casing to match `tok`
            if r.token != tok:
                r.token = tok

    # Optionally zero reserves (shares/usdc) if requested
    if getattr(req, "reset_reserves", False):
        db.query(Reserve).filter(Reserve.market_id == market_id).update(
            {Reserve.shares: 0, Reserve.usdc: 0.0},
            synchronize_session=False
        )

    # ---- 8) Wipe ALL holdings for this market to prevent carry-over between seasons
    db.query(Holding).filter(Holding.market_id == market_id).delete(synchronize_session=False)

    db.commit()
    db.refresh(m)
    return _market_to_out(db, m)


@app.post(
    "/admin/{market_id}/delete",
    tags=["Admin"],
    summary="DELETE one market and all its data",
    description="Hard-deletes the market and all related rows (reserves, outcomes, holdings, tx). Requires admin password."
)
def admin_delete_market(
    market_id: int,
    req: AdminDeleteMarketRequest = Body(...),
    db: Session = Depends(get_db),
):
    _assert_admin(req.password)

    m = db.get(Market, market_id)
    if not m:
        raise HTTPException(status_code=404, detail="Market not found")

    # Delete children first (if you don't have ON DELETE CASCADE)
    tx_del        = db.execute(delete(Tx).where(Tx.market_id == market_id)).rowcount
    hold_del      = db.execute(delete(Holding).where(Holding.market_id == market_id)).rowcount
    res_del       = db.execute(delete(Reserve).where(Reserve.market_id == market_id)).rowcount

    # If you have a MarketOutcome table, include it:
    try:
        mo_del = db.execute(delete(MarketOutcome).where(MarketOutcome.market_id == market_id)).rowcount
    except NameError:
        mo_del = 0  # table not present

    # Finally delete the market
    db.delete(m)
    db.commit()

    return {
        "status": "ok",
        "deleted": {
            "market_id": market_id,
            "tx": tx_del,
            "holdings": hold_del,
            "reserves": res_del,
            "outcomes": mo_del,
            "markets": 1
        }
    }

# Resolve market
@app.post(
    "/admin/{market_id}/resolve",
    tags=["Admin"],
    summary="Resolve a market",
    description="Resolves a market. Requires admin password in body.",
)
def admin_resolve(
    market_id: int,
    payload: AdminResolveRequest = Body(...),
    db: Session = Depends(get_db),
):
    _assert_admin(payload.password)

    # --- validate market exists and not already resolved
    m = db.get(Market, market_id)
    if not m:
        raise HTTPException(status_code=404, detail="Market not found")
    if int(m.resolved or 0) == 1:
        raise HTTPException(status_code=400, detail="Market already resolved")

    # --- fetch valid outcomes (case-insensitive)
    mo_rows = db.query(MarketOutcome.token).filter(MarketOutcome.market_id == market_id).all()
    valid_tokens_upper = {(t or "").upper() for (t,) in mo_rows}
    if not valid_tokens_upper:
        raise HTTPException(status_code=400, detail="Market has no configured outcomes")

    winner_upper = (payload.winner_token or "").strip().upper()
    if winner_upper not in valid_tokens_upper:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid winner_token for market {market_id}. Allowed: {sorted(valid_tokens_upper)}",
        )

    # Resolve canonical casing for winner token (keep DB’s canonical display/case)
    token_canonical = None
    for (t,) in mo_rows:
        if (t or "").upper() == winner_upper:
            token_canonical = t
            break
    if not token_canonical:
        token_canonical = payload.winner_token.strip()  # fallback

    # --- pool and winning supply for THIS market
    tot_pool = float(
        db.query(func.coalesce(func.sum(Reserve.usdc), 0.0))
          .filter(Reserve.market_id == market_id)
          .scalar() or 0.0
    )

    win_res = (
        db.query(Reserve)
          .filter(Reserve.market_id == market_id, func.upper(Reserve.token) == winner_upper)
          .first()
    )
    win_shares = int(win_res.shares if win_res else 0)
 
    # --- compute payouts + capture per-user shares (at resolution)
    payouts_by_user: dict[int, float] = {}
    shares_by_user: dict[int, int] = {}

    if tot_pool > 0 and win_shares > 0:
        # Use current holdings as the authoritative snapshot for resolution
        winners = (
            db.query(Holding)
              .filter(
                  Holding.market_id == market_id,
                  func.upper(Holding.token) == winner_upper,
                  Holding.shares > 0,
              )
              .all()
        )
        for h in winners:
            share = int(h.shares or 0)
            if share <= 0:
                continue
            payout = tot_pool * (share / win_shares)
            payouts_by_user[h.user_id] = payouts_by_user.get(h.user_id, 0.0) + float(payout)
            shares_by_user[h.user_id]  = shares_by_user.get(h.user_id, 0) + share

    # --- credit users & log resolution tx (qty = winner shares at resolution)
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat()
    for uid, amt in payouts_by_user.items():
        u = db.get(User, uid)
        if not u:
            continue

        u.balance = float(u.balance or 0.0) + float(amt)
        outstanding_shares = int(shares_by_user.get(uid, 0))
        db.add(
            Tx(
                ts=now_iso,
                user_id=uid,
                market_id=market_id,
                user_name=u.username,
                action="Resolve",
                token=token_canonical,
                qty=outstanding_shares,  # ← actual winner shares
                qty_change=outstanding_shares,                          # ← resolves don't change circulating supply
                buy_price=None,
                sell_price=None,
                buy_delta=None,
                sell_delta=float(amt),
                balance_after=u.balance,
            )
        )

    # --- clear reserves for this market
    for r in db.query(Reserve).filter(Reserve.market_id == market_id).all():
        # r.shares = 0
        r.usdc = 0.0 # only update this

    win_res.usdc = tot_pool # update win_res usdc as total pool for archival

    # (Optional) zero holdings for this market if you want a hard wipe:
    # db.query(Holding).filter(Holding.market_id == market_id).update({Holding.shares: 0})

    m.winner_token = token_canonical
    m.resolved = 1
    m.resolved_ts = now_iso
    m.end_ts = now_iso

    db.commit()
    return {"status": "ok", "payout_pool": tot_pool, "winner_token": token_canonical}

# Market Configure
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

    # --- end_ts (must be in the future) ---
    # Accepts ISO-8601 strings like '2026-06-30T12:00:00', '2026-06-30T12:00:00Z', or with offsets
    if getattr(req, "end_ts", None) is not None:
        try:
            new_end_iso = parse_to_naive_utc(req.end_ts)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid end_ts format (expected ISO-8601)")

        now_iso = _now_iso()
        now_dt = datetime.fromisoformat(now_iso)
        end_dt = datetime.fromisoformat(new_end_iso)
        if end_dt <= now_dt:
            raise HTTPException(status_code=400, detail="end_ts must be strictly after current UTC time")

        m.end_ts = new_end_iso

    if req.outcomes is not None:
        tokens = [t.strip().upper() for t in req.outcomes if t.strip()]
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
@app.get(
    "/markets",
    response_model=List[MarketWithReservesOut],
    tags=["Market"],
    summary="List all markets (with outcomes & reserves)"
)
def list_markets(db: Session = Depends(get_db)):
    """Return all markets with outcomes (token/display/sort_order) and reserves."""
    ms = db.query(Market).order_by(Market.id.asc()).all()
    if not ms:
        return []

    market_ids = [m.id for m in ms]

    # Batch fetch outcomes for all markets; preserve sort_order asc then token asc
    mo_rows = (
        db.query(MarketOutcome)
          .filter(MarketOutcome.market_id.in_(market_ids))
          .order_by(MarketOutcome.market_id.asc(),
                    MarketOutcome.sort_order.asc(),
                    MarketOutcome.token.asc())
          .all()
    )
    outcomes_by_mid = defaultdict(list)
    for o in mo_rows:
        outcomes_by_mid[o.market_id].append(
            OutcomeOut(
                token=o.token,
                display=o.display,
                sort_order=int(o.sort_order or 0),
            )
        )

    # Batch fetch reserves for all markets
    res_rows = (
        db.query(Reserve)
          .filter(Reserve.market_id.in_(market_ids))
          .all()
    )
    reserves_by_mid = defaultdict(list)
    for r in res_rows:
        reserves_by_mid[r.market_id].append(r)  # your schema already accepts this model instance

    # Build full outputs
    out: List[MarketWithReservesOut] = []
    for m in ms:
        out.append(
            MarketWithReservesOut(
                id=m.id,
                start_ts=m.start_ts,
                end_ts=m.end_ts,
                winner_token=m.winner_token,
                resolved=m.resolved,
                resolved_ts=m.resolved_ts,
                question=m.question or "",
                resolution_note=m.resolution_note or "",
                outcomes=outcomes_by_mid.get(m.id, []),
                reserves=reserves_by_mid.get(m.id, []),
            )
        )
    return out

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
    # resolve canonical token for this market (case-insensitive)
    token_canonical = _canonical_token_for_market(db, market_id, token)
    if not token_canonical:
        valid = list_market_outcomes(db, market_id)
        raise HTTPException(status_code=400, detail=f"Invalid token for market {market_id}. Valid: {valid}")

    data = ownership_breakdown(
        db, market_id, token_canonical,
        min_shares=min_shares,
        limit=limit,
        order=order,
    )
    return data


# ====== User Holdings ======
# # Fetch user holdings via user_id
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
    rows = q.order_by(Holding.token.asc()).all()
    return _enrich_holdings_with_prices(db, rows)

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
    rows = q.order_by(Holding.token.asc()).all()
    return _enrich_holdings_with_prices(db, rows)

# ====== Transaction Logs ======
@app.get(
    "/tx",
    response_model=List[TxOut],
    tags=["Transactions"],
    summary="List transaction logs (filter, latest-N, pagination)",
    description=(
        "Returns transaction logs ordered by timestamp. "
        "Use `market_id` to filter; `limit` to fetch the latest N; "
        "`page` + `page_size` for pagination; and `sort` to choose ordering."
    ),
)
def get_tx(
    response: Response,
    market_id: Annotated[int | None, Query(ge=1, description="Filter by market id")] = None,
    # Choose ordering; default asc to keep backward-compat with your existing behavior
    sort: Annotated[Literal["asc", "desc"], Query(description="Sort by timestamp")] = "asc",
    # Quick 'latest N' (ignores pagination if set). Bound to reasonable max.
    limit: Annotated[int | None, Query(ge=1, le=1000, description="Return only the latest N entries (ignores pagination)")] = None,
    # Pagination (1-based page index)
    page: Annotated[int | None, Query(ge=1, description="1-based page index")] = None,
    page_size: Annotated[int | None, Query(ge=1, le=1000, description="Items per page")] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Tx)
    if market_id is not None:
        q = q.filter(Tx.market_id == market_id)

    # Apply ordering first
    order_col = Tx.ts.desc() if sort == "desc" else Tx.ts.asc()
    q = q.order_by(order_col)

    # ---- Latest N (takes precedence over pagination)
    if limit is not None:
        rows = q.limit(limit).all()
        # For latest-N we expose count of returned rows (not total) to avoid an extra COUNT(*)
        response.headers["X-Total-Count"] = str(len(rows))
        return rows

    # ---- Pagination
    # If either `page` or `page_size` is provided, default the other sensibly
    if page is not None or page_size is not None:
        page = page or 1
        page_size = page_size or 50

        total = q.count()
        rows = q.offset((page - 1) * page_size).limit(page_size).all()

        response.headers["X-Total-Count"] = str(total)
        response.headers["X-Page"] = str(page)
        response.headers["X-Page-Size"] = str(page_size)
        return rows

    # ---- Default (no limit, no pagination)
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
@app.get(
    "/leaderboard",
    response_model=List[LeaderboardRow],
    tags=["Leaderboard"],
    summary="View leaderboard standings",
    description="Leaderboard computed from trades in the selected market. "
                "If market_id is omitted, uses all tx across all markets (global view)."
)
def leaderboard(
    market_id: Annotated[int | None, Query(ge=1, description="Optional market filter")] = None,
    db: Session = Depends(get_db),
):
    rows = compute_user_points(db, market_id=market_id)
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
    # resolve canonical token for this market (case-insensitive)
    token_canonical = _canonical_token_for_market(db, market_id, token)
    if not token_canonical:
        valid = list_market_outcomes(db, market_id)
        raise HTTPException(status_code=400, detail=f"Invalid token for market {market_id}. Valid: {valid}")

    # fetch reserve using case-insensitive match, or use canonical value
    r = (
        db.query(Reserve)
          .filter(
              Reserve.market_id == market_id,
              func.upper(Reserve.token) == token_canonical.upper(),  # robust to casing
          )
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
        token=token_canonical,  # return the canonical outcome name
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