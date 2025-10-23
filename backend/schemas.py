from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, List, Literal, Dict
from datetime import datetime

class HealthOut(BaseModel):
    status: Literal["ok"]

class StartRequest(BaseModel):
    duration_days: Optional[int] = None     # default to MARKET_DURATION_DAYS if None
    reset_reserves: bool = False            # optionally zero reserves on start

class HoldingOut(BaseModel):
    user_id: int
    market_id: int
    token: str
    shares: int
    price: float = 0.0
    value: float = 0.0
    model_config = ConfigDict(from_attributes=True)

class ResolvedHoldingOut(BaseModel):
    """
    Holding that was 'held to resolution' and settled.
    - shares: winner shares the user still held at the instant of resolution
    - sell_delta: the USDC payout credited at resolution
    - resolved_ts: when the resolution occurred (server timestamp)
    """
    user_id: int
    market_id: int
    token: str
    shares: int
    sell_delta: float
    resolved_ts: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    username: str

class UserSummaryOut(BaseModel):
    id: int
    username: str
    balance: float
    model_config = ConfigDict(from_attributes=True)
    
# class UserOut(BaseModel):
#     id: int
#     username: str
#     balance: float
#     holdings: List[HoldingOut] = []          # NEW: embed full rows
#     # (optional convenience: map of token->shares)
#     holdings_by_token: Dict[str, int] = {}   # NEW: handy for UIs
#     model_config = ConfigDict(from_attributes=True)

class UserOut(BaseModel):
    id: int
    username: str
    balance: float
    holdings: List[HoldingOut] = []                 # live holdings
    holdings_by_token: Dict[str, int] = {}          # convenience map for UIs
    resolved_holdings: List[ResolvedHoldingOut] = []  # NEW: holdings settled at resolution
    model_config = ConfigDict(from_attributes=True)
    
class UserWithPointsOut(UserOut):
    volume_points: float
    pnl_points: float
    total_points: float

# Markets
class ResetRequest(BaseModel):
    wipe_users: bool = False

class StartRequest(BaseModel):
    duration_days: Optional[int] = None
    reset_reserves: bool = False

# Extends your StartRequest (adds 'password')
class AdminStartRequest(StartRequest):
    password: str = Field(..., min_length=1, description="Admin password")
    duration_days: Optional[int] = None
    reset_reserves: bool = False
    end_ts: Optional[str] = None 

    # NEW (all optional)
    question: Optional[str] = None
    resolution_note: Optional[str] = None
    outcomes: Optional[List[str]] = None  # e.g. ["YES","NO"]

# Extends your ResetRequest (adds 'password')
class AdminResetRequest(ResetRequest):
    password: str = Field(..., min_length=1, description="Admin password")

# Resolve needs a structured payload
class AdminResolveRequest(BaseModel):
    password: str = Field(..., min_length=1, description="Admin password")
    winner_token: str = Field(..., description="Winning outcome token")

# --- Admin-related ----
class AdminConfigureMarketRequest(BaseModel):
    password: str
    question: str | None = None
    resolution_note: str | None = None
    outcomes: list[str] | None = None  # canonical outcomes list (order matters)
    end_ts: Optional[str] = None 

class AdminDeleteMarketRequest(BaseModel):
    password: str

class ReserveOut(BaseModel):
    market_id: int
    token: str
    shares: int
    usdc: float
    model_config = ConfigDict(from_attributes=True)

class MarketOut(BaseModel):
    id: int
    start_ts: str
    end_ts: str
    winner_token: Optional[str]
    resolved: int
    resolved_ts: Optional[str]
    question: str
    resolution_note: str
    outcomes: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

class OutcomeOut(BaseModel):
    token: str
    display: str | None = None
    sort_order: int
    class Config: from_attributes = True

class MarketWithReservesOut(MarketOut):
    outcomes: List[OutcomeOut] = Field(default_factory=list)
    reserves: List['ReserveOut'] = Field(default_factory=list)



class TxOut(BaseModel):
    id: int
    market_id: int
    ts: str
    user_id: int
    user_name: str
    action: str
    token: str
    qty: int
    qty_change: int
    buy_price: Optional[float]
    sell_price: Optional[float]
    buy_delta: Optional[float]
    sell_delta: Optional[float]
    balance_after: Optional[float]
    class Config: from_attributes = True

class TradeIn(BaseModel):
    user_id: int
    market_id: int
    token: str
    mode: str  # "qty" or "usdc"
    amount: float  # qty (int-ish) if mode=qty, otherwise USDC
    side: str  # "buy" or "sell"

# For Outcome token ownership breakdown
class OwnershipHolder(BaseModel):
    user_id: int
    username: str
    shares: int
    share_pct: float  # 0..1 fraction of total outstanding for this token in this market
    model_config = ConfigDict(from_attributes=True)

class OwnershipBreakdownOut(BaseModel):
    market_id: int
    token: str
    total_shares: int
    holders: List[OwnershipHolder] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True)

class LeaderboardRow(BaseModel):
    user: str
    portfolio_value: float
    pnl: float
    mint_points: float
    realised_pnl_points: float
    total_points: float

class PointsTimelineRow(BaseModel):
    ts: str
    user: str
    total_points: float



# ---- Preview Schemas ----
class PreviewRequest(BaseModel):
    # trade-style inputs
    side: Literal["buy", "sell"] = "buy"
    mode: Literal["qty", "usdc"] = "qty"
    # When mode="qty", interpret `amount` as a quantity (round to nearest int).
    # When mode="usdc", interpret `amount` as USDC.
    amount: Optional[float] = None

    @model_validator(mode="after")
    def _validate_amount(self):
        if self.amount is None:
            raise ValueError("amount is required")
        if self.mode == "usdc" and float(self.amount) <= 0:
            raise ValueError("amount must be > 0 when mode='usdc'")
        if self.mode == "qty" and float(self.amount) < 0:
            raise ValueError("amount (quantity) must be >= 0 when mode='qty'")
        return self

class PreviewResponse(BaseModel):
    market_id: int
    token: str
    reserve: int
    quantity: int
    new_reserve: int
    buy_price: float                 # price after adding quantity
    sell_price_marginal_net: float   # 1-share marginal sell price after tax at current reserve
    buy_amt_delta: float             # USDC needed to buy quantity
    sell_amt_delta: float            # USDC received to sell quantity after tax
    sell_tax_rate_used: float        # 0..1 (for the provided quantity)