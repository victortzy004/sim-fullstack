from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Literal

class HealthOut(BaseModel):
    status: Literal["ok"]

class StartRequest(BaseModel):
    duration_days: Optional[int] = None     # default to MARKET_DURATION_DAYS if None
    reset_reserves: bool = False            # optionally zero reserves on start

class UserCreate(BaseModel):
    username: str
    
class UserOut(BaseModel):
    id: int
    username: str
    balance: float
    class Config: from_attributes = True

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

# Extends your ResetRequest (adds 'password')
class AdminResetRequest(ResetRequest):
    password: str = Field(..., min_length=1, description="Admin password")

# Resolve needs a structured payload
class AdminResolveRequest(BaseModel):
    password: str = Field(..., min_length=1, description="Admin password")
    winner_token: str = Field(..., description="Winning outcome token")

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
    model_config = ConfigDict(from_attributes=True)

class MarketWithReservesOut(MarketOut):
    reserves: List[ReserveOut] = []


class HoldingOut(BaseModel):
    user_id: int
    token: str
    shares: int

class TxOut(BaseModel):
    id: int
    ts: str
    user_id: int
    action: str
    token: str
    qty: int
    buy_price: Optional[float]
    sell_price: Optional[float]
    buy_delta: Optional[float]
    sell_delta: Optional[float]
    balance_after: Optional[float]
    class Config: from_attributes = True

class TradeIn(BaseModel):
    user_id: int
    token: str
    mode: str  # "qty" or "usdc"
    amount: float  # qty (int-ish) if mode=qty, otherwise USDC
    side: str  # "buy" or "sell"

class LeaderboardRow(BaseModel):
    user: str
    portfolio_value: float
    pnl: float
    volume_points: float
    pnl_points: float
    total_points: float

class PointsTimelineRow(BaseModel):
    ts: str
    user: str
    total_points: float


# ---- Preview Schemas ----
class PreviewRequest(BaseModel):
    quantity: int

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