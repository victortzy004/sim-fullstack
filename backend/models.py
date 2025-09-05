from sqlalchemy import Column, Integer, Float, String, ForeignKey, Index, UniqueConstraint, Text
from sqlalchemy.orm import relationship
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    balance = Column(Float, nullable=False)
    created_at = Column(String, nullable=False)

class Market(Base):
    __tablename__ = "markets"  # <- if your real table is "market", use that consistently

    id = Column(Integer, primary_key=True)
    # store as strings to match your current code (isoformat())
    start_ts = Column(String, nullable=False)
    end_ts = Column(String, nullable=False)
    winner_token = Column(String, nullable=True)
    resolved = Column(Integer, nullable=False, default=0)  # 0/1 flag
    resolved_ts = Column(String, nullable=True)

    # Metadata
    question = Column(Text, nullable=True)
    resolution_note = Column(Text, nullable=True)

    reserves = relationship(
        "Reserve",
        back_populates="market",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    outcomes = relationship(
        "MarketOutcome",
        back_populates="market",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class MarketOutcome(Base):
    """
    Canonical outcomes (tokens) for a market. Trading/resolve should validate
    against this set.
    """
    __tablename__ = "market_outcomes"

    market_id = Column(
        Integer,
        ForeignKey("markets.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    token = Column(String, primary_key=True)      # e.g. "YES"
    display = Column(String, nullable=True)       # optional UI label
    sort_order = Column(Integer, nullable=False, default=0)

    market = relationship("Market", back_populates="outcomes")

    __table_args__ = (
        UniqueConstraint("market_id", "token", name="uq_market_outcome"),
        Index("ix_market_outcome_market_token", "market_id", "token"),
    )

class Reserve(Base):
    __tablename__ = "reserves"
    id = Column(Integer, primary_key=True)

    market_id = Column(
        Integer,
        ForeignKey("markets.id", ondelete="CASCADE"),  # <- adjust to "market.id" if your table is singular
        nullable=False,
        index=True,
    )
    token  = Column(String, nullable=False)
    shares = Column(Integer, nullable=False, default=0)
    usdc   = Column(Float,   nullable=False, default=0.0)

    market = relationship("Market", back_populates="reserves")

    __table_args__ = (
        UniqueConstraint("market_id", "token", name="uq_reserve_market_token"),
        Index("ix_reserve_market_token", "market_id", "token"),
    )

class Holding(Base):
    __tablename__ = "holdings"
    user_id   = Column(Integer, primary_key=True)
    market_id = Column(Integer, ForeignKey("markets.id", ondelete="CASCADE"), primary_key=True, index=True)
    token     = Column(String,  primary_key=True)
    shares    = Column(Integer, nullable=False)


class Tx(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    market_id = Column(                     
        Integer,
        ForeignKey("markets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ts = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)
    user_name = Column(String, nullable=False)
    action = Column(String, nullable=False)
    token = Column(String, nullable=False)
    qty = Column(Integer, nullable=False)
    buy_price = Column(Float, nullable=True)
    sell_price = Column(Float, nullable=True)
    buy_delta = Column(Float, nullable=True)
    sell_delta = Column(Float, nullable=True)
    balance_after = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_tx_market_ts", "market_id", "ts"),
    )