"""Market data endpoints - bars, quotes, options chains, curves, macro."""

from typing import Optional, List
from datetime import date, datetime
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


class BarResponse(BaseModel):
    instrument_id: int
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_factor: float = 1.0
    source: str


class BarsListResponse(BaseModel):
    bars: List[BarResponse]
    total: int


class DataHealthResponse(BaseModel):
    feeds: dict
    stale_symbols: List[str]
    last_successful_pull: Optional[str] = None
    quarantine_count: int


@router.get("/bars", response_model=BarsListResponse)
async def get_bars(
    instrument_ids: str = Query(..., description="Comma-separated instrument IDs"),
    freq: str = Query("1d", description="Bar frequency: 1d, 1m"),
    start: Optional[date] = None,
    end: Optional[date] = None,
    adjusted: bool = Query(True, description="Apply corporate action adjustments"),
    as_of: Optional[datetime] = Query(None, description="Point-in-time as-of timestamp"),
):
    """Get normalized OHLCV bars with point-in-time support."""
    return BarsListResponse(bars=[], total=0)


@router.get("/health", response_model=DataHealthResponse)
async def data_health():
    """Get data feed health status."""
    return DataHealthResponse(
        feeds={
            "yahoo_finance": "pending",
            "fred": "pending",
            "fmp": "pending",
            "polygon": "pending",
            "reddit": "pending",
            "trading_economics": "pending",
            "b3_anbima": "pending",
            "fiscal_ai": "pending",
        },
        stale_symbols=[],
        quarantine_count=0,
    )
