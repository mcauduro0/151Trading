"""Instrument reference data endpoints."""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class InstrumentResponse(BaseModel):
    id: int
    symbol: str
    asset_class: str
    subtype: Optional[str] = None
    venue: Optional[str] = None
    currency: str
    timezone: str
    active: bool


class InstrumentListResponse(BaseModel):
    instruments: List[InstrumentResponse]
    total: int


@router.get("", response_model=InstrumentListResponse)
async def list_instruments(
    asset_class: Optional[str] = Query(None, description="Filter by asset class"),
    symbol: Optional[str] = Query(None, description="Filter by symbol (partial match)"),
    active: Optional[bool] = Query(True, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List instruments with optional filters."""
    # Placeholder - will be connected to database in Phase 2
    return InstrumentListResponse(instruments=[], total=0)


@router.get("/{instrument_id}", response_model=InstrumentResponse)
async def get_instrument(instrument_id: int):
    """Get instrument by ID."""
    # Placeholder
    return InstrumentResponse(
        id=instrument_id,
        symbol="PLACEHOLDER",
        asset_class="equity",
        currency="USD",
        timezone="America/New_York",
        active=True,
    )
