"""Data ingestion management endpoints."""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


class IngestionJobResponse(BaseModel):
    job_id: str
    provider: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    records_processed: int = 0
    errors: List[str] = []


class IngestionStatusResponse(BaseModel):
    providers: dict
    last_run: Optional[str] = None
    next_scheduled: Optional[str] = None


@router.post("/ingest/{provider}")
async def trigger_ingestion(
    provider: str,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols to ingest"),
    full_refresh: bool = Query(False, description="Force full data refresh"),
):
    """Trigger data ingestion for a specific provider."""
    return IngestionJobResponse(
        job_id="placeholder",
        provider=provider,
        status="queued",
    )


@router.get("/status", response_model=IngestionStatusResponse)
async def ingestion_status():
    """Get overall ingestion status across all providers."""
    return IngestionStatusResponse(
        providers={
            "yahoo_finance": {"status": "idle", "last_run": None},
            "fred": {"status": "idle", "last_run": None},
            "fmp": {"status": "idle", "last_run": None},
            "polygon": {"status": "idle", "last_run": None},
            "reddit": {"status": "idle", "last_run": None},
            "trading_economics": {"status": "idle", "last_run": None},
            "b3_anbima": {"status": "idle", "last_run": None},
            "fiscal_ai": {"status": "idle", "last_run": None},
        },
    )
