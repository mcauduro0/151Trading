"""Backtest management endpoints."""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


class BacktestRunRequest(BaseModel):
    strategy_version_id: str
    dataset_manifest_id: Optional[str] = None
    universe_id: Optional[str] = None
    params: dict = {}
    tc_model_id: Optional[str] = None
    benchmark_ids: List[str] = []
    run_mode: str = "single"  # single, grid, walk_forward, portfolio


class BacktestRunResponse(BaseModel):
    run_id: str
    status: str
    queued_at: str


class BacktestResultResponse(BaseModel):
    run_id: str
    status: str
    metrics: dict = {}
    config: dict = {}


@router.post("", response_model=BacktestRunResponse)
async def submit_backtest(request: BacktestRunRequest):
    """Submit a new backtest run."""
    return BacktestRunResponse(
        run_id="placeholder",
        status="queued",
        queued_at="2026-03-09T00:00:00Z",
    )


@router.get("/{run_id}", response_model=BacktestResultResponse)
async def get_backtest(run_id: str):
    """Get backtest run status and results."""
    return BacktestResultResponse(
        run_id=run_id,
        status="pending",
    )


@router.get("/{run_id}/timeseries")
async def get_backtest_timeseries(run_id: str):
    """Get backtest daily PnL, NAV, drawdown, turnover, exposures."""
    return {"run_id": run_id, "timeseries": []}
