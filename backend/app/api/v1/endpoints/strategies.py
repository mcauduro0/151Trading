"""Strategy catalog and management endpoints."""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


class StrategyResponse(BaseModel):
    strategy_id: str
    code: str
    name: str
    source_book: str
    asset_class: str
    style: Optional[str] = None
    sub_style: Optional[str] = None
    horizon: Optional[str] = None
    directionality: Optional[str] = None
    complexity: Optional[str] = None
    status: str
    owner: Optional[str] = None
    description: Optional[str] = None


class StrategyListResponse(BaseModel):
    strategies: List[StrategyResponse]
    total: int
    filters_applied: dict


class StrategyDetailResponse(StrategyResponse):
    parameters: List[dict] = []
    data_requirements: List[str] = []
    known_failure_modes: List[str] = []
    capacity_notes: Optional[str] = None
    math_formula: Optional[str] = None
    backtest_count: int = 0


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    asset_class: Optional[str] = Query(None),
    style: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    source_book: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Search in name and description"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List strategies with filtering, search, and pagination."""
    return StrategyListResponse(
        strategies=[],
        total=0,
        filters_applied={
            "asset_class": asset_class,
            "style": style,
            "status": status,
            "source_book": source_book,
        },
    )


@router.get("/{strategy_id}", response_model=StrategyDetailResponse)
async def get_strategy(strategy_id: str):
    """Get full strategy detail including parameters, data requirements, and failure modes."""
    return StrategyDetailResponse(
        strategy_id=strategy_id,
        code="PLACEHOLDER",
        name="Placeholder Strategy",
        source_book="151TS",
        asset_class="equity",
        status="research_only",
    )
