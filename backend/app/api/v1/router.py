"""API v1 Router - aggregates all endpoint groups."""

from fastapi import APIRouter
from app.api.v1.endpoints import health, instruments, market_data, strategies, backtests, data_ingestion

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(instruments.router, prefix="/instruments", tags=["Instruments"])
api_router.include_router(market_data.router, prefix="/market", tags=["Market Data"])
api_router.include_router(strategies.router, prefix="/strategies", tags=["Strategies"])
api_router.include_router(backtests.router, prefix="/backtests", tags=["Backtests"])
api_router.include_router(data_ingestion.router, prefix="/data", tags=["Data Ingestion"])
