"""Celery task definitions for async operations."""

from app.workers import celery_app
from app.core.logging import get_logger

logger = get_logger("workers.tasks")


@celery_app.task(bind=True, name="app.workers.tasks.daily_ingestion")
def daily_ingestion(self):
    """Run daily data ingestion across all enabled providers."""
    logger.info("Starting daily data ingestion")
    # Will be implemented with actual provider adapters
    return {"status": "completed", "providers_processed": 0}


@celery_app.task(bind=True, name="app.workers.tasks.daily_risk_snapshot")
def daily_risk_snapshot(self):
    """Compute and store daily risk snapshot."""
    logger.info("Starting daily risk snapshot")
    return {"status": "completed"}


@celery_app.task(bind=True, name="app.workers.tasks.run_backtest")
def run_backtest(self, run_id: str, config: dict):
    """Execute a backtest run asynchronously."""
    logger.info("Starting backtest", run_id=run_id)
    return {"run_id": run_id, "status": "completed"}


@celery_app.task(bind=True, name="app.workers.tasks.ingest_provider")
def ingest_provider(self, provider: str, symbols: list = None, full_refresh: bool = False):
    """Ingest data from a specific provider."""
    logger.info("Starting provider ingestion", provider=provider)
    return {"provider": provider, "status": "completed", "records": 0}
