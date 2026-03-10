"""Celery worker configuration and task definitions."""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "trading_system",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 min soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    beat_schedule={
        "daily-data-ingestion": {
            "task": "app.workers.tasks.daily_ingestion",
            "schedule": 21600.0,  # Every 6 hours
        },
        "daily-risk-snapshot": {
            "task": "app.workers.tasks.daily_risk_snapshot",
            "schedule": 86400.0,  # Daily
        },
    },
)

celery_app.autodiscover_tasks(["app.workers"])
