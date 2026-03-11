"""
GB Trading — Data Quality Monitor
===================================
Monitors data freshness, detects gaps, identifies outliers,
tracks provider health, and generates quality scores.

All numerical checks apply winsorization (5%-95%).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


def winsorize(data: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    """Winsorize array at given percentiles."""
    lo = np.nanpercentile(data, lower * 100)
    hi = np.nanpercentile(data, upper * 100)
    return np.clip(data, lo, hi)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DataQualityAlert:
    """A data quality alert."""
    provider: str
    metric: str
    severity: AlertSeverity
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProviderHealth:
    """Health status of a data provider."""
    name: str
    status: HealthStatus
    last_successful_fetch: Optional[datetime] = None
    last_error: Optional[str] = None
    latency_ms: float = 0.0
    uptime_pct: float = 100.0
    records_fetched: int = 0
    error_count_24h: int = 0
    quality_score: float = 100.0


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    timestamp: datetime
    overall_score: float
    providers: List[ProviderHealth]
    alerts: List[DataQualityAlert]
    staleness_checks: Dict[str, Dict]
    gap_analysis: Dict[str, Dict]
    outlier_summary: Dict[str, Dict]


class DataQualityMonitor:
    """
    Monitors data quality across all providers.
    
    Checks:
    - Staleness: Is data fresh enough?
    - Gaps: Are there missing data points?
    - Outliers: Are there suspicious values?
    - Provider health: Are APIs responding?
    - Consistency: Do cross-provider values agree?
    """

    # Maximum acceptable staleness per data type (in hours)
    STALENESS_THRESHOLDS = {
        "price_eod": 18,       # EOD data should be < 18h old
        "price_intraday": 1,   # Intraday should be < 1h old
        "fundamentals": 168,   # Weekly fundamental updates
        "macro": 48,           # Macro data within 2 days
        "sentiment": 24,       # Sentiment within 1 day
        "options": 4,          # Options data within 4h
    }

    # Maximum acceptable gap percentage
    GAP_THRESHOLDS = {
        "price_eod": 0.02,     # 2% missing days
        "fundamentals": 0.05,  # 5% missing quarters
        "macro": 0.03,         # 3% missing observations
    }

    def __init__(self):
        self.providers: Dict[str, ProviderHealth] = {}
        self.alerts: List[DataQualityAlert] = []

    def register_provider(self, name: str) -> None:
        """Register a data provider for monitoring."""
        self.providers[name] = ProviderHealth(
            name=name,
            status=HealthStatus.HEALTHY,
        )

    def update_provider_health(
        self,
        name: str,
        success: bool,
        latency_ms: float = 0.0,
        records: int = 0,
        error: Optional[str] = None,
    ) -> ProviderHealth:
        """Update provider health after a fetch attempt."""
        if name not in self.providers:
            self.register_provider(name)

        provider = self.providers[name]
        provider.latency_ms = latency_ms
        provider.records_fetched += records

        if success:
            provider.last_successful_fetch = datetime.utcnow()
            provider.status = HealthStatus.HEALTHY
            if latency_ms > 5000:
                provider.status = HealthStatus.WARNING
                self._add_alert(name, "latency", AlertSeverity.WARNING,
                                f"High latency: {latency_ms:.0f}ms", latency_ms, 5000)
        else:
            provider.error_count_24h += 1
            provider.last_error = error
            if provider.error_count_24h >= 5:
                provider.status = HealthStatus.CRITICAL
                self._add_alert(name, "errors", AlertSeverity.CRITICAL,
                                f"Multiple failures: {provider.error_count_24h} errors in 24h")
            else:
                provider.status = HealthStatus.WARNING
                self._add_alert(name, "error", AlertSeverity.WARNING,
                                f"Fetch error: {error}")

        # Update quality score
        provider.quality_score = self._compute_provider_score(provider)
        return provider

    def check_staleness(
        self,
        data: pd.DataFrame,
        data_type: str = "price_eod",
        provider: str = "unknown",
    ) -> Dict:
        """Check if data is stale (too old)."""
        threshold_hours = self.STALENESS_THRESHOLDS.get(data_type, 24)

        if data.empty or not hasattr(data.index, 'max'):
            self._add_alert(provider, "staleness", AlertSeverity.CRITICAL,
                            f"No data available for {data_type}")
            return {"status": "no_data", "stale": True, "hours_old": None}

        try:
            latest = pd.Timestamp(data.index.max())
            if latest.tzinfo:
                latest = latest.tz_localize(None)
            now = pd.Timestamp.utcnow()
            hours_old = (now - latest).total_seconds() / 3600
        except Exception:
            return {"status": "error", "stale": True, "hours_old": None}

        is_stale = hours_old > threshold_hours

        if is_stale:
            severity = AlertSeverity.CRITICAL if hours_old > threshold_hours * 3 else AlertSeverity.WARNING
            self._add_alert(provider, "staleness", severity,
                            f"Data is {hours_old:.1f}h old (threshold: {threshold_hours}h)",
                            hours_old, threshold_hours)

        return {
            "status": "stale" if is_stale else "fresh",
            "stale": is_stale,
            "hours_old": round(hours_old, 1),
            "threshold_hours": threshold_hours,
            "latest_date": str(latest),
        }

    def check_gaps(
        self,
        data: pd.DataFrame,
        data_type: str = "price_eod",
        provider: str = "unknown",
    ) -> Dict:
        """Detect missing data points / gaps in time series."""
        if data.empty:
            return {"status": "no_data", "gap_pct": 1.0, "gaps": []}

        threshold = self.GAP_THRESHOLDS.get(data_type, 0.05)

        try:
            idx = pd.DatetimeIndex(data.index)
            if data_type == "price_eod":
                # Generate business day range
                full_range = pd.bdate_range(idx.min(), idx.max())
            else:
                full_range = pd.date_range(idx.min(), idx.max(), freq="D")

            missing = full_range.difference(idx)
            gap_pct = len(missing) / len(full_range) if len(full_range) > 0 else 0
        except Exception:
            return {"status": "error", "gap_pct": 0, "gaps": []}

        has_gaps = gap_pct > threshold

        if has_gaps:
            severity = AlertSeverity.CRITICAL if gap_pct > threshold * 3 else AlertSeverity.WARNING
            self._add_alert(provider, "gaps", severity,
                            f"Gap rate {gap_pct:.1%} exceeds {threshold:.1%}",
                            gap_pct, threshold)

        # Find contiguous gap blocks
        gap_blocks = []
        if len(missing) > 0:
            missing_sorted = sorted(missing)
            block_start = missing_sorted[0]
            block_end = missing_sorted[0]
            for d in missing_sorted[1:]:
                if (d - block_end).days <= 3:  # Allow weekends
                    block_end = d
                else:
                    gap_blocks.append({
                        "start": str(block_start.date()),
                        "end": str(block_end.date()),
                        "days": (block_end - block_start).days + 1,
                    })
                    block_start = d
                    block_end = d
            gap_blocks.append({
                "start": str(block_start.date()),
                "end": str(block_end.date()),
                "days": (block_end - block_start).days + 1,
            })

        return {
            "status": "gaps_detected" if has_gaps else "clean",
            "gap_pct": round(gap_pct, 4),
            "threshold": threshold,
            "total_missing": len(missing),
            "total_expected": len(full_range),
            "gap_blocks": gap_blocks[:10],  # Top 10 gaps
        }

    def check_outliers(
        self,
        data: pd.DataFrame,
        column: str = "close",
        provider: str = "unknown",
        z_threshold: float = 4.0,
    ) -> Dict:
        """
        Detect outliers using z-score on winsorized returns.
        """
        if data.empty or column not in data.columns:
            return {"status": "no_data", "outliers": []}

        values = data[column].dropna().values
        if len(values) < 10:
            return {"status": "insufficient_data", "outliers": []}

        # Compute returns and winsorize
        returns = np.diff(values) / values[:-1]
        clean_returns = winsorize(returns)

        mu = np.mean(clean_returns)
        sigma = np.std(clean_returns)
        if sigma == 0:
            return {"status": "zero_variance", "outliers": []}

        z_scores = (returns - mu) / sigma
        outlier_mask = np.abs(z_scores) > z_threshold
        outlier_indices = np.where(outlier_mask)[0]

        outliers = []
        for idx in outlier_indices[:20]:  # Top 20 outliers
            outliers.append({
                "index": int(idx + 1),
                "date": str(data.index[idx + 1]) if hasattr(data.index, '__getitem__') else str(idx + 1),
                "value": float(values[idx + 1]),
                "return": float(returns[idx]),
                "z_score": float(z_scores[idx]),
            })

        if len(outliers) > 0:
            self._add_alert(provider, "outliers", AlertSeverity.WARNING,
                            f"Found {len(outliers)} outliers (z > {z_threshold})",
                            len(outliers), z_threshold)

        return {
            "status": "outliers_found" if outliers else "clean",
            "total_outliers": len(outlier_indices),
            "outlier_pct": round(len(outlier_indices) / len(returns), 4),
            "z_threshold": z_threshold,
            "outliers": outliers,
        }

    def check_cross_provider_consistency(
        self,
        data_a: pd.Series,
        data_b: pd.Series,
        provider_a: str,
        provider_b: str,
        tolerance: float = 0.02,
    ) -> Dict:
        """Check if two providers agree on the same data."""
        common_idx = data_a.index.intersection(data_b.index)
        if len(common_idx) == 0:
            return {"status": "no_overlap", "consistent": False}

        a = data_a.loc[common_idx].values
        b = data_b.loc[common_idx].values

        # Compute relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(a - b) / np.where(np.abs(a) > 1e-10, np.abs(a), 1.0)

        inconsistent_mask = rel_diff > tolerance
        inconsistent_pct = np.mean(inconsistent_mask)

        if inconsistent_pct > 0.05:
            self._add_alert(
                f"{provider_a}/{provider_b}", "consistency", AlertSeverity.WARNING,
                f"Cross-provider mismatch: {inconsistent_pct:.1%} of values differ by > {tolerance:.0%}",
            )

        return {
            "status": "inconsistent" if inconsistent_pct > 0.05 else "consistent",
            "consistent": inconsistent_pct <= 0.05,
            "inconsistent_pct": round(float(inconsistent_pct), 4),
            "max_difference": round(float(np.max(rel_diff)), 4),
            "mean_difference": round(float(np.mean(rel_diff)), 6),
            "overlap_points": len(common_idx),
        }

    # ------------------------------------------------------------------
    # Full quality report
    # ------------------------------------------------------------------
    def generate_report(self) -> DataQualityReport:
        """Generate a comprehensive data quality report."""
        overall_score = self._compute_overall_score()

        return DataQualityReport(
            timestamp=datetime.utcnow(),
            overall_score=overall_score,
            providers=list(self.providers.values()),
            alerts=self.alerts[-50:],  # Last 50 alerts
            staleness_checks={},
            gap_analysis={},
            outlier_summary={},
        )

    def get_dashboard_summary(self) -> Dict:
        """Get a summary suitable for the frontend dashboard."""
        healthy = sum(1 for p in self.providers.values() if p.status == HealthStatus.HEALTHY)
        warning = sum(1 for p in self.providers.values() if p.status == HealthStatus.WARNING)
        critical = sum(1 for p in self.providers.values() if p.status == HealthStatus.CRITICAL)
        offline = sum(1 for p in self.providers.values() if p.status == HealthStatus.OFFLINE)

        recent_alerts = [
            {
                "provider": a.provider,
                "metric": a.metric,
                "severity": a.severity.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in self.alerts[-10:]
        ]

        return {
            "overall_score": self._compute_overall_score(),
            "providers_total": len(self.providers),
            "providers_healthy": healthy,
            "providers_warning": warning,
            "providers_critical": critical,
            "providers_offline": offline,
            "recent_alerts": recent_alerts,
            "provider_details": [
                {
                    "name": p.name,
                    "status": p.status.value,
                    "latency_ms": p.latency_ms,
                    "quality_score": p.quality_score,
                    "last_fetch": p.last_successful_fetch.isoformat() if p.last_successful_fetch else None,
                    "error_count": p.error_count_24h,
                }
                for p in self.providers.values()
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_alert(
        self,
        provider: str,
        metric: str,
        severity: AlertSeverity,
        message: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.alerts.append(DataQualityAlert(
            provider=provider,
            metric=metric,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
        ))

    def _compute_provider_score(self, provider: ProviderHealth) -> float:
        score = 100.0
        if provider.status == HealthStatus.WARNING:
            score -= 20
        elif provider.status == HealthStatus.CRITICAL:
            score -= 50
        elif provider.status == HealthStatus.OFFLINE:
            score = 0
        if provider.latency_ms > 5000:
            score -= 10
        if provider.error_count_24h > 0:
            score -= min(provider.error_count_24h * 5, 30)
        return max(0, score)

    def _compute_overall_score(self) -> float:
        if not self.providers:
            return 100.0
        scores = [p.quality_score for p in self.providers.values()]
        return round(float(np.mean(scores)), 1)
