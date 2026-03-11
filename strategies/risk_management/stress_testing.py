"""
GB Trading — Stress Testing Framework
======================================
Monte Carlo simulation, historical scenario replay, correlation breakdown
analysis, and tail-risk metrics for portfolio stress testing.

Applies winsorization (5%-95%) to all return data before analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from scipy import stats


# ---------------------------------------------------------------------------
# Winsorization utility (5%-95% as per system-wide requirement)
# ---------------------------------------------------------------------------
def winsorize(data: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    """Winsorize array at given percentiles to reduce outlier impact."""
    lo = np.nanpercentile(data, lower * 100)
    hi = np.nanpercentile(data, upper * 100)
    return np.clip(data, lo, hi)


class ScenarioType(Enum):
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    CUSTOM = "custom"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    scenario_type: ScenarioType
    description: str
    shocks: Dict[str, float]  # asset -> return shock
    duration_days: int = 1
    probability: Optional[float] = None


@dataclass
class StressResult:
    """Result of a stress test run."""
    scenario_name: str
    portfolio_loss: float
    portfolio_loss_pct: float
    worst_position: str
    worst_position_loss: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    recovery_days: Optional[int] = None
    position_losses: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-built historical crisis scenarios
# ---------------------------------------------------------------------------
HISTORICAL_SCENARIOS: List[StressScenario] = [
    StressScenario(
        name="2008 GFC",
        scenario_type=ScenarioType.HISTORICAL,
        description="Global Financial Crisis — equity crash, credit freeze, vol spike",
        shocks={"SPY": -0.38, "QQQ": -0.42, "IWM": -0.35, "TLT": 0.20,
                "GLD": 0.05, "VXX": 1.50, "HYG": -0.25, "EEM": -0.45,
                "USO": -0.55, "UUP": 0.12},
        duration_days=120,
        probability=0.02,
    ),
    StressScenario(
        name="2020 COVID Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="Pandemic-driven selloff — rapid drawdown, V-shaped recovery",
        shocks={"SPY": -0.34, "QQQ": -0.28, "IWM": -0.40, "TLT": 0.15,
                "GLD": -0.03, "VXX": 3.00, "HYG": -0.18, "EEM": -0.30,
                "USO": -0.65, "UUP": 0.05},
        duration_days=23,
        probability=0.03,
    ),
    StressScenario(
        name="2022 Rate Shock",
        scenario_type=ScenarioType.HISTORICAL,
        description="Aggressive Fed tightening — bonds and equities fall together",
        shocks={"SPY": -0.25, "QQQ": -0.33, "IWM": -0.22, "TLT": -0.30,
                "GLD": -0.02, "VXX": 0.80, "HYG": -0.15, "EEM": -0.20,
                "USO": 0.30, "UUP": 0.15},
        duration_days=280,
        probability=0.05,
    ),
    StressScenario(
        name="Flash Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="Sudden intraday liquidity vacuum — 2010 style",
        shocks={"SPY": -0.09, "QQQ": -0.10, "IWM": -0.12, "TLT": 0.03,
                "GLD": 0.01, "VXX": 0.60, "HYG": -0.05, "EEM": -0.08,
                "USO": -0.06, "UUP": 0.02},
        duration_days=1,
        probability=0.05,
    ),
    StressScenario(
        name="Stagflation",
        scenario_type=ScenarioType.HISTORICAL,
        description="High inflation + slow growth — 1970s style",
        shocks={"SPY": -0.15, "QQQ": -0.20, "IWM": -0.18, "TLT": -0.20,
                "GLD": 0.25, "VXX": 0.40, "HYG": -0.12, "EEM": -0.15,
                "USO": 0.35, "UUP": -0.08},
        duration_days=365,
        probability=0.04,
    ),
    StressScenario(
        name="EM Currency Crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="Emerging market contagion — capital flight, USD strength",
        shocks={"SPY": -0.08, "QQQ": -0.06, "IWM": -0.10, "TLT": 0.08,
                "GLD": 0.03, "VXX": 0.30, "HYG": -0.10, "EEM": -0.35,
                "USO": -0.15, "UUP": 0.10},
        duration_days=90,
        probability=0.06,
    ),
]


class StressTestEngine:
    """
    Portfolio stress testing engine.
    
    Supports:
    - Historical scenario replay
    - Monte Carlo simulation (correlated returns)
    - Correlation breakdown analysis
    - Tail risk metrics (VaR, CVaR, max drawdown)
    """

    def __init__(self, confidence_levels: Tuple[float, ...] = (0.95, 0.99)):
        self.confidence_levels = confidence_levels
        self.scenarios = list(HISTORICAL_SCENARIOS)

    def add_scenario(self, scenario: StressScenario) -> None:
        """Add a custom stress scenario."""
        self.scenarios.append(scenario)

    # ------------------------------------------------------------------
    # Historical scenario stress test
    # ------------------------------------------------------------------
    def run_historical_scenario(
        self,
        weights: Dict[str, float],
        portfolio_value: float,
        scenario: StressScenario,
    ) -> StressResult:
        """Apply a historical scenario to the portfolio."""
        position_losses: Dict[str, float] = {}
        total_loss = 0.0

        for asset, weight in weights.items():
            shock = scenario.shocks.get(asset, 0.0)
            position_value = portfolio_value * weight
            loss = position_value * shock
            position_losses[asset] = loss
            total_loss += loss

        worst_asset = min(position_losses, key=position_losses.get) if position_losses else "N/A"
        worst_loss = position_losses.get(worst_asset, 0.0)

        return StressResult(
            scenario_name=scenario.name,
            portfolio_loss=total_loss,
            portfolio_loss_pct=total_loss / portfolio_value if portfolio_value else 0,
            worst_position=worst_asset,
            worst_position_loss=worst_loss,
            var_95=total_loss * 0.65,
            var_99=total_loss * 0.85,
            cvar_95=total_loss * 0.75,
            cvar_99=total_loss * 0.95,
            max_drawdown=abs(total_loss / portfolio_value) if portfolio_value else 0,
            recovery_days=scenario.duration_days * 2,
            position_losses=position_losses,
        )

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    def monte_carlo_simulation(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        portfolio_value: float,
        n_simulations: int = 10_000,
        horizon_days: int = 21,
        seed: int = 42,
    ) -> StressResult:
        """
        Run Monte Carlo simulation with correlated returns.
        Returns are winsorized at 5%-95% before simulation.
        """
        rng = np.random.default_rng(seed)

        # Align weights with available return columns
        common_assets = [a for a in weights if a in returns.columns]
        if not common_assets:
            return StressResult(
                scenario_name="Monte Carlo",
                portfolio_loss=0, portfolio_loss_pct=0,
                worst_position="N/A", worst_position_loss=0,
                var_95=0, var_99=0, cvar_95=0, cvar_99=0,
                max_drawdown=0,
            )

        ret_matrix = returns[common_assets].dropna()
        w = np.array([weights[a] for a in common_assets])

        # Winsorize returns (5%-95%)
        clean_returns = pd.DataFrame(
            {col: winsorize(ret_matrix[col].values) for col in ret_matrix.columns},
            index=ret_matrix.index,
        )

        mu = clean_returns.mean().values
        cov = clean_returns.cov().values

        # Cholesky decomposition for correlated sampling
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov += np.eye(len(cov)) * 1e-8
            L = np.linalg.cholesky(cov)

        # Simulate portfolio paths
        portfolio_returns = np.zeros(n_simulations)
        for i in range(n_simulations):
            cumulative = np.zeros(len(common_assets))
            for _ in range(horizon_days):
                z = rng.standard_normal(len(common_assets))
                daily_ret = mu + L @ z
                cumulative += daily_ret
            portfolio_returns[i] = np.dot(w, cumulative)

        # Compute risk metrics
        portfolio_pnl = portfolio_returns * portfolio_value
        var_95 = np.percentile(portfolio_pnl, 5)
        var_99 = np.percentile(portfolio_pnl, 1)
        cvar_95 = portfolio_pnl[portfolio_pnl <= var_95].mean() if np.any(portfolio_pnl <= var_95) else var_95
        cvar_99 = portfolio_pnl[portfolio_pnl <= var_99].mean() if np.any(portfolio_pnl <= var_99) else var_99
        max_dd = abs(np.min(portfolio_pnl) / portfolio_value)

        # Worst asset contribution
        asset_contributions = {}
        for idx, asset in enumerate(common_assets):
            asset_ret = clean_returns[asset].values
            asset_var = np.percentile(asset_ret * weights[asset] * portfolio_value, 5)
            asset_contributions[asset] = asset_var

        worst_asset = min(asset_contributions, key=asset_contributions.get) if asset_contributions else "N/A"

        return StressResult(
            scenario_name=f"Monte Carlo ({n_simulations} sims, {horizon_days}d)",
            portfolio_loss=float(var_95),
            portfolio_loss_pct=float(var_95 / portfolio_value),
            worst_position=worst_asset,
            worst_position_loss=float(asset_contributions.get(worst_asset, 0)),
            var_95=float(abs(var_95)),
            var_99=float(abs(var_99)),
            cvar_95=float(abs(cvar_95)),
            cvar_99=float(abs(cvar_99)),
            max_drawdown=float(max_dd),
            position_losses=asset_contributions,
        )

    # ------------------------------------------------------------------
    # Correlation breakdown analysis
    # ------------------------------------------------------------------
    def correlation_breakdown_analysis(
        self,
        returns: pd.DataFrame,
        lookback_normal: int = 252,
        lookback_stress: int = 21,
        threshold: float = 0.3,
    ) -> Dict:
        """
        Detect correlation regime changes that signal systemic risk.
        Compares recent correlation structure to long-term baseline.
        Returns are winsorized before analysis.
        """
        if len(returns) < lookback_normal:
            return {"status": "insufficient_data", "alerts": []}

        # Winsorize returns
        clean = pd.DataFrame(
            {col: winsorize(returns[col].values) for col in returns.columns},
            index=returns.index,
        )

        normal_corr = clean.iloc[-lookback_normal:].corr()
        stress_corr = clean.iloc[-lookback_stress:].corr()

        # Compute correlation changes
        corr_diff = stress_corr - normal_corr
        alerts = []

        n = len(corr_diff.columns)
        for i in range(n):
            for j in range(i + 1, n):
                asset_i = corr_diff.columns[i]
                asset_j = corr_diff.columns[j]
                change = corr_diff.iloc[i, j]

                if abs(change) > threshold:
                    alerts.append({
                        "pair": f"{asset_i}/{asset_j}",
                        "normal_corr": float(normal_corr.iloc[i, j]),
                        "stress_corr": float(stress_corr.iloc[i, j]),
                        "change": float(change),
                        "severity": "HIGH" if abs(change) > 0.5 else "MEDIUM",
                    })

        # Average correlation level
        mask = np.triu(np.ones_like(stress_corr, dtype=bool), k=1)
        avg_corr_normal = float(normal_corr.values[mask].mean())
        avg_corr_stress = float(stress_corr.values[mask].mean())

        regime = "CRISIS" if avg_corr_stress > 0.7 else "ELEVATED" if avg_corr_stress > 0.5 else "NORMAL"

        return {
            "status": regime,
            "avg_correlation_normal": avg_corr_normal,
            "avg_correlation_stress": avg_corr_stress,
            "correlation_increase": avg_corr_stress - avg_corr_normal,
            "alerts": sorted(alerts, key=lambda x: abs(x["change"]), reverse=True),
            "diversification_ratio": 1.0 - avg_corr_stress,
        }

    # ------------------------------------------------------------------
    # Run all scenarios
    # ------------------------------------------------------------------
    def run_all_scenarios(
        self,
        weights: Dict[str, float],
        portfolio_value: float,
        returns: Optional[pd.DataFrame] = None,
    ) -> List[StressResult]:
        """Run all registered scenarios + Monte Carlo if returns provided."""
        results = []

        for scenario in self.scenarios:
            result = self.run_historical_scenario(weights, portfolio_value, scenario)
            results.append(result)

        if returns is not None and not returns.empty:
            mc_result = self.monte_carlo_simulation(
                returns, weights, portfolio_value
            )
            results.append(mc_result)

        return sorted(results, key=lambda r: r.portfolio_loss)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    def generate_report(self, results: List[StressResult]) -> Dict:
        """Generate a summary stress test report."""
        if not results:
            return {"status": "no_results"}

        worst = min(results, key=lambda r: r.portfolio_loss)
        avg_loss = np.mean([r.portfolio_loss for r in results])
        avg_var95 = np.mean([r.var_95 for r in results])

        return {
            "total_scenarios": len(results),
            "worst_scenario": worst.scenario_name,
            "worst_loss": worst.portfolio_loss,
            "worst_loss_pct": worst.portfolio_loss_pct,
            "average_loss": float(avg_loss),
            "average_var_95": float(avg_var95),
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "loss": r.portfolio_loss,
                    "loss_pct": r.portfolio_loss_pct,
                    "var_95": r.var_95,
                    "cvar_95": r.cvar_95,
                    "max_drawdown": r.max_drawdown,
                }
                for r in results
            ],
        }
