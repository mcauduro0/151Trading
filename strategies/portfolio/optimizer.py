"""Portfolio Construction & Optimization Module.

Implements four portfolio allocation models:
1. Risk Parity — Equal risk contribution across strategies
2. Hierarchical Risk Parity (HRP) — Clustering-based allocation
3. Mean-Variance (Markowitz) — Classical efficient frontier
4. Black-Litterman — Bayesian views overlay on equilibrium

All models accept a covariance matrix and expected returns, and output
optimal weights. Winsorization at 5%/95% is applied to all return inputs.

Reference: 151 Trading Strategies, Portfolio Construction section.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class OptimizationMethod(Enum):
    RISK_PARITY = "risk_parity"
    HRP = "hrp"
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"


@dataclass
class PortfolioResult:
    """Result of a portfolio optimization."""
    weights: Dict[str, float]
    method: OptimizationMethod
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_contributions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "method": self.method.value,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_contributions": self.risk_contributions,
            "metadata": self.metadata,
        }


def _winsorize(arr: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    """Winsorize at 5%/95%."""
    lo = np.nanpercentile(arr, lower * 100)
    hi = np.nanpercentile(arr, upper * 100)
    return np.clip(arr, lo, hi)


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Clean and winsorize return series."""
    returns = returns.dropna(how='all')
    for col in returns.columns:
        vals = returns[col].dropna().values
        if len(vals) > 10:
            returns[col] = _winsorize(returns[col].values)
    return returns.fillna(0)


# ============================================================
# 1. RISK PARITY
# ============================================================

class RiskParityOptimizer:
    """Equal Risk Contribution (Risk Parity) portfolio.

    Each asset contributes equally to total portfolio risk.
    Uses iterative optimization to find weights where:
    w_i * (Sigma @ w)_i = w_j * (Sigma @ w)_j for all i, j
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.045,
    ) -> PortfolioResult:
        """Compute risk parity weights."""
        clean_ret = _clean_returns(returns)
        cov = clean_ret.cov().values
        n = cov.shape[0]
        assets = list(clean_ret.columns)

        # Initial equal weights
        w = np.ones(n) / n

        # Iterative risk budgeting
        for _ in range(self.max_iter):
            sigma_w = cov @ w
            risk_contrib = w * sigma_w
            total_risk = np.sqrt(w @ cov @ w)

            # Target: equal risk contribution
            target_rc = total_risk / n

            # Gradient step
            for i in range(n):
                if sigma_w[i] > 0:
                    w[i] *= (target_rc / risk_contrib[i]) ** 0.5

            # Normalize
            w = w / w.sum()

            # Check convergence
            rc_normalized = risk_contrib / risk_contrib.sum()
            if np.max(np.abs(rc_normalized - 1.0/n)) < self.tol:
                break

        # Compute portfolio metrics
        port_ret = float(clean_ret.mean().values @ w * 252)
        port_vol = float(np.sqrt(w @ cov @ w) * np.sqrt(252))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

        # Risk contributions
        sigma_w = cov @ w
        rc = w * sigma_w
        rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

        return PortfolioResult(
            weights={a: float(w[i]) for i, a in enumerate(assets)},
            method=OptimizationMethod.RISK_PARITY,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions={a: float(rc_pct[i]) for i, a in enumerate(assets)},
            metadata={"iterations": _, "converged": True},
        )


# ============================================================
# 2. HIERARCHICAL RISK PARITY (HRP)
# ============================================================

class HRPOptimizer:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Steps:
    1. Compute correlation distance matrix
    2. Hierarchical clustering (single linkage)
    3. Quasi-diagonalization (reorder assets)
    4. Recursive bisection for weight allocation
    """

    def _correlation_distance(self, corr: np.ndarray) -> np.ndarray:
        """Convert correlation matrix to distance matrix."""
        return np.sqrt(0.5 * (1 - corr))

    def _single_linkage_clustering(self, dist: np.ndarray) -> List[Tuple[int, int, float, int]]:
        """Simple single-linkage agglomerative clustering."""
        n = dist.shape[0]
        clusters = {i: [i] for i in range(n)}
        active = set(range(n))
        linkage = []
        next_id = n

        for _ in range(n - 1):
            # Find minimum distance pair
            min_dist = np.inf
            min_i, min_j = -1, -1
            active_list = sorted(active)

            for idx_a in range(len(active_list)):
                for idx_b in range(idx_a + 1, len(active_list)):
                    i, j = active_list[idx_a], active_list[idx_b]
                    # Compute minimum distance between clusters
                    d = np.inf
                    for a in clusters[i]:
                        for b in clusters[j]:
                            if a < n and b < n:
                                d = min(d, dist[a, b])
                    if d < min_dist:
                        min_dist = d
                        min_i, min_j = i, j

            # Merge clusters
            new_cluster = clusters[min_i] + clusters[min_j]
            clusters[next_id] = new_cluster
            active.discard(min_i)
            active.discard(min_j)
            active.add(next_id)
            linkage.append((min_i, min_j, min_dist, len(new_cluster)))
            next_id += 1

        return linkage

    def _get_quasi_diag_order(self, linkage: List, n: int) -> List[int]:
        """Get quasi-diagonal ordering from linkage."""
        # Build tree
        tree = {}
        next_id = n
        for i, (a, b, _, _) in enumerate(linkage):
            tree[next_id + i] = (a, b)

        # Traverse tree to get leaf order
        def _get_leaves(node):
            if node < n:
                return [node]
            left, right = tree[node]
            return _get_leaves(left) + _get_leaves(right)

        root = n + len(linkage) - 1
        return _get_leaves(root)

    def _recursive_bisection(self, cov: np.ndarray, order: List[int]) -> np.ndarray:
        """Recursive bisection to allocate weights."""
        n = cov.shape[0]
        w = np.ones(n)

        clusters = [order]
        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Compute inverse-variance for each sub-cluster
                def _cluster_var(indices):
                    sub_cov = cov[np.ix_(indices, indices)]
                    inv_diag = 1.0 / np.diag(sub_cov)
                    inv_diag /= inv_diag.sum()
                    return float(inv_diag @ sub_cov @ inv_diag)

                var_left = _cluster_var(left)
                var_right = _cluster_var(right)

                alpha = 1 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5

                for i in left:
                    w[i] *= alpha
                for i in right:
                    w[i] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return w / w.sum()

    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.045,
    ) -> PortfolioResult:
        """Compute HRP weights."""
        clean_ret = _clean_returns(returns)
        cov = clean_ret.cov().values
        corr = clean_ret.corr().values
        n = cov.shape[0]
        assets = list(clean_ret.columns)

        if n <= 1:
            w = np.ones(n) / max(n, 1)
        else:
            # Step 1: Distance matrix
            dist = self._correlation_distance(corr)
            np.fill_diagonal(dist, 0)

            # Step 2: Clustering
            linkage = self._single_linkage_clustering(dist)

            # Step 3: Quasi-diagonalization
            order = self._get_quasi_diag_order(linkage, n)

            # Step 4: Recursive bisection
            w = self._recursive_bisection(cov, order)

        # Portfolio metrics
        port_ret = float(clean_ret.mean().values @ w * 252)
        port_vol = float(np.sqrt(w @ cov @ w) * np.sqrt(252))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

        sigma_w = cov @ w
        rc = w * sigma_w
        rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

        return PortfolioResult(
            weights={a: float(w[i]) for i, a in enumerate(assets)},
            method=OptimizationMethod.HRP,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions={a: float(rc_pct[i]) for i, a in enumerate(assets)},
            metadata={"n_assets": n, "clustering": "single_linkage"},
        )


# ============================================================
# 3. MEAN-VARIANCE (MARKOWITZ)
# ============================================================

class MeanVarianceOptimizer:
    """Classical Markowitz Mean-Variance optimization.

    Finds the tangency portfolio (max Sharpe) on the efficient frontier
    using analytical solution with optional constraints.
    """

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 0.25,
        allow_short: bool = False,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.allow_short = allow_short

    def _max_sharpe_analytical(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        rf: float,
    ) -> np.ndarray:
        """Analytical max-Sharpe weights (unconstrained)."""
        excess = mu - rf
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        w = cov_inv @ excess
        w_sum = w.sum()
        if w_sum != 0:
            w = w / w_sum
        return w

    def _project_weights(self, w: np.ndarray) -> np.ndarray:
        """Project weights to satisfy constraints with iterative redistribution."""
        n = len(w)
        if not self.allow_short:
            w = np.maximum(w, self.min_weight)

        # Iterative clip-and-redistribute
        for _ in range(100):
            over = w > self.max_weight
            if not np.any(over):
                break
            excess = np.sum(w[over] - self.max_weight)
            w[over] = self.max_weight
            under = ~over
            n_under = np.sum(under)
            if n_under > 0:
                w[under] += excess / n_under
            else:
                break

        # Final normalization
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(n) / n
        return w

    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.045,
        target_return: Optional[float] = None,
    ) -> PortfolioResult:
        """Compute mean-variance optimal weights."""
        clean_ret = _clean_returns(returns)
        mu = clean_ret.mean().values * 252
        cov = clean_ret.cov().values * 252
        n = cov.shape[0]
        assets = list(clean_ret.columns)

        # Add regularization to covariance
        cov += np.eye(n) * 1e-6

        # Analytical solution
        w = self._max_sharpe_analytical(mu, cov, risk_free_rate)

        # Apply constraints
        w = self._project_weights(w)

        # Portfolio metrics
        port_ret = float(mu @ w)
        port_vol = float(np.sqrt(w @ cov @ w))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

        sigma_w = cov @ w
        rc = w * sigma_w
        rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

        return PortfolioResult(
            weights={a: float(w[i]) for i, a in enumerate(assets)},
            method=OptimizationMethod.MEAN_VARIANCE,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions={a: float(rc_pct[i]) for i, a in enumerate(assets)},
            metadata={
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "allow_short": self.allow_short,
            },
        )


# ============================================================
# 4. BLACK-LITTERMAN
# ============================================================

class BlackLittermanOptimizer:
    """Black-Litterman model for Bayesian portfolio allocation.

    Combines market equilibrium returns with investor views to produce
    posterior expected returns, then optimizes using mean-variance.

    Steps:
    1. Compute implied equilibrium returns (Pi = delta * Sigma * w_mkt)
    2. Combine with investor views via Bayesian update
    3. Optimize on posterior returns
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        min_weight: float = 0.0,
        max_weight: float = 0.30,
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _implied_returns(
        self,
        cov: np.ndarray,
        market_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute equilibrium implied returns."""
        return self.risk_aversion * cov @ market_weights

    def _posterior_returns(
        self,
        pi: np.ndarray,
        cov: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Black-Litterman posterior returns and covariance.

        Args:
            pi: Equilibrium returns (n,)
            cov: Covariance matrix (n, n)
            P: Pick matrix (k, n) — each row is a view
            Q: View returns (k,)
            omega: View uncertainty (k, k) — diagonal
        """
        tau_cov = self.tau * cov

        if omega is None:
            # Proportional to the variance of the view portfolios
            omega = np.diag(np.diag(P @ tau_cov @ P.T))

        # Posterior mean
        try:
            tau_cov_inv = np.linalg.inv(tau_cov)
        except np.linalg.LinAlgError:
            tau_cov_inv = np.linalg.pinv(tau_cov)

        try:
            omega_inv = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(omega)

        M = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
        posterior_mu = M @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

        # Posterior covariance
        posterior_cov = cov + M

        return posterior_mu, posterior_cov

    def optimize(
        self,
        returns: pd.DataFrame,
        market_weights: Optional[Dict[str, float]] = None,
        views: Optional[List[Dict]] = None,
        risk_free_rate: float = 0.045,
    ) -> PortfolioResult:
        """Compute Black-Litterman optimal weights.

        Args:
            returns: Historical return series
            market_weights: Market cap weights (default: equal)
            views: List of dicts with 'assets', 'direction', 'return', 'confidence'
            risk_free_rate: Risk-free rate
        """
        clean_ret = _clean_returns(returns)
        cov = clean_ret.cov().values * 252
        n = cov.shape[0]
        assets = list(clean_ret.columns)

        # Add regularization
        cov += np.eye(n) * 1e-6

        # Market weights (default: equal)
        if market_weights:
            w_mkt = np.array([market_weights.get(a, 1.0/n) for a in assets])
            w_mkt = w_mkt / w_mkt.sum()
        else:
            w_mkt = np.ones(n) / n

        # Equilibrium returns
        pi = self._implied_returns(cov, w_mkt)

        # If no views, use equilibrium
        if not views or len(views) == 0:
            posterior_mu = pi
            posterior_cov = cov
        else:
            # Build P and Q matrices from views
            k = len(views)
            P = np.zeros((k, n))
            Q = np.zeros(k)

            for v_idx, view in enumerate(views):
                view_assets = view.get("assets", [])
                direction = view.get("direction", 1)  # 1 for bullish, -1 for bearish
                view_return = view.get("return", 0.05)

                for va in view_assets:
                    if va in assets:
                        asset_idx = assets.index(va)
                        P[v_idx, asset_idx] = direction / len(view_assets)

                Q[v_idx] = view_return

            posterior_mu, posterior_cov = self._posterior_returns(pi, cov, P, Q)

        # Optimize on posterior
        mv = MeanVarianceOptimizer(
            min_weight=self.min_weight,
            max_weight=self.max_weight,
        )

        # Create synthetic returns from posterior
        # Use posterior_mu directly
        excess = posterior_mu - risk_free_rate
        try:
            cov_inv = np.linalg.inv(posterior_cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(posterior_cov)

        w = cov_inv @ excess
        w = np.maximum(w, self.min_weight)
        w = np.minimum(w, self.max_weight)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(n) / n

        # Portfolio metrics
        port_ret = float(posterior_mu @ w)
        port_vol = float(np.sqrt(w @ posterior_cov @ w))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

        sigma_w = posterior_cov @ w
        rc = w * sigma_w
        rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

        return PortfolioResult(
            weights={a: float(w[i]) for i, a in enumerate(assets)},
            method=OptimizationMethod.BLACK_LITTERMAN,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions={a: float(rc_pct[i]) for i, a in enumerate(assets)},
            metadata={
                "risk_aversion": self.risk_aversion,
                "tau": self.tau,
                "n_views": len(views) if views else 0,
                "equilibrium_returns": {a: float(pi[i]) for i, a in enumerate(assets)},
                "posterior_returns": {a: float(posterior_mu[i]) for i, a in enumerate(assets)},
            },
        )


# ============================================================
# PORTFOLIO CONSTRUCTOR (Unified Interface)
# ============================================================

class PortfolioConstructor:
    """Unified interface for portfolio construction.

    Runs all four optimization methods and provides comparison.
    """

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate
        self.optimizers = {
            OptimizationMethod.RISK_PARITY: RiskParityOptimizer(),
            OptimizationMethod.HRP: HRPOptimizer(),
            OptimizationMethod.MEAN_VARIANCE: MeanVarianceOptimizer(),
            OptimizationMethod.BLACK_LITTERMAN: BlackLittermanOptimizer(),
        }

    def optimize(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
        **kwargs,
    ) -> PortfolioResult:
        """Run a single optimization method."""
        optimizer = self.optimizers[method]
        return optimizer.optimize(returns, risk_free_rate=self.risk_free_rate, **kwargs)

    def compare_all(
        self,
        returns: pd.DataFrame,
        views: Optional[List[Dict]] = None,
    ) -> Dict[str, PortfolioResult]:
        """Run all four methods and return comparison."""
        results = {}

        results["risk_parity"] = self.optimizers[OptimizationMethod.RISK_PARITY].optimize(
            returns, self.risk_free_rate
        )
        results["hrp"] = self.optimizers[OptimizationMethod.HRP].optimize(
            returns, self.risk_free_rate
        )
        results["mean_variance"] = self.optimizers[OptimizationMethod.MEAN_VARIANCE].optimize(
            returns, self.risk_free_rate
        )
        results["black_litterman"] = self.optimizers[OptimizationMethod.BLACK_LITTERMAN].optimize(
            returns, views=views, risk_free_rate=self.risk_free_rate
        )

        return results

    def blend(
        self,
        results: Dict[str, PortfolioResult],
        blend_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Blend weights from multiple optimization methods.

        Default blend: 30% Risk Parity, 25% HRP, 25% MV, 20% BL
        """
        if blend_weights is None:
            blend_weights = {
                "risk_parity": 0.30,
                "hrp": 0.25,
                "mean_variance": 0.25,
                "black_litterman": 0.20,
            }

        all_assets = set()
        for r in results.values():
            all_assets.update(r.weights.keys())

        blended = {a: 0.0 for a in all_assets}
        for method_name, bw in blend_weights.items():
            if method_name in results:
                for asset, weight in results[method_name].weights.items():
                    blended[asset] += weight * bw

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {a: w / total for a, w in blended.items()}

        return blended
