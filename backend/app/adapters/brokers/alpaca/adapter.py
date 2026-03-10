"""Alpaca broker adapter for paper and live trading.

Implements the broker interface for order submission, position management,
and account monitoring via the Alpaca Trade API.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.brokers.alpaca")


class AlpacaBrokerAdapter:
    """Alpaca broker adapter for paper trading (primary) and live trading."""

    def __init__(self):
        self.api_key = settings.alpaca_api_key
        self.secret_key = settings.alpaca_secret_key
        self.base_url = settings.alpaca_base_url
        self.paper_mode = settings.alpaca_paper_mode
        self._api = None

    async def connect(self) -> bool:
        """Initialize Alpaca API connection."""
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API credentials not configured")
            return False
        try:
            import alpaca_trade_api as tradeapi
            self._api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
            )
            account = self._api.get_account()
            logger.info(
                "Alpaca connection verified",
                account_status=account.status,
                buying_power=account.buying_power,
                paper_mode=self.paper_mode,
            )
            return True
        except Exception as e:
            logger.error("Alpaca connection failed", error=str(e))
            return False

    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        if not self._api:
            await self.connect()
        if not self._api:
            return {"error": "Not connected"}
        try:
            account = self._api.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "paper_mode": self.paper_mode,
            }
        except Exception as e:
            logger.error("Get account error", error=str(e))
            return {"error": str(e)}

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit an order to Alpaca."""
        if not self._api:
            await self.connect()
        if not self._api:
            return {"error": "Not connected"}
        try:
            kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }
            if limit_price and order_type == "limit":
                kwargs["limit_price"] = limit_price

            order = self._api.submit_order(**kwargs)
            logger.info(
                "Order submitted",
                order_id=order.id,
                symbol=symbol,
                side=side,
                qty=qty,
            )
            return {
                "order_id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": str(order.qty),
                "type": order.type,
                "status": order.status,
                "submitted_at": str(order.submitted_at),
            }
        except Exception as e:
            logger.error("Submit order error", symbol=symbol, error=str(e))
            return {"error": str(e)}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        if not self._api:
            return {"error": "Not connected"}
        try:
            self._api.cancel_order(order_id)
            logger.info("Order cancelled", order_id=order_id)
            return {"order_id": order_id, "status": "cancelled"}
        except Exception as e:
            logger.error("Cancel order error", order_id=order_id, error=str(e))
            return {"error": str(e)}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        if not self._api:
            await self.connect()
        if not self._api:
            return []
        try:
            positions = self._api.list_positions()
            return [{
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
            } for p in positions]
        except Exception as e:
            logger.error("Get positions error", error=str(e))
            return []

    async def get_orders(self, status: str = "open") -> List[Dict[str, Any]]:
        """Get orders by status."""
        if not self._api:
            await self.connect()
        if not self._api:
            return []
        try:
            orders = self._api.list_orders(status=status)
            return [{
                "order_id": o.id,
                "symbol": o.symbol,
                "side": o.side,
                "qty": str(o.qty),
                "type": o.type,
                "status": o.status,
                "submitted_at": str(o.submitted_at),
                "filled_at": str(o.filled_at) if o.filled_at else None,
                "filled_qty": str(o.filled_qty) if o.filled_qty else "0",
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
            } for o in orders]
        except Exception as e:
            logger.error("Get orders error", error=str(e))
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check Alpaca API connectivity and account status."""
        try:
            account = await self.get_account()
            return {
                "broker": "alpaca",
                "status": "healthy" if "error" not in account else "unhealthy",
                "paper_mode": self.paper_mode,
                "account_status": account.get("status", "unknown"),
                "last_check": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"broker": "alpaca", "status": "unhealthy", "error": str(e)}
