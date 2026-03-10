"""Unit tests for health endpoint."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test that health endpoint returns correct structure."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data
        assert "services" in data


@pytest.mark.asyncio
async def test_strategies_list_endpoint():
    """Test that strategies list endpoint returns correct structure."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert "total" in data
        assert "filters_applied" in data


@pytest.mark.asyncio
async def test_instruments_list_endpoint():
    """Test that instruments list endpoint returns correct structure."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/instruments")
        assert response.status_code == 200
        data = response.json()
        assert "instruments" in data
        assert "total" in data


@pytest.mark.asyncio
async def test_data_health_endpoint():
    """Test that data health endpoint returns all providers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/market/health")
        assert response.status_code == 200
        data = response.json()
        assert "feeds" in data
        expected_providers = [
            "yahoo_finance", "fred", "fmp", "polygon",
            "reddit", "trading_economics", "b3_anbima", "fiscal_ai"
        ]
        for provider in expected_providers:
            assert provider in data["feeds"]


@pytest.mark.asyncio
async def test_ingestion_status_endpoint():
    """Test that ingestion status endpoint returns all providers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/data/status")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
