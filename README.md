# 151 Trading System

A comprehensive research and paper trading platform implementing 251 trading strategies across 10 asset classes, based on the 151 Trading Strategies book and supplementary Financial Alchemy strategies.

## Architecture

The system follows a modular, event-driven architecture with clear separation between research, execution, and monitoring layers.

| Component | Technology | Purpose |
|---|---|---|
| Backend API | FastAPI + Python 3.11 | REST API, strategy orchestration, data ingestion |
| Database | PostgreSQL 16 + TimescaleDB | Market data, fundamentals, strategy registry, OMS |
| Task Queue | Celery + Redis | Async backtests, data ingestion, scheduled jobs |
| Frontend | Next.js + TypeScript + TailwindCSS | Strategy dashboard, backtest lab, risk monitor |
| Monitoring | Prometheus + Grafana | Metrics, alerting, data health |
| Broker | Alpaca (paper trading) | Order execution, position management |

## Data Providers

| Provider | Data Type | Status |
|---|---|---|
| Yahoo Finance | Equities, ETFs, indices OHLCV | Active |
| FRED | Macro-economic series, yield curves | Active |
| FMP | Fundamentals, financials, ratios | Active |
| Polygon.io | Real-time quotes, options chains | Active |
| Reddit | Sentiment from financial subreddits | Active |
| Trading Economics | Economic calendar, global indicators | Active |
| B3 / Anbima | Brazilian equities, fixed income | Active |
| Fiscal AI | As-reported financials, KPIs | Active |

## Database Schemas

The database uses 8 schemas: `ref` (reference data), `md` (market data), `fa` (fundamentals), `strat` (strategy registry), `research` (backtesting), `oms` (order management), `risk` (risk management), and `audit` (audit trail).

## Quick Start

```bash
# Clone repository
git clone https://github.com/mcauduro0/151Trading.git
cd 151Trading

# Copy environment config
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Run tests
cd backend && pytest tests/ -v
```

## Project Structure

```
151Trading/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/v1/         # REST endpoints
│   │   ├── core/           # Config, database, logging
│   │   ├── adapters/       # Data providers & brokers
│   │   ├── services/       # Business logic
│   │   └── workers/        # Celery tasks
│   └── tests/
├── frontend/               # Next.js frontend
├── strategies/             # Strategy implementations
│   ├── base.py            # Abstract base class
│   ├── equity/            # Equity strategies
│   ├── options/           # Options strategies
│   └── ...                # Other asset classes
├── shared/schemas/         # Shared data schemas
├── data/                   # Data storage
├── infra/                  # Docker, monitoring
├── research/               # Alpha research
├── docs/                   # Documentation
└── .github/workflows/      # CI/CD
```

## Development Plan

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the complete 26-week sprint plan.

## License

Private - All rights reserved.
