-- =============================================================================
-- 151 Trading System - Database Initialization
-- Creates all schemas and tables as specified in D5 Section 7
-- =============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- SCHEMA: ref (Reference Data)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS ref;

CREATE TABLE ref.instruments (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    asset_class     TEXT NOT NULL,  -- equity, etf, option, future, fx, crypto, bond, index
    subtype         TEXT,
    venue           TEXT,
    currency        TEXT NOT NULL DEFAULT 'USD',
    timezone        TEXT NOT NULL DEFAULT 'America/New_York',
    issuer_id       BIGINT,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(symbol, venue, asset_class)
);

CREATE TABLE ref.instrument_identifiers (
    instrument_id   BIGINT NOT NULL REFERENCES ref.instruments(id),
    id_type         TEXT NOT NULL,  -- isin, cusip, sedol, figi, ticker
    id_value        TEXT NOT NULL,
    valid_from      TIMESTAMPTZ,
    valid_to        TIMESTAMPTZ,
    PRIMARY KEY (instrument_id, id_type, id_value)
);

CREATE TABLE ref.calendars (
    calendar_id     TEXT NOT NULL,
    trade_date      DATE NOT NULL,
    open_ts         TIMESTAMPTZ,
    close_ts        TIMESTAMPTZ,
    session_type    TEXT NOT NULL DEFAULT 'regular',  -- regular, early_close, holiday
    PRIMARY KEY (calendar_id, trade_date)
);

CREATE TABLE ref.corporate_actions (
    action_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instrument_id   BIGINT NOT NULL REFERENCES ref.instruments(id),
    action_type     TEXT NOT NULL,  -- split, dividend, spinoff, merger
    ex_date         DATE NOT NULL,
    record_date     DATE,
    pay_date        DATE,
    factor          NUMERIC,
    details         JSONB,
    source          TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE ref.universes (
    universe_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL UNIQUE,
    asset_class     TEXT NOT NULL,
    rules           JSONB,
    version         INTEGER NOT NULL DEFAULT 1,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- SCHEMA: md (Market Data - Live and Historical)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS md;

CREATE TABLE md.bars_1d (
    instrument_id   BIGINT NOT NULL REFERENCES ref.instruments(id),
    ts              DATE NOT NULL,
    open            NUMERIC,
    high            NUMERIC,
    low             NUMERIC,
    close           NUMERIC,
    volume          NUMERIC,
    adj_factor      NUMERIC NOT NULL DEFAULT 1.0,
    source          TEXT NOT NULL,
    received_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (instrument_id, ts)
);

CREATE TABLE md.bars_1m (
    instrument_id   BIGINT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC,
    high            NUMERIC,
    low             NUMERIC,
    close           NUMERIC,
    volume          NUMERIC,
    source          TEXT NOT NULL,
    PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('md.bars_1m', 'ts', if_not_exists => TRUE);

CREATE TABLE md.quotes (
    instrument_id   BIGINT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    bid             NUMERIC,
    ask             NUMERIC,
    bid_size        NUMERIC,
    ask_size        NUMERIC,
    source          TEXT NOT NULL,
    PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('md.quotes', 'ts', if_not_exists => TRUE);

CREATE TABLE md.options_contracts (
    option_id       BIGSERIAL PRIMARY KEY,
    underlying_id   BIGINT NOT NULL REFERENCES ref.instruments(id),
    expiry          DATE NOT NULL,
    strike          NUMERIC NOT NULL,
    right           TEXT NOT NULL,  -- call, put
    multiplier      NUMERIC NOT NULL DEFAULT 100,
    style           TEXT NOT NULL DEFAULT 'american',
    venue           TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE md.options_eod (
    option_id       BIGINT NOT NULL REFERENCES md.options_contracts(option_id),
    trade_date      DATE NOT NULL,
    bid             NUMERIC,
    ask             NUMERIC,
    last            NUMERIC,
    volume          NUMERIC,
    open_interest   NUMERIC,
    iv              NUMERIC,
    delta           NUMERIC,
    gamma           NUMERIC,
    vega            NUMERIC,
    theta           NUMERIC,
    source          TEXT NOT NULL,
    PRIMARY KEY (option_id, trade_date)
);

CREATE TABLE md.futures_contracts (
    fut_id          BIGSERIAL PRIMARY KEY,
    root            TEXT NOT NULL,
    expiry          DATE NOT NULL,
    contract_month  TEXT NOT NULL,
    multiplier      NUMERIC NOT NULL,
    venue           TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE md.curve_points (
    curve_id        TEXT NOT NULL,
    ts              DATE NOT NULL,
    tenor           TEXT NOT NULL,
    rate            NUMERIC NOT NULL,
    source          TEXT NOT NULL,
    PRIMARY KEY (curve_id, ts, tenor)
);

CREATE TABLE md.macro_series (
    series_id       TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,
    name            TEXT NOT NULL,
    region          TEXT,
    frequency       TEXT,
    units           TEXT
);

CREATE TABLE md.macro_observations (
    series_id       TEXT NOT NULL REFERENCES md.macro_series(series_id),
    obs_date        DATE NOT NULL,
    value           NUMERIC,
    vintage_date    DATE,
    released_at     TIMESTAMPTZ,
    received_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (series_id, obs_date, vintage_date)
);

-- =============================================================================
-- SCHEMA: fa (Fundamentals and Text)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS fa;

CREATE TABLE fa.filings (
    filing_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    issuer_id       BIGINT NOT NULL,
    form_type       TEXT NOT NULL,
    filed_at        TIMESTAMPTZ,
    accepted_at     TIMESTAMPTZ,
    period_end      DATE,
    accession_no    TEXT,
    source          TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE fa.facts (
    issuer_id       BIGINT NOT NULL,
    fact_name       TEXT NOT NULL,
    fact_value      NUMERIC,
    fact_unit       TEXT,
    period_end      DATE NOT NULL,
    filed_at        TIMESTAMPTZ,
    accepted_at     TIMESTAMPTZ,
    source          TEXT NOT NULL,
    PRIMARY KEY (issuer_id, fact_name, period_end, accepted_at)
);

CREATE TABLE fa.text_events (
    event_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    issuer_id       BIGINT,
    event_type      TEXT NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL,
    source          TEXT NOT NULL,
    title           TEXT,
    body_ref        TEXT,
    language        TEXT DEFAULT 'en'
);

CREATE TABLE fa.sentiment_scores (
    event_id        UUID PRIMARY KEY REFERENCES fa.text_events(event_id),
    issuer_id       BIGINT,
    published_at    TIMESTAMPTZ NOT NULL,
    sentiment_score NUMERIC,
    relevance_score NUMERIC,
    novelty_score   NUMERIC,
    model_version   TEXT
);

-- =============================================================================
-- SCHEMA: strat (Strategy Registry)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS strat;

CREATE TABLE strat.strategies (
    strategy_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code            TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    source_book     TEXT NOT NULL,  -- 151TS, FA
    asset_class     TEXT NOT NULL,
    style           TEXT,
    sub_style       TEXT,
    horizon         TEXT,
    directionality  TEXT,
    complexity      TEXT,
    status          TEXT NOT NULL DEFAULT 'research_only',
    owner           TEXT,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE strat.strategy_versions (
    version_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id     UUID NOT NULL REFERENCES strat.strategies(strategy_id),
    version_no      INTEGER NOT NULL,
    code_ref        TEXT,
    notes           TEXT,
    approved_by     TEXT,
    approved_at     TIMESTAMPTZ,
    hash            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(strategy_id, version_no)
);

CREATE TABLE strat.strategy_parameters (
    version_id      UUID NOT NULL REFERENCES strat.strategy_versions(version_id),
    param_name      TEXT NOT NULL,
    param_type      TEXT NOT NULL,
    default_value   JSONB,
    bounds          JSONB,
    required        BOOLEAN NOT NULL DEFAULT TRUE,
    description     TEXT,
    PRIMARY KEY (version_id, param_name)
);

CREATE TABLE strat.strategy_schedules (
    strategy_id     UUID NOT NULL REFERENCES strat.strategies(strategy_id),
    schedule_type   TEXT NOT NULL,
    cron_expr       TEXT,
    event_type      TEXT,
    timezone        TEXT DEFAULT 'America/New_York',
    PRIMARY KEY (strategy_id, schedule_type)
);

CREATE TABLE strat.dataset_manifests (
    manifest_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    description     TEXT,
    sources         JSONB NOT NULL,
    adjustment_policy JSONB,
    as_of           TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- SCHEMA: research (Backtesting and Alpha Research)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE research.backtest_runs (
    run_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_version_id UUID NOT NULL REFERENCES strat.strategy_versions(version_id),
    manifest_id     UUID REFERENCES strat.dataset_manifests(manifest_id),
    run_mode        TEXT NOT NULL DEFAULT 'single',  -- single, grid, walk_forward, portfolio
    status          TEXT NOT NULL DEFAULT 'queued',
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    config          JSONB,
    error_message   TEXT
);

CREATE TABLE research.backtest_metrics (
    run_id          UUID NOT NULL REFERENCES research.backtest_runs(run_id),
    metric_name     TEXT NOT NULL,
    metric_value    NUMERIC,
    PRIMARY KEY (run_id, metric_name)
);

CREATE TABLE research.backtest_timeseries (
    run_id          UUID NOT NULL REFERENCES research.backtest_runs(run_id),
    ts              DATE NOT NULL,
    pnl             NUMERIC,
    nav             NUMERIC,
    drawdown        NUMERIC,
    gross           NUMERIC,
    net             NUMERIC,
    turnover        NUMERIC,
    PRIMARY KEY (run_id, ts)
);

CREATE TABLE research.backtest_trades (
    run_id          UUID NOT NULL REFERENCES research.backtest_runs(run_id),
    trade_id        UUID NOT NULL DEFAULT uuid_generate_v4(),
    instrument_id   BIGINT NOT NULL,
    ts_trade        TIMESTAMPTZ NOT NULL,
    side            TEXT NOT NULL,
    qty             NUMERIC NOT NULL,
    price           NUMERIC NOT NULL,
    fee             NUMERIC DEFAULT 0,
    PRIMARY KEY (run_id, trade_id)
);

CREATE TABLE research.alpha_library (
    alpha_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    expression      TEXT NOT NULL,
    universe_id     UUID,
    delay           INTEGER NOT NULL DEFAULT 1,
    neutralization  TEXT,
    decay           JSONB,
    owner           TEXT,
    approved        BOOLEAN NOT NULL DEFAULT FALSE,
    sharpe          NUMERIC,
    turnover        NUMERIC,
    fitness         NUMERIC,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE research.alpha_correlations (
    alpha_id_a      UUID NOT NULL REFERENCES research.alpha_library(alpha_id),
    alpha_id_b      UUID NOT NULL REFERENCES research.alpha_library(alpha_id),
    window          TEXT NOT NULL,
    correlation     NUMERIC,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (alpha_id_a, alpha_id_b, window)
);

-- =============================================================================
-- SCHEMA: oms (Order Management System)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS oms;

CREATE TABLE oms.portfolios (
    portfolio_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL UNIQUE,
    mode            TEXT NOT NULL DEFAULT 'paper',  -- paper, live
    broker          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE oms.order_intents (
    intent_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id    UUID NOT NULL REFERENCES oms.portfolios(portfolio_id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    basket          JSONB NOT NULL,
    execution_style TEXT NOT NULL DEFAULT 'market_on_close',
    state           TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE oms.orders (
    order_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    intent_id       UUID REFERENCES oms.order_intents(intent_id),
    instrument_id   BIGINT NOT NULL,
    side            TEXT NOT NULL,  -- buy, sell
    qty             NUMERIC NOT NULL,
    order_type      TEXT NOT NULL DEFAULT 'market',
    limit_price     NUMERIC,
    tif             TEXT DEFAULT 'day',
    broker          TEXT,
    broker_order_id TEXT,
    state           TEXT NOT NULL DEFAULT 'intent_created',
    submitted_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE oms.fills (
    fill_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id        UUID NOT NULL REFERENCES oms.orders(order_id),
    instrument_id   BIGINT NOT NULL,
    ts_fill         TIMESTAMPTZ NOT NULL,
    qty             NUMERIC NOT NULL,
    price           NUMERIC NOT NULL,
    fee             NUMERIC DEFAULT 0,
    liquidity_flag  TEXT,
    broker_exec_id  TEXT
);

CREATE TABLE oms.positions (
    portfolio_id    UUID NOT NULL REFERENCES oms.portfolios(portfolio_id),
    instrument_id   BIGINT NOT NULL,
    ts_position     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    qty             NUMERIC NOT NULL DEFAULT 0,
    avg_cost        NUMERIC,
    market_value    NUMERIC,
    unrealized_pnl  NUMERIC,
    PRIMARY KEY (portfolio_id, instrument_id, ts_position)
);

CREATE TABLE oms.cash_ledger (
    portfolio_id    UUID NOT NULL REFERENCES oms.portfolios(portfolio_id),
    ts_cash         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    currency        TEXT NOT NULL DEFAULT 'USD',
    cash_amount     NUMERIC NOT NULL,
    reserved_amount NUMERIC DEFAULT 0,
    PRIMARY KEY (portfolio_id, ts_cash, currency)
);

-- =============================================================================
-- SCHEMA: risk (Risk Management)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS risk;

CREATE TABLE risk.limits (
    limit_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scope_type      TEXT NOT NULL,  -- portfolio, strategy, instrument
    scope_id        UUID NOT NULL,
    limit_name      TEXT NOT NULL,
    threshold       NUMERIC NOT NULL,
    hard            BOOLEAN NOT NULL DEFAULT TRUE,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE risk.snapshots (
    snapshot_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id    UUID NOT NULL,
    ts_snapshot     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    gross           NUMERIC,
    net             NUMERIC,
    var_1d          NUMERIC,
    cvar_1d         NUMERIC,
    max_drawdown    NUMERIC,
    margin_used     NUMERIC,
    data            JSONB
);

CREATE TABLE risk.factor_exposures (
    snapshot_id     UUID NOT NULL REFERENCES risk.snapshots(snapshot_id),
    factor_name     TEXT NOT NULL,
    exposure        NUMERIC,
    PRIMARY KEY (snapshot_id, factor_name)
);

CREATE TABLE risk.greeks (
    snapshot_id     UUID NOT NULL REFERENCES risk.snapshots(snapshot_id),
    portfolio_id    UUID NOT NULL,
    delta           NUMERIC,
    gamma           NUMERIC,
    vega            NUMERIC,
    theta           NUMERIC,
    dv01            NUMERIC,
    PRIMARY KEY (snapshot_id, portfolio_id)
);

CREATE TABLE risk.stress_runs (
    run_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    requested_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scenario_set    JSONB,
    status          TEXT NOT NULL DEFAULT 'queued'
);

CREATE TABLE risk.stress_results (
    run_id          UUID NOT NULL REFERENCES risk.stress_runs(run_id),
    scenario_name   TEXT NOT NULL,
    pnl_shock       NUMERIC,
    gross_after     NUMERIC,
    net_after       NUMERIC,
    PRIMARY KEY (run_id, scenario_name)
);

-- =============================================================================
-- SCHEMA: audit (Audit Trail)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS audit;

CREATE TABLE audit.events (
    event_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_event        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor           TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    action          TEXT NOT NULL,
    payload         JSONB,
    request_id      TEXT
);

CREATE TABLE audit.data_incidents (
    incident_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_opened       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    severity        TEXT NOT NULL DEFAULT 'warning',
    source          TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    description     TEXT,
    state           TEXT NOT NULL DEFAULT 'open',
    resolved_at     TIMESTAMPTZ,
    resolution_note TEXT
);

-- =============================================================================
-- Indexes for performance
-- =============================================================================
CREATE INDEX idx_instruments_symbol ON ref.instruments(symbol);
CREATE INDEX idx_instruments_asset_class ON ref.instruments(asset_class);
CREATE INDEX idx_bars_1d_ts ON md.bars_1d(ts);
CREATE INDEX idx_bars_1d_instrument ON md.bars_1d(instrument_id);
CREATE INDEX idx_options_underlying ON md.options_contracts(underlying_id);
CREATE INDEX idx_facts_issuer ON fa.facts(issuer_id);
CREATE INDEX idx_facts_period ON fa.facts(period_end);
CREATE INDEX idx_strategies_code ON strat.strategies(code);
CREATE INDEX idx_strategies_status ON strat.strategies(status);
CREATE INDEX idx_strategies_asset_class ON strat.strategies(asset_class);
CREATE INDEX idx_backtest_runs_strategy ON research.backtest_runs(strategy_version_id);
CREATE INDEX idx_backtest_runs_status ON research.backtest_runs(status);
CREATE INDEX idx_orders_state ON oms.orders(state);
CREATE INDEX idx_orders_intent ON oms.orders(intent_id);
CREATE INDEX idx_positions_portfolio ON oms.positions(portfolio_id);
CREATE INDEX idx_audit_events_entity ON audit.events(entity_type, entity_id);
CREATE INDEX idx_audit_events_ts ON audit.events(ts_event);
CREATE INDEX idx_data_incidents_state ON audit.data_incidents(state);

-- =============================================================================
-- Insert default portfolio for paper trading
-- =============================================================================
INSERT INTO oms.portfolios (name, mode, broker) VALUES ('default_paper', 'paper', 'alpaca');
