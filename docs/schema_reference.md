# Database Schema Reference (from D5)

## Schemas: ref, md, fa, strat, research, oms, risk, audit

### ref schema
- ref.instruments: id, symbol, asset_class, subtype, venue, currency, timezone, issuer_id, active
- ref.instrument_identifiers: instrument_id, id_type, id_value, valid_from, valid_to
- ref.calendars: calendar_id, trade_date, open_ts, close_ts, session_type
- ref.corporate_actions: action_id uuid, instrument_id, action_type, ex_date, record_date, pay_date, factor, source
- ref.universes: universe_id uuid, name, asset_class, rules jsonb, version, active

### md schema
- md.bars_1d: instrument_id, ts date, OHLCV, adj_factor, source, received_at
- md.bars_1m: instrument_id, ts timestamptz, OHLCV, source (TimescaleDB hypertable)
- md.quotes: instrument_id, ts timestamptz, bid, ask, bid_size, ask_size, source (hypertable)
- md.options_contracts: option_id, underlying_id, expiry, strike, right, multiplier, style, venue, active
- md.options_eod: option_id, trade_date, bid, ask, last, volume, OI, iv, delta, gamma, vega, theta, source
- md.futures_contracts: fut_id, root, expiry, contract_month, multiplier, venue, active
- md.curve_points: curve_id, ts date, tenor, rate, source
- md.macro_series: series_id, provider, name, region, frequency, units
- md.macro_observations: series_id, obs_date, value, vintage_date, released_at, received_at

### fa schema
- fa.filings: filing_id uuid, issuer_id, form_type, filed_at, accepted_at, period_end, accession_no, source
- fa.facts: issuer_id, fact_name, fact_value, fact_unit, period_end, filed_at, accepted_at, source
- fa.text_events: event_id uuid, issuer_id, event_type, published_at, source, title, body_ref, language
- fa.sentiment_scores: event_id uuid, issuer_id, published_at, sentiment_score, relevance_score, novelty_score, model_version

### strat schema
- strat.strategies: strategy_id uuid, code, name, source_book, asset_class, style, status, owner
- strat.strategy_versions: version_id uuid, strategy_id, version_no, code_ref, notes, approved_by, approved_at, hash
- strat.strategy_parameters: version_id, param_name, param_type, default_value jsonb, bounds jsonb, required
- strat.strategy_schedules: strategy_id, schedule_type, cron_expr, event_type, timezone
- strat.dataset_manifests: manifest_id uuid, description, sources jsonb, adjustment_policy jsonb, as_of, created_at

### research schema
- research.backtest_runs: run_id uuid, strategy_version_id, manifest_id, run_mode, status, submitted_at, completed_at, config jsonb
- research.backtest_metrics: run_id, metric_name, metric_value
- research.backtest_timeseries: run_id, ts date, pnl, nav, drawdown, gross, net, turnover
- research.backtest_trades: run_id, trade_id uuid, instrument_id, ts_trade, side, qty, price, fee
- research.alpha_library: alpha_id uuid, expression, universe_id, delay, neutralization, decay jsonb, owner, approved
- research.alpha_correlations: alpha_id_a, alpha_id_b, window, correlation, computed_at

### oms schema
- oms.order_intents: intent_id uuid, portfolio_id, created_at, basket jsonb, execution_style, state
- oms.orders: order_id uuid, intent_id, instrument_id, side, qty, order_type, limit_price, tif, broker, state, submitted_at
- oms.fills: fill_id uuid, order_id, instrument_id, ts_fill, qty, price, fee, liquidity_flag, broker_exec_id
- oms.positions: portfolio_id, instrument_id, ts_position, qty, avg_cost, market_value, unrealized_pnl
- oms.cash_ledger: portfolio_id, ts_cash, currency, cash_amount, reserved_amount

### risk schema
- risk.limits: limit_id uuid, scope_type, scope_id, limit_name, threshold, hard, active
- risk.snapshots: snapshot_id uuid, portfolio_id, ts_snapshot, gross, net, var_1d, cvar_1d, max_drawdown, margin_used
- risk.factor_exposures: snapshot_id, factor_name, exposure
- risk.greeks: snapshot_id, portfolio_id, delta, gamma, vega, theta, dv01
- risk.stress_runs: run_id uuid, requested_at, scenario_set jsonb, status
- risk.stress_results: run_id, scenario_name, pnl_shock, gross_after, net_after

### audit schema
- audit.events: event_id uuid, ts_event, actor, entity_type, entity_id, action, payload jsonb, request_id
- audit.data_incidents: incident_id uuid, ts_opened, severity, source, dataset, description, state
