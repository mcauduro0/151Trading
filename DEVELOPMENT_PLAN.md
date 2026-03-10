# 151Trading System — Plano de Desenvolvimento Completo

**Versão:** 1.0  
**Data:** 09 de março de 2026  
**Autor:** Manus AI  
**Repositório:** mcauduro0/151Trading  
**Documento de Referência:** 151Trading_plan.docx (105 páginas, 7 Deliverables)

---

## 1. Visão Geral do Projeto

O projeto **151Trading** tem como objetivo construir uma plataforma completa de pesquisa, backtesting e trading que implementa o universo de estratégias descrito em duas fontes primárias: o livro *151 Trading Strategies* (151TS) e o livro *Finding Alphas* (FA). O catálogo completo compreende **251 entradas** (175 de 151TS e 76 de FA), abrangendo opções, ações, ETFs, renda fixa, volatilidade, FX, commodities, futuros, ativos estruturados, conversíveis, arbitragem tributária, ativos em distress, imóveis, caixa, criptomoedas, macro global e infraestrutura.

O documento de referência contém **7 Deliverables** que formam a espinha dorsal do projeto:

| Deliverable | Conteúdo | Páginas |
|---|---|---|
| **D1** — Catálogo Completo de Estratégias | 251 estratégias catalogadas com ID, classe de ativo, estilo, horizonte, direcionalidade e complexidade | 1–12 |
| **D2** — Deep-Dive Matemático (Partes 1 e 2) | Formulações exatas, parâmetros, propriedades estatísticas, expectativas realistas e verdades brutais para cada cluster | 15–44 |
| **D3** — Matriz de Requisitos de Dados | Perfis de dados, mapeamento estratégia-dados, endpoints de provedores e snippets Python | 44–50 |
| **D4** — Auditoria do Repositório GitHub | Avaliação de 20+ bibliotecas open-source, stack recomendado e correções | 50–52 |
| **D5** — Documento de Arquitetura do Sistema | Dois planos (pesquisa e live), componentes, APIs, schemas de banco, fluxos de mensagens | 52–68 |
| **D6** — Wireframes e Especificações do Frontend | 8 páginas principais, sistema de design, hierarquia de componentes, fluxos UX | 68–93 |
| **D7** — Plano do Projeto | 13 sprints (26 semanas), orçamento de esforço, gates de aceitação, registro de riscos | 94–105 |

---

## 2. Princípios Orientadores da Implementação

Antes de detalhar cada fase, é fundamental estabelecer os princípios que governarão todas as decisões de implementação, extraídos diretamente do documento de referência:

**Disciplina de dados point-in-time.** Cada dataset deve preservar três timestamps: `created_at` (quando a fonte criou), `received_at` (quando o sistema recebeu) e `effective_at` (quando o valor se torna legal para uso em simulação). O livro Finding Alphas é explícito: usar dados antes de estarem disponíveis cria viés de look-ahead que invalida toda a pesquisa.

**Honestidade sobre implementabilidade.** Nem todas as 251 estratégias são igualmente implementáveis com dados públicos. Crédito estruturado, CDS basis, swap-spread arbitrage, arbitragem tributária, investimento em distressed debt com controle, imóveis privados e infraestrutura direta são catalogados e arquitetados, mas não são alvos de primeira onda para produção.

**Motor genérico sobre classes individuais.** O capítulo de opções só cabe no cronograma se construirmos um motor de payoff genérico parametrizado. Implementar cada estrutura como uma classe isolada é proibido — desperdiça tempo e aumenta bugs.

**Risco desde o primeiro sprint.** O documento original corrige o roadmap: uma camada fina de risco deve existir desde o Sprint 1, não apenas nos Sprints 11-12. Finding Alphas mostra que neutralização, rank transforms e decay podem melhorar materialmente o information ratio.

**Frontend como produto.** O frontend não é uma camada de relatórios. É o sistema operacional para o portfólio de estratégias. Dashboard, Strategy Manager, Book, Risk, Backtests, Research Workspace e Data Monitor são superfícies de primeira classe.

**Execução via GitHub Actions e Manus.** O deployment utiliza GitHub Actions como runner principal, sem DigitalOcean. O sistema deve operar em seu máximo potencial.

---

## 3. Arquitetura do Sistema (Resumo Executivo)

A arquitetura segue um modelo de **dois planos**:

### Plano 1 — Pesquisa e Simulação
- Código Python de estratégias, workspace de expressões alpha, geração de features em batch
- Backtests pesados sobre snapshots DuckDB e Parquet
- Análise offline e reprodutibilidade total

### Plano 2 — Trading Live e Paper
- Serviços FastAPI finos
- PostgreSQL como sistema de registro (SoR)
- TimescaleDB para séries temporais live
- Redis para cache e fan-out de UI
- Adaptadores de broker: Alpaca, Interactive Brokers, Binance

### Stack Tecnológico Confirmado

| Camada | Tecnologia |
|---|---|
| Backend API | FastAPI (Python) |
| Frontend | Next.js + TypeScript + shadcn/ui + Tailwind + Radix |
| Banco Principal | PostgreSQL + TimescaleDB |
| Banco Analítico | DuckDB + Parquet |
| Cache/Pub-Sub | Redis |
| Task Queue | Celery |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Observabilidade | Prometheus + Grafana |
| Gráficos | Lightweight Charts, Recharts, D3 |
| Tabelas | TanStack Table |
| Fórmulas | KaTeX |
| State Management | TanStack Query (server) + Zustand (client) |

### Decomposição Lógica de Serviços

| Serviço | Responsabilidade |
|---|---|
| **Data Service** | Ingestão, normalização, qualidade, serving, snapshots de pesquisa |
| **Strategy Service** | Registry, runtime, expressões alpha, módulo ML |
| **Backtest Service** | Jobs assíncronos, walk-forward, sensibilidade, artefatos |
| **Portfolio & Risk Service** | Construção de portfólio, risk parity, VaR, stress tests, limites |
| **Execution & OMS** | Intents, ordens, fills, reconciliação, adaptadores de broker |
| **Event Relay** | Outbox + Redis Streams para propagação de eventos |
| **BFF (Backend for Frontend)** | Agregação de chamadas para payloads prontos por página |

---

## 4. Plano de Desenvolvimento Sprint a Sprint

O programa tem **26 semanas** divididas em **13 sprints de 2 semanas** (Sprint 0 a Sprint 12). O orçamento estimado é de **125 person-weeks** para uma equipe lean de ~6 contribuidores efetivos.

---

### Sprint 0 — Fundação (Semanas 1–2)

**Objetivo:** Construir o esqueleto da plataforma e remover todas as dependências bloqueantes futuras.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 0.1 | Setup do Monorepo | Estrutura de diretórios, Docker Compose, `.env`, code owners, branch policy (`main` protegido, PRs obrigatórios) | `docker-compose up` sobe todos os serviços sem erro |
| 0.2 | CI/CD Pipeline | GitHub Actions: lint, type-check, testes unitários, build de containers, deploy para staging | Pipeline verde em PR de teste |
| 0.3 | Schemas Base do Banco | Criar todos os schemas PostgreSQL: `ref`, `md`, `fa`, `strat`, `research`, `oms`, `risk`, `audit` conforme D5 seção 7 | Migrations rodam sem erro; todas as tabelas existem |
| 0.4 | Instrument Master | Tabela `ref.instruments`, `ref.instrument_identifiers`, `ref.calendars`, `ref.corporate_actions`, `ref.universes` | Instrumentos SPY, AAPL, TLT inseridos e consultáveis |
| 0.5 | Primeiros 5 Adaptadores de Ingestão | yfinance (OHLCV diário), FRED (macro), Alpha Vantage (opções/FX), Polygon free (bars), Binance (crypto klines) | Cada adaptador puxa dados e persiste em `md.bars_1d` ou equivalente |
| 0.6 | Strategy Abstract Base Class | Interface padrão: `generate_features()`, `generate_signal()`, `size_positions()`, `check_risk()`, `build_orders()`, `on_fill()`, `get_metadata()` | Classe base importável; uma estratégia dummy a implementa |
| 0.7 | Dataset Manifest Format | Schema JSON para manifestos imutáveis de datasets vinculados a backtests | Manifesto criado e validado por schema |
| 0.8 | FastAPI Service Shell | Primeiro serviço FastAPI com health check, OpenAPI docs, CORS, auth stub | `GET /api/v1/health` retorna 200 |
| 0.9 | Next.js App Shell | Route map completo (`/dashboard`, `/strategies`, `/book`, `/risk`, `/backtests`, `/research`, `/data`, `/alerts`), design system base, sidebar, topbar, theme toggle | App renderiza shell com navegação funcional |
| 0.10 | Observabilidade Base | Prometheus metrics, structured logging, health checks, Grafana dashboard inicial | Métricas visíveis no Grafana |

**Esforço Estimado:** 10 person-weeks

**Gate de Aceitação (Gate A):** Um caminho end-to-end funciona: pull de dados do provider → armazenamento normalizado → resposta da API → card no frontend → execução de backtest trivial. Se esta demo não funcionar, Sprint 1 não deve iniciar.

**Testes e Verificação:**
- Teste de integração: dados fluem de yfinance até `md.bars_1d` e são servidos via API
- Teste de UI: shell do frontend renderiza com dados reais (não mocks)
- Teste de CI: pipeline completo roda em menos de 5 minutos

---

### Sprint 1 — Fatores de Equity Core e Dashboard Shell (Semanas 3–4)

**Objetivo:** Provar o modelo de dados e o loop de backtest nas estratégias de maior valor com dados públicos.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 1.1 | Bars Ajustados (Equities/ETFs) | Corporate actions, split/dividend adjustment, `adj_factor` em `md.bars_1d` | Preços ajustados conferem com yfinance adjusted close |
| 1.2 | SEC Filing Ingestion | Adaptador EDGAR XBRL, `fa.filings`, `fa.facts` com `accepted_at` timestamps | Receita da AAPL disponível com PIT correto |
| 1.3 | Factor Primitives Library | Retornos, volatilidade, rolling rank, z-score, neutralização por indústria, scaling de exposição | Funções testadas unitariamente com dados conhecidos |
| 1.4 | Estratégia S-059: Price Momentum | `Rcum(t;T) = P_i(t)/P_i(t-T) - 1`, winner-minus-loser deciles, T=6-12 meses, skip 1 mês | Backtest reproduz resultados consistentes com literatura |
| 1.5 | Estratégia S-061: Value | Cross-sectional sort em B/P ou composite value z-score | Backtest com métricas visíveis |
| 1.6 | Estratégia S-062: Low-Volatility | Buy-low-σ, short-high-σ, σ sobre 126-252 dias | Backtest com métricas visíveis |
| 1.7 | Estratégia S-065: Residual Momentum | Regressão FF3 36 meses, ε ranking, skip 1 mês | Backtest com métricas visíveis |
| 1.8 | Daily Backtest Engine v1 | Loop diário com custos de transação, benchmark comparison, persistência de artefatos | Backtests salvam manifesto, PnL diário, trades, holdings |
| 1.9 | Dashboard Home v1 | NAV curve, PnL card, drawdown chart, feed health, strategy attribution | Dashboard atualiza com dados reais armazenados |
| 1.10 | Pre-Trade Risk Gate v1 | Max gross, max single-name weight, stale-data block, sector neutrality básica | Ordens que violam limites são bloqueadas |

**Esforço Estimado:** 9.5 person-weeks

**Gate de Aceitação:** 4 estratégias de equity fazem backtest reprodutivelmente com manifestos fixos, métricas visíveis e inputs point-in-time safe. Dashboard Home atualiza com dados reais.

**Testes e Verificação:**
- Cada estratégia tem teste de regressão com resultado esperado fixo
- Teste PIT: backtest com dados "futuros" removidos produz resultado diferente (prova que PIT funciona)
- Dashboard renderiza sem erros de console

---

### Sprint 2 — Mean Reversion, Pairs, ETF Rotation e Strategy Manager (Semanas 5–6)

**Objetivo:** Expandir o motor de estratégias para lógica cross-sectional e relative-value.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 2.1 | Estratégia S-066: Pairs Trading | z-score spread, `z_t = (ln P_A - β ln P_B - μ)/σ`, entrada em `|z|>z_in`, saída em `|z|<z_out` | Backtest com pares conhecidos (ex: KO/PEP) |
| 2.2 | Estratégias S-067/S-068: Mean Reversion (single/multi cluster) | `D_i = -γ Re_i`, dollar-neutral, cluster loading matrix | Backtest sector-neutral funcional |
| 2.3 | Estratégia S-060: Earnings Momentum | SUE = `(EPS_q - EPS_{q-4}) / σ(ΔEPS)`, top/bottom decile | Backtest com timestamps de earnings |
| 2.4 | Estratégias S-080/S-081/S-082: Sector Momentum Rotation | `Rcum`, MA filter, dual-momentum gate | 3 variantes backtestadas |
| 2.5 | Estratégia S-083: Alpha Rotation | Jensen alpha de regressão fatorial, ranking | Backtest funcional |
| 2.6 | Estratégia S-085: ETF Mean Reversion | IBS = `(P_C - P_L)/(P_H - P_L)`, cross-sectional | Backtest funcional |
| 2.7 | Strategy Registry | Metadata, tags, owner, lifecycle state, parameter schemas em `strat.strategies` e `strat.strategy_versions` | Estratégias consultáveis via API |
| 2.8 | Strategy Manager UI | Grid view, table view, filtros (asset class, style, status, risk tier), saved views, quick actions | UI funcional com dados reais |
| 2.9 | Strategy Detail v1 | Overview, Math tab (KaTeX), Parameter tab, Backtest history, Data requirements | Fórmulas renderizadas corretamente |
| 2.10 | Audit Trail | `audit.events` para mudanças de estado de estratégia | Eventos registrados e consultáveis |

**Esforço Estimado:** 10 person-weeks

**Gate de Aceitação (Gate B):** Pelo menos 10-12 estratégias de instrumentos listados rodam reprodutivelmente com manifestos, schemas de parâmetros e métricas visíveis no Strategy Manager.

**Testes e Verificação:**
- Teste de regressão para cada nova estratégia
- Strategy Manager carrega catálogo completo sem lag perceptível
- Mudança de parâmetros dispara preview run sem quebrar manifestos

---

### Sprint 3 — Motor de Payoff de Opções e Camada de Dados de Opções (Semanas 7–8)

**Objetivo:** Construir o motor genérico de opções listadas em vez de codificar dezenas de estratégias isoladas.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 3.1 | Option Contract Master | `md.options_contracts`, chain ingest, normalização de strike/expiry | Chains de SPY disponíveis com metadados corretos |
| 3.2 | Greeks Storage | `md.options_eod` com IV, delta, gamma, vega, theta | Greeks armazenados e consultáveis |
| 3.3 | Generic Payoff Composer | Template master: `f_T = ϕS_T + Σ η_l c(S_T, K_l) + Σ ν_m p(S_T, K'_m) + H`, vetor de coeficientes `(ϕ, η, ν)` parametrizado por família | Uma única engine cobre todas as 58 estruturas |
| 3.4 | Pricing Adapters | Black-Scholes, binomial fallback, hooks para interpolação de superfície | Preços teóricos conferem com referências conhecidas |
| 3.5 | 8 Famílias de Opções | Directional overlays (S-001 a S-004, S-054), Verticals (S-005 a S-016), Synthetics (S-009 a S-012), Long-vol (S-021 a S-023, S-027/28, S-033/34, S-035/36), Short-vol (S-024 a S-026, S-029/30, S-031/32), Range (S-039 a S-052), Box/Arb (S-053), Term-structure (S-017 to S-020) | Cada família gera payoff correto para inputs conhecidos |
| 3.6 | Seagulls e Collars | S-055 a S-058 (seagulls), S-054 (collar) | Payoffs verificados |
| 3.7 | Options Data Quality Checks | Gaps em chains, strikes impossíveis, OI inconsistente, Greeks faltantes | Alertas gerados para dados problemáticos |
| 3.8 | Book & Position Viewer Shell | Posições agrupadas, ordens, fills — estrutura inicial | UI renderiza posições de opções corretamente |

**Esforço Estimado:** 10.5 person-weeks

**Gate de Aceitação (Gate C):** A maioria das estruturas de opções listadas mapeia para um motor de payoff e um caminho de pricing. Se a equipe estiver codificando cada estrutura manualmente, parar e refatorar.

**Testes e Verificação:**
- Teste paramétrico: covered call, bull spread, iron condor, box — payoffs conferem com cálculos manuais
- Teste de cobertura: 58 estruturas mapeadas para ≤8 famílias
- Teste de pricing: BS price para ATM call confere com referência QuantLib

---

### Sprint 4 — Sleeve de Volatilidade, Greeks e Book Viewer (Semanas 9–10)

**Objetivo:** Transformar o motor de opções em estratégias de volatilidade utilizáveis e views de portfólio.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 4.1 | Estratégia S-108: VIX Futures Basis | `BVIX = P_UX1 - P_VIX`, `D = BVIX/T`, regras de entrada/saída explícitas | Backtest com dados VIX históricos |
| 4.2 | Estratégia S-109: ETN Carry (VXX/VXZ) | Short VXX, long VXZ, hedge ratio por regressão serial | Backtest com alertas de risco de cauda |
| 4.3 | Estratégia S-111: Volatility Risk Premium | Short-vol premium capture, proxy com dados disponíveis | Backtest funcional |
| 4.4 | Estratégia S-113: Volatility Skew | Long risk reversal, skew threshold | Backtest funcional |
| 4.5 | Gamma-Hedged Research Variants | S-112: VRP com gamma hedging — modo pesquisa | Simulação funcional onde dados suportam |
| 4.6 | Book & Position Viewer Completo | Posições agrupadas, PnL live, Greeks por estratégia/expiry, cash, margin, attribution | UI completa com dados reais |
| 4.7 | OMS State Machine v1 | Order intents, paper fills, fill replay | Ciclo completo: intent → fill → position update |
| 4.8 | Alertas de Opções | Missing chain slices, stale Greeks, margin overuse | Alertas visíveis no UI |

**Esforço Estimado:** 9.5 person-weeks

**Gate de Aceitação:** Books de opções e volatilidade podem ser backtestados e paper-filled end-to-end. Greeks agregam corretamente de posições para estratégia para portfólio.

---

### Sprint 5 — Spine de Dados de Renda Fixa e Analytics de Taxas (Semanas 11–12)

**Objetivo:** Construir a maquinaria de curvas e duração antes de implementar trades de taxas.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 5.1 | Curve Storage | `md.curve_points` para Treasury e rates, normalização de tenor | Curvas diárias disponíveis via API |
| 5.2 | Duration & Convexity Analytics | Modified duration, Macaulay duration, DV01, convexity | Cálculos conferem com referências |
| 5.3 | Roll-Down Estimates | Carry decomposition: `ΔP/P ≈ carry + rolldown - D*Δy + ½C(Δy)²` | Estimativas para pontos da curva |
| 5.4 | Estratégias S-088/S-089/S-090: Bullets, Barbells, Ladders | Alocações por maturidade | Backtests com proxies de ETF e curvas |
| 5.5 | Estratégia S-091: Bond Immunization | `PV_A = PV_L`, `D_A = D_L`, `C_A ≥ C_L` | Simulação de matching asset-liability |
| 5.6 | Butterfly Framework | S-092 (dollar-duration-neutral), S-093 (fifty-fifty), S-094 (regression-weighted), S-095 (maturity-weighted) | 4 variantes implementadas com DV01 matching |
| 5.7 | Risk Monitor Shell | Exposições correntes, DV01, infraestrutura de cenários inicial | UI com dados de risco reais |

**Esforço Estimado:** 9.5 person-weeks

---

### Sprint 6 — FX, Relative Value de Taxas e Risk Monitor v1 (Semanas 13–14)

**Objetivo:** Completar a primeira superfície de risco cross-asset séria.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 6.1 | Estratégia S-116: FX Carry Trade | `D(t,T) = s(t) - f(t,T) ≈ r_f - r_d`, buy high-carry | Backtest com forwards de 1 mês |
| 6.2 | Estratégia S-117: High-Minus-Low Carry | Cross-sectional top/bottom quantile por `D(t,T)` | Backtest cross-sectional |
| 6.3 | Estratégia S-118: Dollar Carry | Basket version contra USD | Backtest funcional |
| 6.4 | Estratégia S-119: Momentum & Carry Combo | `s_i = z(mom_i) + z(carry_i)` | Backtest combinado |
| 6.5 | Estratégia S-120: FX Triangular Arbitrage | `π_t = S_AB × S_BC × S_CA - 1`, modo simulador | Simulação com custos de transação |
| 6.6 | Yield Curve Spread | S-100: flatteners/steepeners, DV01-neutral | Backtest com curvas reais |
| 6.7 | Factor-Style Bond Sleeves | S-096 (low-risk), S-097 (value), S-098 (carry), S-099 (roll-down) | 4 estratégias backtestadas |
| 6.8 | Risk Monitor v1 | VaR, CVaR, factor exposure, correlation matrix, drawdown monitor, limit utilization | UI completa com snapshots de risco |
| 6.9 | Risk Snapshots | `risk.snapshots`, `risk.factor_exposures`, `risk.greeks` persistidos | Dados de risco consultáveis historicamente |

**Esforço Estimado:** 10 person-weeks

**Gate de Aceitação (Gate D):** Posições cross-asset podem ser vistas em uma superfície de risco com decomposição por fator, moeda e estratégia. Qualquer estratégia sem limites de risco básicos não pode avançar para paper mode.

---

### Sprint 7 — Commodities, Futuros e Backtesting Lab Shell (Semanas 15–16)

**Objetivo:** Construir o framework de contratos contínuos e roll necessário para pesquisa de commodities e futuros.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 7.1 | Futures Contract Master | `md.futures_contracts`, expiry handling, roll schedules | Contratos de ES, CL, GC disponíveis |
| 7.2 | Continuous Series | Adjusted continuous series com roll policies explícitas | Séries contínuas sem gaps de roll |
| 7.3 | Term-Structure Snapshots | Curvas de futuros por data | Snapshots consultáveis |
| 7.4 | COT Data Ingestion | CFTC Commitments of Traders | Dados de posicionamento disponíveis |
| 7.5 | Estratégia S-121: Roll Yields | `RollYield ≈ (F_near - F_next)/F_near × 365/ΔT` | Backtest com dados de curva |
| 7.6 | Estratégia S-122: Hedging Pressure | HP = long/(long+short), ranking por pressão | Backtest com dados COT |
| 7.7 | Estratégia S-133: Trend Following | `w_t ∝ sign(R_{t-L:t-1})` ou MA crossover z-scored | Backtest diversificado multi-mercado |
| 7.8 | Estratégias S-131/S-132: Contrarian | Mean-reversion e flow-sensitive em futuros | Backtests funcionais |
| 7.9 | Estratégia S-130: Calendar Spread | Long/short maturidades diferentes | Backtest funcional |
| 7.10 | Backtesting Lab Shell | Config panel, run queue, result artifacts, benchmark comparison básico | UI funcional para lançar e inspecionar runs |

**Esforço Estimado:** 9.5 person-weeks

---

### Sprint 8 — Crypto ML e Backtesting Lab Completo (Semanas 17–18)

**Objetivo:** Adicionar o primeiro sleeve de ML verdadeiro e tornar a experiência de pesquisa séria.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 8.1 | Crypto Feature Pipelines | Binance klines + CoinGecko market charts, features técnicos | Pipeline de features funcional |
| 8.2 | Estratégia S-169: ANN (Crypto) | Cross-entropy training, walk-forward, class prediction | Modelo treinado e avaliado OOS |
| 8.3 | Estratégia S-170: Naive Bayes Sentiment | Bernoulli NB em features de texto, `ŷ_t = argmax P(C_α) Π P(X_a|C_α)` | Classificador funcional com dados reais |
| 8.4 | Model Registry | Versionamento de modelos, artifact storage, leakage checks | Modelos rastreáveis com lineage |
| 8.5 | Train/Val/Test Framework | Partições purged, walk-forward jobs, cross-validation | Framework reutilizável para qualquer ML |
| 8.6 | Backtesting Lab Completo | Walk-forward panels, sensitivity heatmaps, trade inspection, multi-strategy combined backtests | Todas as funcionalidades de D6 §6.6 |
| 8.7 | Métricas Obrigatórias | Return, IR, drawdown, turnover, profit per dollar traded visíveis em todo run | Métricas presentes em todos os backtests |

**Esforço Estimado:** 10 person-weeks

**Gate de Aceitação (Gate E):** Walk-forward, sensibilidade e artifact storage funcionam para estratégias determinísticas e ML. Todo backtest expõe as métricas core de Finding Alphas.

---

### Sprint 9 — Construção de Portfólio, Alpha Combos e Sleeve Macro (Semanas 19–20)

**Objetivo:** Mover de books de estratégias isoladas para alocação a nível de portfólio.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 9.1 | Risk Parity | Alocação inversamente proporcional à volatilidade | Optimizer funcional |
| 9.2 | HRP (Hierarchical Risk Parity) | Clustering hierárquico + alocação | Optimizer funcional |
| 9.3 | Mean-Variance | Markowitz com constraints | Optimizer funcional |
| 9.4 | Black-Litterman | Views + prior de mercado | Optimizer funcional |
| 9.5 | Alpha Combo Logic | Blending ponderado, penalidade de correlação, filtros de capacidade (S-079) | Combinação de alphas funcional |
| 9.6 | Transaction Cost Model | Spread, comissão, market impact, slippage | Modelo integrado em rebalance preview |
| 9.7 | Macro Data Release Calendar | Calendário de releases econômicos | Calendário consultável |
| 9.8 | Estratégia S-174: Trading on Economic Announcements | Buy stocks em FOMC days, risk-free em non-announcement days | Backtest funcional |
| 9.9 | Estratégias S-171/S-172/S-173: Global Macro | Fundamental macro momentum, inflation hedge, global FI | Backtests com dados macro |
| 9.10 | Rebalance Preview | Optimizer outputs em preview com custos | UI de preview funcional |

**Esforço Estimado:** 10 person-weeks

---

### Sprint 10 — Alpha Research Workspace e Proxies de Estratégias Complexas (Semanas 21–22)

**Objetivo:** Entregar o diferenciador da plataforma mantendo honestidade sobre o que não é live-feasible.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 10.1 | Expression Editor | Editor FA-style com autocomplete, syntax validation, unit checking | Expressões como `1/close`, `volume/adv20` funcionam |
| 10.2 | FA Operator Family | `Delay`, `Delta`, `Correlation`, `Rank`, `Ts_Rank`, `Decay_linear`, `IndNeutralize`, `Sign`, `Product` | Todos os operadores implementados e testados |
| 10.3 | Alpha Simulation Engine | Delay, neutralização, decay, truncation, normalização para weights | Simulação end-to-end funcional |
| 10.4 | IC, Turnover, Sharpe, Correlation | Métricas de avaliação de alpha | Todas as métricas calculadas corretamente |
| 10.5 | Decay Analysis | Sensitivity de decay sobre diferentes janelas | Gráfico de decay funcional |
| 10.6 | My Alphas Inventory | Tabela filtrável/ordenável conforme D6 §6.7 | Inventário com todas as colunas de FA |
| 10.7 | Daily PnL Storage | PnL diário e holdings diários para cada alpha run | Dados armazenados e consultáveis (correção vs WebSim) |
| 10.8 | IS vs OS Panel | Sharpe30, Sharpe60, Sharpe120, Sharpe252, total OS Sharpe, alpha-to-pool correlation, novelty score | Painel funcional com thresholds visuais |
| 10.9 | Research Proxies | Módulos research-only para structured assets, convertibles, distressed, real estate, infrastructure, tax arbitrage | Metadados e proxies catalogados, claramente rotulados |

**Esforço Estimado:** 10.5 person-weeks

**Gate de Aceitação (Gate F):** Um pesquisador pode escrever uma expressão alpha, validá-la, simulá-la, inspecionar métricas IS e OS, compará-la com a biblioteca de alphas e decidir se merece promoção.

---

### Sprint 11 — Hardening de Integração, Risk Layer Completo e Paper OMS (Semanas 23–24)

**Objetivo:** Transformar uma coleção de bons módulos em uma plataforma de trading estável.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 11.1 | Broker Adapters (Paper) | Alpaca, Interactive Brokers, Binance — paper mode primeiro | Cada adapter conecta e recebe fills simulados |
| 11.2 | Order Lifecycle Completo | Intent → risk_approved → submitted → ack → partial_fill → filled → cancelled/rejected/expired | Todos os estados transitam corretamente |
| 11.3 | Fill Reconciliation | Cash e margin ledger, cancel flows, incident logging | Posições reconciliam com fills |
| 11.4 | Stress Testing | Cenários históricos e hipotéticos, scenario replays | Stress runs executam e resultados visíveis |
| 11.5 | Circuit Breakers | Strategy-level e portfolio-level de-risking rules | Breakers disparam corretamente |
| 11.6 | Data Monitor | Feed status, coverage, latency, incidents, strategy impact | UI completa conforme D6 §6.8 |
| 11.7 | Integration Tests E2E | Ingestão → sinal → risk check → ordem → fill → posição → UI | Dia de mercado simulado sem divergência de estado |

**Esforço Estimado:** 10 person-weeks

---

### Sprint 12 — Polish, UAT, Paper-Trading Deployment e Go-Live Gate (Semanas 25–26)

**Objetivo:** Produzir um candidato a produção, não apenas uma demo.

**Escopo de Trabalho:**

| # | Tarefa | Detalhe | Critério de Verificação |
|---|---|---|---|
| 12.1 | Frontend Performance | Rendering optimization, websocket patching, table virtualization, chart payload size | First paint < 2s, TTI < 3s, scroll suave em 10k rows |
| 12.2 | Mobile Monitoring Views | Dashboard summary, alerts, recent fills, emergency pause | Funcional em tela < 768px |
| 12.3 | Alert Center & Incidents | Workflows completos de alertas e incidentes | Fluxo end-to-end funcional |
| 12.4 | Runbooks & Documentation | Operator docs, troubleshooting guides, architecture docs | Documentação completa e revisada |
| 12.5 | Security Review | Secret manager, least-privilege roles, audit logs, environment separation | Checklist de segurança aprovado |
| 12.6 | Load Testing | Simulação de carga live-like | Performance aceitável sob carga |
| 12.7 | UAT | Workflows de PM, researcher, risk manager, ops | UAT aceito por todos os perfis |
| 12.8 | Paper-Trading Soak | Set fixo de estratégias, limites fixos, reconciliação diária | 10 dias de trading paper sem incidentes críticos |

**Esforço Estimado:** 10 person-weeks

**Gate de Aceitação (Gate G):** Release candidato à produção, runbooks assinados, UAT aceito, paper mode rodando sem incidentes críticos não resolvidos. Capital live bloqueado até o soak gate passar.

---

## 5. Mapa de Dependências Críticas

O diagrama abaixo mostra as dependências que, se atrasadas, impactam todo o programa:

| Dependência | O Que Bloqueia | Sprint Necessário |
|---|---|---|
| Instrument master, calendários, corporate actions | Quase tudo | Sprint 0 |
| Suporte PIT para filings e fundamentals | Value, earnings momentum, FA modules | Sprint 1 |
| Backtest engine e manifestos de artefatos | Velocidade de pesquisa, reprodutibilidade | Sprint 1 |
| Normalização de options chains e Greeks | Opções, volatilidade, dispersion, Greeks views | Sprint 3 |
| Yield-curve e duration analytics | Renda fixa, macro sleeves | Sprint 5 |
| Continuous futures e roll logic | Commodities, futures trend, carry | Sprint 7 |
| OMS state machine e reconciliação | Paper trading, todas as páginas de holdings live | Sprint 4/11 |
| Risk snapshots e limit engine | Qualquer claim de deployment paper/live | Sprint 6 |
| Alpha-expression parser e validation | Research workspace, diferenciação do produto | Sprint 10 |

---

## 6. Orçamento de Esforço por Workstream

| Workstream | Person-Weeks | % do Total |
|---|---|---|
| Data layer, storage, PIT, provider adapters | 23 | 18.4% |
| Strategy runtime, backtest engine, alpha engine | 19 | 15.2% |
| Strategy implementations (todas as classes de ativo) | 26 | 20.8% |
| Frontend application (todas as páginas) | 24 | 19.2% |
| Portfolio construction e risk services | 14 | 11.2% |
| OMS, broker adapters, paper trading | 7 | 5.6% |
| DevOps, CI/CD, observabilidade, QA, segurança | 7 | 5.6% |
| Documentação, runbooks, UAT, release management | 5 | 4.0% |
| **Total** | **125** | **100%** |

---

## 7. Registro de Riscos

| # | Risco | Prob. | Impacto | Mitigação | Owner |
|---|---|---|---|---|---|
| R1 | Leakage point-in-time | Alta | Crítico | Enforce as_of joins, accepted_at timestamps, manifest pinning, synthetic leakage tests | Data Lead |
| R2 | Dados públicos não confiáveis | Alta | Alto | Quarantine bad slices, provider precedence, data-health state visível | Data Lead |
| R3 | Decay e crowding de estratégias | Alta | Alto | Track factor loadings, post-neutralization performance, penalize alpha correlation | Quant Lead |
| R4 | Overfitting e p-hacking | Alta | Crítico | Force IS/OS splits, walk-forward, holdout windows, model lineage, approval review | Quant Lead |
| R5 | Insuficiência de dados de opções | Média-Alta | Alto | Foco em sleeves factíveis, payoff engine genérico, dispersion/variance como tier 2 | Options Lead |
| R6 | Explosão de escopo do frontend | Alta | Alto | Lock page priorities (Home, Strategies, Book, Risk, Backtests, Research, Data), defer mobile parity | Frontend Lead |
| R7 | Divergência backtest-live | Média | Crítico | Interface compartilhada, cost model compartilhado, order simulator compartilhado, drift checks diários | Architect |
| R8 | Falha de reconciliação OMS/broker | Média | Crítico | Estados de ordem imutáveis, ingest idempotente, reconciliation jobs, incident workflows | Backend Lead |
| R9 | Sleeves institucionais consomem schedule | Alta | Médio | Labels research-only, proxy implementations, live-feasibility tags explícitos | Product + Quant |
| R10 | Dependência de pessoa-chave | Média | Alto | Docs compartilhados, code reviews, paired implementation, bus-factor review por sprint | PM/Architect |
| R11 | Gaps de segurança e controle de acesso | Média | Alto | Secret manager, least-privilege, audit logs, environment separation, release checklist | DevOps Lead |
| R12 | Degradação de performance sob carga live | Média | Médio-Alto | Tabelas virtualizadas, socket patches, server-side aggregation, load testing em Sprint 11-12 | Frontend + Platform |

---

## 8. Definição de "Done"

### Para uma Estratégia:
1. A fórmula é rastreável ao texto-fonte ou a uma aproximação explicitamente documentada
2. O perfil de dados necessário existe e é point-in-time safe
3. O backtest inclui custos de transação realistas e armazena artefatos
4. Controles de risco básicos existem (max size, drawdown stop, stale-data block)
5. A página de detalhe da estratégia expõe premissas, parâmetros, failure modes e notas de capacidade

### Para um Sprint:
1. Código merged e testes passando
2. Telemetria existe
3. Documentação atualizada
4. Uma demo cross-functional funciona de dados até UI

---

## 9. Meta de Entrega na Semana 26

Ao final do Sprint 12, o sistema deve ter:

- Uma plataforma estável de pesquisa e paper-trading no stack escolhido
- Um frontend forte cobrindo Dashboard, Strategy Manager, Book, Risk, Backtests, Research e Data Monitor
- Backtests reprodutíveis com manifestos e PnL diário armazenado
- Construção de portfólio, risk checks e reconciliação OMS
- **35-50 estratégias paper-trading ready**
- **80-110 estratégias research-backtest ready**
- Suporte research-only para sleeves institucionais e de baixa viabilidade
- Um go-live gate que requer 10 dias de paper soak antes de capital ser permitido

---

## 10. Próximos Passos Imediatos

Para iniciar a implementação, proponho a seguinte sequência de ações:

1. **Confirmar e alinhar** este plano com o usuário antes de qualquer execução
2. **Sprint 0 — Fase 1:** Setup do monorepo, Docker Compose, CI/CD pipeline
3. **Sprint 0 — Fase 2:** Schemas de banco, instrument master, primeiros adaptadores
4. **Sprint 0 — Fase 3:** FastAPI shell, Next.js shell, observabilidade
5. **Sprint 0 — Fase 4:** Demo end-to-end e Gate A

Cada sprint será executado com:
- Briefing de início com escopo confirmado
- Checkpoints intermediários com verificação
- Demo de final de sprint com validação de Gate
- Retrospectiva e ajuste de plano se necessário

---

## 11. Perguntas para Clarificação

Antes de iniciar a execução, as seguintes questões precisam de alinhamento:

1. **Prioridade de execução:** Deseja que comecemos imediatamente pelo Sprint 0 (setup de infraestrutura e monorepo), ou prefere revisar e ajustar este plano primeiro?

2. **API Keys e Provedores:** Quais API keys já estão disponíveis? O plano assume yfinance (sem key), FRED (key necessária), Alpha Vantage (key necessária), Polygon free (key necessária), e Binance (key necessária para alguns endpoints).

3. **Escopo do Frontend:** O plano segue fielmente o D6 com 8 páginas principais. Deseja priorizar alguma página específica para entrega antecipada?

4. **Broker Adapters:** Quais brokers são prioritários? O plano inclui Alpaca, Interactive Brokers e Binance. Algum deve ser priorizado?

5. **Equipe:** O plano assume ~6 contribuidores. Se a execução for primariamente via Manus, devemos ajustar o paralelismo e a granularidade das tarefas?

6. **Automação Diária:** Conforme preferência registrada, o sistema deve ter rotina diária automatizada. Deseja que isso seja implementado desde o Sprint 0 ou em sprint posterior?
