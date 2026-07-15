# Pairs Trading Toolkit

A statistical-arbitrage toolkit that screens a universe of stocks for cointegrated pairs, applies a mean-reversion spread strategy to each viable pair, and optimises a portfolio across the winners.

## What it does

- **Pair discovery** — scans the tickers in `tickers.txt` for cointegrated pairs.
- **Spread strategy** — for each successful pair, trades the mean-reverting spread and records performance (`pair_trader.py`).
- **Backtesting** — evaluates strategies over historical data (`Backtesting.py`).
- **Portfolio optimisation** — `integrated_portfolio_optimizer.py` combines the successful pairs into an optimised portfolio (e.g. max-Sharpe), pulling the risk-free rate live from the 13-week T-bill (`^IRX`).
- **Charts** — `charts/` holds per-strategy plots, the efficient frontier, and portfolio backtests (e.g. `pairs_efficient_frontier.png`, `pairs_strategy_GOOG_MSFT.png`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Edit `tickers.txt` to change the search universe.

## Note
Research and learning code — not investment advice. Prices pulled live via yfinance.
