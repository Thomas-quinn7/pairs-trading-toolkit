# Archived charts

Output of retired engine versions, kept only as the "before" side of the
repo's write-up. Nothing in the active pipeline reads or writes here.

- `pairs_strategy_*.png`, `heatmap.png`, `coint_pairs.csv`,
  `pairs_efficient_frontier.png`, `pairs_portfolio_*.png` — produced by the
  quarantined v1 engine (`../pair_trader_v1_lookahead.py`), whose z-score had
  look-ahead bias and whose Markowitz weights were fitted on the reported
  window. The numbers in these images are not earnable in real time.
- `oos_strategy_AMZN_META*.png`, `oos_strategy_GOOGL_META.png`,
  `oos_strategy_GOOGL_MSFT.png` — early runs of the current OOS engine,
  superseded by later screening (Engle-Granger) and styling changes.

Current charts live in `../../charts/`.
