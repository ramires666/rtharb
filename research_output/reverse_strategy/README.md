# Exact reverse entry research

Frozen parameters are read from `research_output/base_strategy_summary.json`: dynamic beta 5d, window 60, z-entry 3.0, hook 0.15/5, exit 0, lockout 3.5. The original SignalGenerator creates source signals; only BUY_LONG and SELL_SHORT are swapped. Exits remain unchanged. Both runs use the same BacktestEngine next-open fills, integer shares, commissions, slippage and forced EOD. Every period includes base/inverse trade CSVs, event timestamp/duration equality checks, and gross-cost-net reconciliation. Last-two-months starts at 2026-06-22.
