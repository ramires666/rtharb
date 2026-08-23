# Verified duration / stop-loss research

Exact frozen entries are generated from `base_strategy_summary.json` parameters and the standard SignalGenerator. Development profitable trades only determine 90/95/97.5/99% duration and intrabar MAE thresholds. Time-stop-only and stop-loss-only overlays are evaluated separately on development, validation, holdout and full periods using next-open fills, raw 1m high/low stop detection, conservative gap handling from BacktestEngine, costs and forced EOD. No combined overlay is tested.
