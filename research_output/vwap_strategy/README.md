# Causal session-VWAP research

VWAP is cumulative within each RTH session using typical price weighted by volume, and includes only bars through the current signal close. Fair value is `NVDA_VWAP * (1 + beta * (QQQ_close / QQQ_VWAP - 1))`. The first 3/5/10/15/30 completed bars are tested as warm-up. Signals use the existing hook, next-open execution, costs and forced EOD; no SL/time-stop. Development selects 10 signal finalists, expands them across beta/window choices, validation selects one, and holdout is evaluated once.
