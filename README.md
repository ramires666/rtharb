# Проверенный пересчёт QQQ → NVDA

## TradingView Lite — последние два месяца

Откройте `open_tradingview_lite_report.bat`. Отчёт использует настоящие
синхронизированные 1-минутные Alpaca SIP-бары QQQ/NVDA, показывает Z-score,
точные входы/выходы, trade-box'ы и переключаемые EMA. Для пересборки из
локального кэша запустите `rebuild_tradingview_lite_report.bat`.

Готовые интерактивные отчёты:

- `tradingview_lite/index.html` — QQQ/NVDA, свечи, EMA/HMA, session VWAP,
  Z-score, точные сделки, trade-box'ы и mark-to-market equity;
- `tradingview_synthetic/index.html` — QQQ против фиксированной mega-cap
  корзины MSFT/AAPL/NVDA/AMZN, схождение и отдельный RR-режим;
- `research_output/vwap_absolute_brackets/REPORT.html` — независимый перебор
  абсолютных стопов и целей VWAP-Z за последний завершённый год.

Рабочая копия и независимый аудит проекта статистического арбитража. Оригинал
Джона находится в соседней папке `rtharb` и не изменён.

## Что считается

- только официальная сессия США в `America/New_York`;
- обычный день: 09:30–16:00 (390 минут), официальный half-day: 09:30–13:00 (210 минут);
- данные Alpaca SIP, QQQ — ведущий инструмент, NVDA — торгуемый ведомый;
- на первой минуте дня нормированные доходности и spread равны нулю;
- beta использует только завершившиеся предыдущие дни;
- rolling mean/std spread считаются внутри текущей сессии по правилам MD;
- сигнал на close исполняется на следующем open;
- учтены комиссия и проскальзывание, позиции не переносятся через ночь;
- time-stop и stop-loss исследуются по duration/MAE прибыльных сделок на
  development, выбираются на validation и один раз проверяются на holdout.

## Главный результат

Старые `+$53,210`, `71.6% win rate`, `120 min / 1.5%` не воспроизводятся.
На полном SIP и с честным исполнением финальный holdout у базовой стратегии
отрицательный; защитные фильтры слегка уменьшают убыток/просадку, но не создают
положительного expectancy. Точные числа находятся в `audit_output/summary.json`
и `audit_output/REPORT.html`.

Последующий независимый VWAP-Z bracket-тест дал один перспективный результат:
стоп `$2.00/акция`, цель `$1.25/акция` (`0.625R`), holdout `+$2,229.66`,
полный год `+$5,475.69`, минутный mark-to-market MDD `$1,926.35 / 1.88%`.
Это frozen-entry исследование на одном годовом holdout, а не доказательство
устойчивого live-edge. Обычное схождение, процентные VWAP-RR варианты и QQQ
против синтетической корзины положительный holdout не подтвердили.

Во всех стандартных исследованиях использованы позиция `$20,000`, стартовый
капитал `$100,000`, комиссия `$0.0035/акция/сторона` и slippage `2 bps` на
каждое исполнение. Borrow fee, налоги, задержка и market impact не включены.

Старые HTML/MD/SVG оставлены только как legacy-материалы и не являются
результатом нового аудита. В частности, `images/session_*.svg` — схематичные
ломаные, а не 390 минутных свечей.

## Запуск

```powershell
..\rtharb\.venv\Scripts\python.exe recalculate_audit.py
..\rtharb\.venv\Scripts\python.exe audit_integrity.py
```

Для просмотра готового отчёта дважды щёлкните `open_audit_report.bat`.

Артефакты нового расчёта:

- `audit_output/REPORT.html` — краткий честный отчёт;
- `audit_output/summary.json` — все метрики;
- `audit_output/training_parameter_grid.csv` — сетка сигналов;
- `audit_output/training_filter_grid.csv` — сетка time-stop/stop-loss;
- `audit_output/trades_*.csv` — полный реестр сделок;
- `audit_output/session_bar_audit.csv` — сверка каждой сессии с календарём;
- `audit_output/session_2026-08-21.svg` — 390 реальных минутных свечей.

## Широкое исследование базового edge

`research_base_strategy.py` отдельно исследует саму стратегию без stop-loss и
time-stop. Перебираются Z-entry, hook/no-hook, hook timeout, beta, rolling
window, exit band и 4σ-lockout. Development/validation/holdout разделены
хронологически. Готовый отчёт: `research_output/BASE_STRATEGY_REPORT.html`.

Запуск одним кликом: `run_base_research.bat`.

Это исследовательский бэктест, не инвестиционная рекомендация. Borrow fee,
налоги, влияние ордера на рынок и задержка данных не моделируются.
