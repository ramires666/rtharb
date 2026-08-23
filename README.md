# Проверенный пересчёт QQQ → NVDA

## TradingView Lite — последние два месяца

Откройте `launchers/open_tradingview_lite_report.bat`. Отчёт использует настоящие
синхронизированные 1-минутные Alpaca SIP-бары QQQ/NVDA, показывает Z-score,
точные входы/выходы, trade-box'ы и переключаемые EMA. Для пересборки из
локального кэша запустите `launchers/rebuild_tradingview_lite_report.bat`.

Готовые интерактивные отчёты:

- `tradingview_lite/index.html` — QQQ/NVDA, свечи, EMA/HMA, session VWAP,
  Z-score, точные сделки, trade-box'ы и mark-to-market equity;
- `tradingview_synthetic/index.html` — QQQ против фиксированной mega-cap
  корзины MSFT/AAPL/NVDA/AMZN, схождение и отдельный RR-режим;
- `tradingview_vwap_absolute/index.html` — подробный самостоятельный raw
  event-driven VWAP-Z fixed-bracket: QQQ как lead, торгуется только NVDA,
  точные VWAP/fair/Z, stop/target-зоны, сделки, equity и drawdown.
- `tradingview_vwap_absolute_multi_asset/index.html` — поэтапный независимый
  тест девяти акций против reference QQQ: отдельные VWAP, Z, сделки,
  фиксированные долларовые stop/target, equity и drawdown для каждого тикера.

Полный пересчёт последнего отчёта одним кликом:
`launchers/run_vwap_absolute_event_driven.bat`.

Рабочая копия и независимый аудит проекта статистического арбитража. Оригинал
Джона находится в соседней папке `rtharb` и не изменён.

## Структура проекта

- `rtharb/research/` — рабочие исследования стратегий;
- `rtharb/reporting/` — сборщики интерактивных отчётов;
- `rtharb/audit/` — независимые проверки и пересчёты;
- `configs/` — конфигурация Alpaca SIP и стратегии;
- `data_cache/` — локальные raw 1-minute Parquet;
- `tests/` — автоматические проверки;
- `research_output/`, `audit_output/`, `equity_output/` — воспроизводимые результаты;
- `tradingview_*` — актуальные интерактивные отчёты с equity;
- `launchers/` — однокнопочные Windows-запускатели;
- `old/` — опровергнутые проверки и ранние прототипы, не рабочий пайплайн.

Для multi-asset VWAP-bracket заранее зафиксирован исследовательский universe:
`NVDA, MSFT, AAPL, AMZN, GOOGL, META, AVGO, TSLA, AMD`; QQQ используется
только как ведущий reference. Raw Alpaca SIP-котировки и их SHA-256/coverage
проверяются командой `launchers/download_mega_cap_data.bat` и manifest-файлом
`data_cache/mega_cap_sip_manifest.json`.
Расчёт запускается через `launchers/run_vwap_absolute_multi_asset.bat`, а
текущий интерактивный результат — через
`launchers/open_vwap_absolute_multi_asset_report.bat`.

Завершённый итог одинакового frozen-сигнала для девяти отдельно торгуемых
акций (full net P&L / честный holdout net P&L): `NVDA +$6,691.89 / +$2,199.54`,
`MSFT +$323.76 / -$1,664.11`, `AAPL -$3,421.55 / -$1,898.55`,
`AMZN +$446.16 / -$102.06`, `GOOGL -$4,427.83 / -$3,080.42`,
`META -$594.74 / -$1,748.42`, `AVGO +$1,708.67 / -$909.27`,
`TSLA +$1,167.86 / -$948.74`, `AMD +$2,053.44 / -$2,814.79`.
Только NVDA положительна на holdout; это exploratory multiple testing, а не
доказанный live-edge. Точные equity, drawdown, gross/costs и каждая сделка
доступны в интерактивном отчёте.

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

Расширенный multi-asset raw event-driven VWAP-Z bracket-тест для NVDA выбрал
внутренний, не упирающийся в границу сетки стоп `$5.25/акция` и цель
`$1.25/акция` (`0.238R`): holdout `+$2,199.54`, полный год `+$6,691.89`,
минутный mark-to-market MDD `$1,929.83 / 1.81%`.
Ранний одиночный тест выбрал граничный стоп `$3.00`, поэтому его результат
`+$6,597.51` нельзя считать окончательным оптимумом.
Старый frozen-cohort результат `$2.00/$1.25` не регенерировал сигналы после
раннего bracket-выхода и оставлен только как аудиторское сравнение. Новый тест
генерирует entry-события напрямую из каждой raw минуты, когда позиция закрыта.
Архив старой проверки целиком перенесён в `old/frozen_vwap_absolute/`.
Это один исторический годовой holdout, а не доказательство устойчивого
live-edge. Обычное схождение, процентные VWAP-RR варианты и QQQ против
синтетической корзины положительный holdout не подтвердили.

Во всех стандартных исследованиях использованы позиция `$20,000`, стартовый
капитал `$100,000`, комиссия `$0.0035/акция/сторона` и slippage `2 bps` на
каждое исполнение. Borrow fee, налоги, задержка и market impact не включены.

Старые HTML/MD/SVG оставлены только как legacy-материалы и не являются
результатом нового аудита. В частности, `images/session_*.svg` — схематичные
ломаные, а не 390 минутных свечей.

## Запуск

```powershell
..\rtharb\.venv\Scripts\python.exe -m rtharb.audit.recalculate
..\rtharb\.venv\Scripts\python.exe -m rtharb.audit.integrity
```

Для просмотра готового отчёта дважды щёлкните `launchers/open_audit_report.bat`.

Артефакты нового расчёта:

- `audit_output/REPORT.html` — краткий честный отчёт;
- `audit_output/summary.json` — все метрики;
- `audit_output/training_parameter_grid.csv` — сетка сигналов;
- `audit_output/training_filter_grid.csv` — сетка time-stop/stop-loss;
- `audit_output/trades_*.csv` — полный реестр сделок;
- `audit_output/session_bar_audit.csv` — сверка каждой сессии с календарём;
- `audit_output/session_2026-08-21.svg` — 390 реальных минутных свечей.

## Широкое исследование базового edge

`rtharb/research/base_strategy.py` отдельно исследует саму стратегию без stop-loss и
time-stop. Перебираются Z-entry, hook/no-hook, hook timeout, beta, rolling
window, exit band и 4σ-lockout. Development/validation/holdout разделены
хронологически. Готовый отчёт: `research_output/BASE_STRATEGY_REPORT.html`.

Запуск одним кликом: `launchers/run_base_research.bat`.

Это исследовательский бэктест, не инвестиционная рекомендация. Borrow fee,
налоги, влияние ордера на рынок и задержка данных не моделируются.
