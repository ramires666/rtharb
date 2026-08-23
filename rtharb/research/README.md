# Исследования

Воспроизводимые event-driven и walk-forward исследования. Запускаются из
корня проекта как Python-модули, например:

```powershell
python -m rtharb.research.vwap_absolute_event_driven
```

Сгенерированные CSV/JSON/HTML не хранятся здесь: они записываются в
`research_output/` и соответствующие `tradingview_*` каталоги.
