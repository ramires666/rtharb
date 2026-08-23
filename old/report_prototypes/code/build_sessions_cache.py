"""Generate on-demand 1-minute session chunks and interactive UI."""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.analysis.matrix_comparator import MatrixComparator
from rtharb.config import AppConfig


def build_all_sessions():
    print("⏳ Loading 2-year 1m data from local Parquet cache...")
    loader = DataLoader(cache_dir="data_cache", source="alpaca")
    df_lead, df_target = loader.get_synchronized_pair("QQQ", "NVDA", days_back=730, source="alpaca")

    print(f"✅ Loaded {len(df_target):,} bars across {df_target['session_date'].nunique()} trading days.")

    print("⏳ Computing Fair Value, Beta, Spreads, and Z-Scores...")
    fv = FairValueModel(beta_mode="dynamic_rolling", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)

    print("⏳ Simulating 4 Scenarios...")
    cfg = AppConfig()
    comparator = MatrixComparator(cfg)
    matrix_res = comparator.run_all_scenarios(df_metrics)

    rec_res = matrix_res["results"]["B: Entry Lockout Only (Recommended)"]
    trades_b = rec_res["trades_df"]
    df_b_signals = rec_res["df_results"]

    # 1. Real Equity Curves (Sampled every 10 bars for fast rendering)
    eq_df = matrix_res["equity_curves"]
    step = 10
    eq_sampled = eq_df.iloc[::step].copy()
    if eq_df.index[-1] != eq_sampled.index[-1]:
        eq_sampled = pd.concat([eq_sampled, eq_df.iloc[[-1]]])

    equity_payload = {
        "timestamps": [t.strftime("%Y-%m-%d %H:%M") for t in eq_sampled.index],
        "scenarios": {}
    }
    for col in eq_sampled.columns:
        equity_payload["scenarios"][col] = [round(float(v), 2) for v in eq_sampled[col].values]

    # 2. Performance Comparison Rows
    comp_metrics = []
    for s_name, res in matrix_res["results"].items():
        m = res["metrics"]
        comp_metrics.append({
            "name": s_name,
            "pnl": f"${m.total_pnl:,.2f}",
            "return": f"{m.total_return_pct:+.2f}%",
            "cagr": f"{m.cagr_pct:.2f}%",
            "sharpe": f"{m.sharpe_ratio:.2f}",
            "sortino": f"{m.sortino_ratio:.2f}",
            "max_dd_pct": f"{m.max_drawdown_pct:.2f}%",
            "max_dd_usd": f"${m.max_drawdown_usd:,.2f}",
            "trades": m.total_trades,
            "win_rate": f"{m.win_rate_pct:.1f}%",
            "profit_factor": f"{m.profit_factor:.2f}",
            "commissions": f"${m.total_commissions:,.2f}",
            "emergency_exits": m.exit_reasons_breakdown.get("EMERGENCY_4SIGMA", 0)
        })

    metrics_rows = ""
    for row in comp_metrics:
        is_rec = "Recommended" in row["name"]
        badge = '<span class="badge badge-success">⭐ ЛУЧШИЙ</span>' if is_rec else ""
        metrics_rows += f"""
        <tr class="{'highlight-row' if is_rec else ''}">
            <td><strong>{row['name']}</strong> {badge}</td>
            <td class="text-green font-bold">{row['pnl']} ({row['return']})</td>
            <td><strong>{row['sharpe']}</strong></td>
            <td>{row['sortino']}</td>
            <td class="text-red"><strong>{row['max_dd_pct']}</strong> ({row['max_dd_usd']})</td>
            <td><strong>{row['win_rate']}</strong></td>
            <td>{row['profit_factor']}</td>
            <td>{row['trades']}</td>
            <td>{row['commissions']}</td>
            <td>{row['emergency_exits']}</td>
        </tr>
        """

    # 3. Real Trades List
    trades_payload = []
    for _, tr in trades_b.iterrows():
        pnl = float(tr["net_pnl"])
        trades_payload.append({
            "id": int(tr["trade_id"]),
            "date": tr["entry_time"].strftime("%Y-%m-%d"),
            "dir": "🟢 LONG" if tr["direction"] == 1 else "🔴 SHORT",
            "entry_time": tr["entry_time"].strftime("%Y-%m-%d %H:%M"),
            "entry_price": f"${tr['entry_price']:.2f}",
            "exit_time": tr["exit_time"].strftime("%Y-%m-%d %H:%M"),
            "exit_price": f"${tr['exit_price']:.2f}",
            "pnl": round(pnl, 2),
            "pnl_str": f"{'+' if pnl >= 0 else ''}${pnl:,.2f}",
            "return_pct": f"{tr['return_pct']*100:+.2f}%",
            "duration": f"{int(tr['duration_bars'])} мин",
            "reason": tr["exit_reason"],
            "entry_z": f"{tr['entry_z_score']:.2f}",
            "exit_z": f"{tr['exit_z_score']:.2f}"
        })

    # 4. Create on-demand sessions folder
    sessions_dir = project_root / "standalone_report" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    all_days = sorted(df_metrics["session_date"].unique())
    days_meta = []

    print(f"⏳ Generating on-demand 1-minute session chunks for {len(all_days)} trading days...")
    for d in all_days:
        d_str = str(d)
        df_d = df_b_signals[df_b_signals["session_date"] == d]
        if df_d.empty:
            continue

        trs_on_day = trades_b[trades_b["entry_time"].dt.date == d]
        trade_summary = ""
        pnl_day = 0.0
        if not trs_on_day.empty:
            pnl_day = trs_on_day["net_pnl"].sum()
            trade_summary = f"Сделок: {len(trs_on_day)} | PnL: {'+' if pnl_day >= 0 else ''}${pnl_day:,.2f}"
        else:
            trade_summary = "0 сделок (без входов)"

        days_meta.append({
            "date": d_str,
            "label": f"{d_str} — {trade_summary}",
            "trades_count": len(trs_on_day),
            "pnl": round(pnl_day, 2)
        })

        # Full 1-minute OHLC & Fair Value (every single minute from 09:30 to 16:00)
        high_vals = df_d["target_high"].values if "target_high" in df_d.columns else df_d["target_close"].values
        low_vals = df_d["target_low"].values if "target_low" in df_d.columns else df_d["target_close"].values

        session_obj = {
            "date": d_str,
            "times": [t.strftime("%H:%M") for t in df_d.index],
            "open": [round(float(v), 2) for v in df_d["target_open"].values],
            "high": [round(float(v), 2) for v in high_vals],
            "low": [round(float(v), 2) for v in low_vals],
            "close": [round(float(v), 2) for v in df_d["target_close"].values],
            "fair": [round(float(v), 2) for v in df_d["target_fair_price"].values],
            "z_score": [round(float(v), 3) for v in df_d["z_score"].values],
            "signals": [str(s) for s in df_d["signal"].values],
            "notes": [str(n) for n in df_d["signal_note"].values]
        }

        # Write lightweight JS file per session (loaded on demand via script tag)
        js_file = sessions_dir / f"{d_str}.js"
        js_file.write_text(f"window.onSessionLoaded('{d_str}', {json.dumps(session_obj)});\n", encoding="utf-8")

    print(f"✅ Generated {len(days_meta)} session files in {sessions_dir}")

    # 5. Build index.html
    html_content = generate_html_page(
        metrics_rows=metrics_rows,
        equity_payload=equity_payload,
        days_meta=days_meta,
        trades_payload=trades_payload
    )

    out_file = project_root / "standalone_report" / "index.html"
    out_file.write_text(html_content, encoding="utf-8")
    print(f"🎉 SUCCESS! Written {out_file} (Size: {out_file.stat().st_size:,} bytes)")


def generate_html_page(metrics_rows, equity_payload, days_meta, trades_payload):
    equity_json_str = json.dumps(equity_payload)
    days_meta_str = json.dumps(days_meta)
    trades_json_str = json.dumps(trades_payload)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Внутридневной Стат-Арбитраж: NVDA vs QQQ (Реальные 1m данные Alpaca)</title>
    <!-- Plotly CDN for High Resolution Interactive Charts -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0B0E14;
            --bg-secondary: #141822;
            --bg-card: #1C2230;
            --accent-blue: #2979FF;
            --accent-green: #00E676;
            --accent-red: #FF5252;
            --accent-yellow: #FFD600;
            --accent-cyan: #00E5FF;
            --text-primary: #FFFFFF;
            --text-muted: #8F9CAE;
            --border-color: #262D3D;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1750px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #141822 0%, #1C2230 100%);
            padding: 24px 32px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{ font-size: 28px; font-weight: 800; }}
        .header p {{ color: var(--text-muted); font-size: 15px; margin-top: 4px; }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 700;
        }}
        .badge-success {{ background-color: rgba(0, 230, 118, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .badge-blue {{ background-color: rgba(41, 121, 255, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
        
        .section-card {{
            background-color: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 24px;
            margin-bottom: 28px;
        }}
        .section-title {{
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .section-subtitle {{ color: var(--text-muted); font-size: 14px; margin-bottom: 20px; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
            text-align: left;
        }}
        th {{
            background-color: var(--bg-card);
            color: var(--text-muted);
            font-weight: 600;
            padding: 14px 16px;
            border-bottom: 2px solid var(--border-color);
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        tr:hover {{ background-color: rgba(255, 255, 255, 0.03); }}
        .highlight-row {{ background-color: rgba(0, 230, 118, 0.08); }}
        .text-green {{ color: var(--accent-green); }}
        .text-red {{ color: var(--accent-red); }}
        .font-bold {{ font-weight: 700; }}

        .chart-box {{
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            position: relative;
        }}
        
        .controls-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 20px;
            background-color: var(--bg-card);
            padding: 16px 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .selector-box {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .selector-box label {{ font-weight: 600; font-size: 15px; }}
        select {{
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--accent-blue);
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            outline: none;
            max-width: 550px;
        }}

        .tf-buttons {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .tf-btn {{
            background-color: var(--bg-secondary);
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .tf-btn:hover {{ color: var(--text-primary); border-color: var(--accent-blue); }}
        .tf-btn.active {{
            background-color: var(--accent-blue);
            color: #FFFFFF;
            border-color: var(--accent-blue);
        }}
        .loading-overlay {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(11, 14, 20, 0.7);
            display: none;
            justify-content: center;
            align-items: center;
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-cyan);
            z-index: 10;
        }}
    </style>
</head>
<body>
    <div class="container">
        
        <!-- HEADER -->
        <div class="header">
            <div>
                <h1>🎯 Внутридневной Стат-Арбитраж (NVDA vs QQQ)</h1>
                <p>100% Реальные 1-минутные данные Alpaca (2024–2026) | Подгрузка сессий на лету из Parquet | 502 торговых дня</p>
            </div>
            <div>
                <span class="badge badge-blue">RTH: 09:30 - 16:00 ET</span>
                <span class="badge badge-success" style="margin-left: 8px;">Single-Leg Mean Reversion</span>
            </div>
        </div>

        <!-- 1. COMPARATIVE EQUITY CURVES SECTION -->
        <div class="section-card">
            <div class="section-title">
                <span>📈 1. Реальный сравнительный график Equity (Все 4 Стратегии)</span>
            </div>
            <div class="section-subtitle">
                Реальная динамика баланса депозита ($100,000 стартовый, $20,000 на сделку, комиссии $0.0035/акцию, проскальзывание 2 bps):
            </div>
            
            <div class="chart-box">
                <div id="equity_chart" style="height: 560px; width: 100%;"></div>
            </div>

            <div style="margin-top: 24px;">
                <div class="section-title" style="font-size: 18px;">📋 Сводная таблица эффективности 4-х режимов</div>
                <table>
                    <thead>
                        <tr>
                            <th>Сценарий</th>
                            <th>Чистая прибыль (PnL)</th>
                            <th>Sharpe Ratio</th>
                            <th>Sortino</th>
                            <th>Max Drawdown</th>
                            <th>Винрейт (%)</th>
                            <th>Profit Factor</th>
                            <th>Всего сделок</th>
                            <th>Комиссии</th>
                            <th>Сбросы (4σ)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {metrics_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 2. DETAILED INTRADAY TRADE INSPECTOR SECTION -->
        <div class="section-card">
            <div class="section-title">
                <span>🔍 2. Инспектор сессий (Настоящие 1-минутные свечи Alpaca, подгрузка на лету)</span>
                <span style="font-size: 14px; font-weight: 500; color: var(--accent-cyan);">⚡ Быстрая подгрузка без забивания памяти</span>
            </div>
            <div class="section-subtitle">
                Выберите любую дату из 502 сессий. Данные загружаются моментально по клику. Вы можете переключать таймфрейм между 1m, 5m и 15m:
            </div>

            <div class="controls-bar">
                <div class="selector-box">
                    <label for="day_select">📅 Выберите торговый день:</label>
                    <select id="day_select" onchange="selectDay(this.value)">
                    </select>
                </div>

                <div class="tf-buttons">
                    <span style="font-weight: 600; font-size: 14px; color: var(--text-muted); margin-right: 6px;">Таймфрейм:</span>
                    <button class="tf-btn active" id="tf_1m" onclick="changeTimeframe('1m')">1 минута (1m)</button>
                    <button class="tf-btn" id="tf_5m" onclick="changeTimeframe('5m')">5 минут (5m)</button>
                    <button class="tf-btn" id="tf_15m" onclick="changeTimeframe('15m')">15 минут (15m)</button>
                </div>
            </div>

            <div class="chart-box">
                <div id="loader_overlay" class="loading-overlay">⏳ Загрузка 1-минутных данных сессии...</div>
                <div id="day_price_chart" style="height: 580px; width: 100%;"></div>
            </div>
            <div class="chart-box" style="margin-top: 14px;">
                <div id="day_zscore_chart" style="height: 320px; width: 100%;"></div>
            </div>
        </div>

        <!-- 3. COMPLETED TRADES TABLE -->
        <div class="section-card">
            <div class="section-title">
                <span>📜 3. Полный реестр совершенных сделок ({len(trades_payload)} сделок)</span>
            </div>
            <div class="section-subtitle">
                Реальный журнал с точным временем входа/выхода, ценами, прибылями и убытками:
            </div>
            <div style="max-height: 550px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 8px;">
                <table id="trades_table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Направление</th>
                            <th>Время входа</th>
                            <th>Цена входа</th>
                            <th>Время выхода</th>
                            <th>Цена выхода</th>
                            <th>Чистый PnL ($)</th>
                            <th>Доходность</th>
                            <th>Длительность</th>
                            <th>Причина закрытия</th>
                            <th>Вход Z</th>
                            <th>Выход Z</th>
                        </tr>
                    </thead>
                    <tbody id="trades_tbody">
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <!-- JAVASCRIPT & ON-DEMAND LOADER -->
    <script>
        const equityData = {equity_json_str};
        const daysMeta = {days_meta_str};
        const tradesData = {trades_json_str};

        // Cache for loaded session objects (in-memory LRU cache)
        const loadedSessionsCache = {{}};
        let currentDayKey = null;
        let currentTimeframe = '1m';

        // 1. RENDER COMPARATIVE EQUITY CHART
        function renderEquityChart() {{
            const traces = [];
            const colorMap = {{
                "A: Pure Reversion (No 4σ caps)": "#8F9CAE",
                "B: Entry Lockout Only (Recommended)": "#00E676",
                "C: Emergency Exit Only": "#FF5252",
                "D: Conservative (Lockout + Exit)": "#2979FF"
            }};
            const widthMap = {{
                "B: Entry Lockout Only (Recommended)": 4,
                "A: Pure Reversion (No 4σ caps)": 2,
                "C: Emergency Exit Only": 2,
                "D: Conservative (Lockout + Exit)": 2
            }};

            for (const [name, vals] of Object.entries(equityData.scenarios)) {{
                traces.push({{
                    x: equityData.timestamps,
                    y: vals,
                    type: 'scatter',
                    mode: 'lines',
                    name: name,
                    line: {{
                        color: colorMap[name] || '#FFFFFF',
                        width: widthMap[name] || 2
                    }}
                }});
            }}

            const layout = {{
                paper_bgcolor: '#1C2230',
                plot_bgcolor: '#1C2230',
                font: {{ color: '#FFFFFF', family: 'sans-serif', size: 14 }},
                xaxis: {{ gridcolor: '#262D3D', title: 'Дата и время', showgrid: true }},
                yaxis: {{ gridcolor: '#262D3D', title: 'Баланс счета ($)', showgrid: true, tickformat: '$,.0f' }},
                margin: {{ l: 75, r: 40, t: 30, b: 60 }},
                legend: {{ orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center', font: {{ size: 13, weight: 'bold' }} }},
                hovermode: 'x unified'
            }};

            Plotly.newPlot('equity_chart', traces, layout, {{ responsive: true, displayModeBar: true }});
        }}

        // 2. POPULATE DAY SELECTOR
        function initDaySelector() {{
            const select = document.getElementById('day_select');
            select.innerHTML = '';
            
            daysMeta.forEach((item, idx) => {{
                const opt = document.createElement('option');
                opt.value = item.date;
                opt.textContent = `${{idx + 1}}. ${{item.label}}`;
                select.appendChild(opt);
            }});

            if (daysMeta.length > 0) {{
                // Find a day with trades to display first
                const firstTradeDay = daysMeta.find(d => d.trades_count > 0) || daysMeta[0];
                select.value = firstTradeDay.date;
                selectDay(firstTradeDay.date);
            }}
        }}

        // 3. ON-DEMAND SESSION LOADER (LOADS FROM sessions/{date}.js VIA SCRIPT TAG OR FETCH)
        function selectDay(dateStr) {{
            currentDayKey = dateStr;
            if (loadedSessionsCache[dateStr]) {{
                renderSessionChart(loadedSessionsCache[dateStr]);
                return;
            }}

            // Show loading indicator
            const overlay = document.getElementById('loader_overlay');
            overlay.style.display = 'flex';

            // Dynamic script tag injection (works natively on file:/// without CORS issues)
            const script = document.createElement('script');
            script.src = `sessions/${{dateStr}}.js`;
            script.onerror = () => {{
                overlay.style.display = 'none';
                alert(`Не удалось загрузить файл sessions/${{dateStr}}.js`);
            }};
            document.body.appendChild(script);
        }}

        // Global callback invoked by sessions/{date}.js
        window.onSessionLoaded = function(dateStr, sessionData) {{
            loadedSessionsCache[dateStr] = sessionData;
            document.getElementById('loader_overlay').style.display = 'none';
            if (currentDayKey === dateStr) {{
                renderSessionChart(sessionData);
            }}
        }};

        // 4. TIMEFRAME SWITCHER (1m, 5m, 15m)
        function changeTimeframe(tf) {{
            currentTimeframe = tf;
            document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`tf_${{tf}}`).classList.add('active');
            
            if (currentDayKey && loadedSessionsCache[currentDayKey]) {{
                renderSessionChart(loadedSessionsCache[currentDayKey]);
            }}
        }}

        // 5. RESAMPLE 1M DATA TO 5M OR 15M IN BROWSER ON THE FLY
        function resampleSession(raw1m, stepMinutes) {{
            if (stepMinutes <= 1) return raw1m;

            const resTimes = [];
            const resOpen = [];
            const resHigh = [];
            const resLow = [];
            const resClose = [];
            const resFair = [];
            const resZ = [];
            const resSignals = [];
            const resNotes = [];

            for (let i = 0; i < raw1m.times.length; i += stepMinutes) {{
                const chunkEnd = Math.min(i + stepMinutes, raw1m.times.length);
                const tSlice = raw1m.times.slice(i, chunkEnd);
                const opSlice = raw1m.open.slice(i, chunkEnd);
                const hiSlice = raw1m.high.slice(i, chunkEnd);
                const loSlice = raw1m.low.slice(i, chunkEnd);
                const clSlice = raw1m.close.slice(i, chunkEnd);
                const fairSlice = raw1m.fair.slice(i, chunkEnd);
                const zSlice = raw1m.z_score.slice(i, chunkEnd);
                const sigSlice = raw1m.signals.slice(i, chunkEnd);
                const noteSlice = raw1m.notes.slice(i, chunkEnd);

                resTimes.push(tSlice[0]);
                resOpen.push(opSlice[0]);
                resHigh.push(Math.max(...hiSlice));
                resLow.push(Math.min(...loSlice));
                resClose.push(clSlice[clSlice.length - 1]);
                resFair.push(fairSlice[fairSlice.length - 1]);
                resZ.push(zSlice[zSlice.length - 1]);

                // Propagate any trade signals within this bar
                const foundSig = sigSlice.find(s => s !== 'NONE') || 'NONE';
                const foundNote = noteSlice.find(n => n && n !== 'None') || '';
                resSignals.push(foundSig);
                resNotes.push(foundNote);
            }}

            return {{
                date: raw1m.date,
                times: resTimes,
                open: resOpen,
                high: resHigh,
                low: resLow,
                close: resClose,
                fair: resFair,
                z_score: resZ,
                signals: resSignals,
                notes: resNotes
            }};
        }}

        // 6. RENDER INTRADAY PRICE & Z-SCORE CHART
        function renderSessionChart(raw1mData) {{
            const stepMap = {{ '1m': 1, '5m': 5, '15m': 15 }};
            const data = resampleSession(raw1mData, stepMap[currentTimeframe] || 1);

            const buyIdx = [];
            const shortIdx = [];
            const exitIdx = [];

            for (let i = 0; i < data.signals.length; i++) {{
                const sig = data.signals[i];
                if (sig === 'BUY_LONG') buyIdx.push(i);
                else if (sig === 'SELL_SHORT') shortIdx.push(i);
                else if (sig.startsWith('EXIT_')) exitIdx.push(i);
            }}

            const priceTraces = [
                {{
                    x: data.times,
                    open: data.open,
                    high: data.high,
                    low: data.low,
                    close: data.close,
                    type: 'candlestick',
                    name: `NVDA Свечи (${{currentTimeframe}})`,
                    increasing: {{ line: {{ color: '#00E676', width: 1.5 }} }},
                    decreasing: {{ line: {{ color: '#FF5252', width: 1.5 }} }}
                }},
                {{
                    x: data.times,
                    y: data.fair,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Справедливая цена (QQQ * Beta)',
                    line: {{ color: '#FFD600', width: 2.5, dash: 'dash' }}
                }}
            ];

            if (buyIdx.length > 0) {{
                priceTraces.push({{
                    x: buyIdx.map(i => data.times[i]),
                    y: buyIdx.map(i => data.close[i]),
                    mode: 'markers+text',
                    type: 'scatter',
                    name: '🟢 ВХОД LONG',
                    text: buyIdx.map(i => '▲ BUY'),
                    textposition: 'bottom center',
                    textfont: {{ color: '#00E676', size: 15, weight: 'bold' }},
                    marker: {{ symbol: 'triangle-up', size: 22, color: '#00E676' }}
                }});
            }}

            if (shortIdx.length > 0) {{
                priceTraces.push({{
                    x: shortIdx.map(i => data.times[i]),
                    y: shortIdx.map(i => data.close[i]),
                    mode: 'markers+text',
                    type: 'scatter',
                    name: '🔴 ВХОД SHORT',
                    text: shortIdx.map(i => '▼ SHORT'),
                    textposition: 'top center',
                    textfont: {{ color: '#FF5252', size: 15, weight: 'bold' }},
                    marker: {{ symbol: 'triangle-down', size: 22, color: '#FF5252' }}
                }});
            }}

            if (exitIdx.length > 0) {{
                priceTraces.push({{
                    x: exitIdx.map(i => data.times[i]),
                    y: exitIdx.map(i => data.close[i]),
                    mode: 'markers+text',
                    type: 'scatter',
                    name: '✖ ВЫХОД',
                    text: exitIdx.map(i => data.signals[i] === 'EXIT_FORCED_EOD' ? '✖ EOD ВЫХОД' : '✖ TAKE PROFIT'),
                    textposition: 'top right',
                    textfont: {{ color: '#FFD600', size: 14, weight: 'bold' }},
                    marker: {{ symbol: 'x', size: 18, color: '#FFD600' }}
                }});
            }}

            const priceLayout = {{
                title: `📊 Сессия ${{data.date}} | Реальные свечи NVDA (${{currentTimeframe}}, ${{data.times.length}} баров) vs Справедливая цена`,
                paper_bgcolor: '#1C2230',
                plot_bgcolor: '#1C2230',
                font: {{ color: '#FFFFFF', size: 14 }},
                xaxis: {{ gridcolor: '#262D3D', rangeslider: {{ visible: false }} }},
                yaxis: {{ gridcolor: '#262D3D', title: 'Цена NVDA ($)', tickformat: '$,.2f' }},
                margin: {{ l: 75, r: 40, t: 50, b: 40 }},
                legend: {{ orientation: 'h', y: 1.06, x: 0.5, xanchor: 'center' }},
                hovermode: 'x unified'
            }};

            Plotly.newPlot('day_price_chart', priceTraces, priceLayout, {{ responsive: true }});

            const zTraces = [
                {{
                    x: data.times,
                    y: data.z_score,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Z-Score расхождения',
                    line: {{ color: '#00E5FF', width: 2.5 }}
                }}
            ];

            const zLayout = {{
                title: `📐 Z-Score Отклонения и уровни 4σ (${{data.date}})`,
                paper_bgcolor: '#1C2230',
                plot_bgcolor: '#1C2230',
                font: {{ color: '#FFFFFF', size: 13 }},
                xaxis: {{ gridcolor: '#262D3D', title: 'Время (Нью-Йорк, ET)' }},
                yaxis: {{ gridcolor: '#262D3D', title: 'Z-Score (σ)' }},
                margin: {{ l: 75, r: 40, t: 50, b: 50 }},
                shapes: [
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: 1.5, y1: 1.5, line: {{ color: '#FF5252', width: 1.5, dash: 'dot' }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: -1.5, y1: -1.5, line: {{ color: '#00E676', width: 1.5, dash: 'dot' }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: 0, y1: 0, line: {{ color: '#8F9CAE', width: 1 }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: 4.0, y1: 4.0, line: {{ color: '#D50000', width: 2, dash: 'dash' }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: -4.0, y1: -4.0, line: {{ color: '#D50000', width: 2, dash: 'dash' }} }}
                ],
                annotations: [
                    {{ x: data.times[1], y: 1.6, text: 'Вход SHORT (+1.5σ)', showarrow: false, font: {{ color: '#FF5252', size: 12 }} }},
                    {{ x: data.times[1], y: -1.6, text: 'Вход LONG (-1.5σ)', showarrow: false, font: {{ color: '#00E676', size: 12 }} }},
                    {{ x: data.times[1], y: 4.2, text: '⛔ БЛОКИРОВКА 4σ', showarrow: false, font: {{ color: '#D50000', size: 12, weight: 'bold' }} }},
                    {{ x: data.times[1], y: -4.2, text: '⛔ БЛОКИРОВКА 4σ', showarrow: false, font: {{ color: '#D50000', size: 12, weight: 'bold' }} }}
                ],
                hovermode: 'x unified'
            }};

            Plotly.newPlot('day_zscore_chart', zTraces, zLayout, {{ responsive: true }});
        }}

        // 7. RENDER TRADES TABLE
        function renderTradesTable() {{
            const tbody = document.getElementById('trades_tbody');
            tbody.innerHTML = '';

            tradesData.forEach(tr => {{
                const trEl = document.createElement('tr');
                const isWin = tr.pnl >= 0;
                trEl.innerHTML = `
                    <td><strong>#${{tr.id}}</strong></td>
                    <td><strong>${{tr.dir}}</strong></td>
                    <td>${{tr.entry_time}}</td>
                    <td>${{tr.entry_price}}</td>
                    <td>${{tr.exit_time}}</td>
                    <td>${{tr.exit_price}}</td>
                    <td class="${{isWin ? 'text-green' : 'text-red'}} font-bold">${{tr.pnl_str}}</td>
                    <td class="${{isWin ? 'text-green' : 'text-red'}} font-bold">${{tr.return_pct}}</td>
                    <td>${{tr.duration}}</td>
                    <td><span class="badge ${{tr.reason === 'TAKE_PROFIT' ? 'badge-success' : 'badge-blue'}}">${{tr.reason}}</span></td>
                    <td>${{tr.entry_z}}σ</td>
                    <td>${{tr.exit_z}}σ</td>
                `;
                tbody.appendChild(trEl);
            }});
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            renderEquityChart();
            initDaySelector();
            renderTradesTable();
        }});
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    build_all_sessions()
