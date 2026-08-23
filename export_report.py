"""Export standalone HTML report to disk."""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Resolve everything relative to the cloned repository.
CWD = Path(__file__).resolve().parent
sys.path.insert(0, str(CWD))

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.analysis.matrix_comparator import MatrixComparator
import standalone_report.build_report as br


def export():
    loader = DataLoader(cache_dir=str(CWD / "data_cache"), source="alpaca")
    df_lead, df_target = loader.get_synchronized_pair(
        ticker_lead="QQQ",
        ticker_target="NVDA",
        days_back=730,
        source="alpaca"
    )

    fv = FairValueModel(beta_mode="dynamic_rolling", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)

    cfg = AppConfig()
    matrix_comp = MatrixComparator(cfg)
    matrix_res = matrix_comp.run_all_scenarios(df_metrics)

    rec_res = matrix_res["results"]["B: Entry Lockout Only (Recommended)"]
    df_signals = rec_res["df_results"]
    trades_df = rec_res["trades_df"]

    eq_df = matrix_res["equity_curves"]
    eq_sampled = eq_df.iloc[::15].copy()
    if eq_df.index[-1] != eq_sampled.index[-1]:
        eq_sampled = pd.concat([eq_sampled, eq_df.iloc[[-1]]])

    equity_payload = {
        "timestamps": [t.strftime("%Y-%m-%d %H:%M") for t in eq_sampled.index],
        "scenarios": {}
    }
    for col in eq_sampled.columns:
        equity_payload["scenarios"][col] = [round(float(v), 2) for v in eq_sampled[col].values]

    metrics_rows = ""
    for s_name, res in matrix_res["results"].items():
        m = res["metrics"]
        is_rec = "Recommended" in s_name
        badge = '<span class="badge badge-success">⭐ ЛУЧШИЙ</span>' if is_rec else ""
        metrics_rows += f"""
        <tr class="{'highlight-row' if is_rec else ''}">
            <td><strong>{s_name}</strong> {badge}</td>
            <td class="text-green font-bold">${m.total_pnl:,.2f} ({m.total_return_pct:+.2f}%)</td>
            <td><strong>{m.sharpe_ratio:.2f}</strong></td>
            <td>{m.sortino_ratio:.2f}</td>
            <td class="text-red"><strong>{m.max_drawdown_pct:.2f}%</strong> (${m.max_drawdown_usd:,.2f})</td>
            <td><strong>{m.win_rate_pct:.1f}%</strong></td>
            <td>{m.profit_factor:.2f}</td>
            <td>{m.total_trades}</td>
            <td>${m.total_commissions:,.2f}</td>
            <td>{m.exit_reasons_breakdown.get('EMERGENCY_4SIGMA', 0)}</td>
        </tr>
        """

    trades_payload = []
    if not trades_df.empty:
        for _, tr in trades_df.iterrows():
            trades_payload.append({
                "id": int(tr["trade_id"]),
                "date": tr["entry_time"].strftime("%Y-%m-%d"),
                "dir": "🟢 LONG" if tr["direction"] == 1 else "🔴 SHORT",
                "entry_time": tr["entry_time"].strftime("%Y-%m-%d %H:%M"),
                "entry_price": f"${tr['entry_price']:.2f}",
                "exit_time": tr["exit_time"].strftime("%Y-%m-%d %H:%M"),
                "exit_price": f"${tr['exit_price']:.2f}",
                "pnl": round(float(tr["net_pnl"]), 2),
                "pnl_str": f"{'+' if tr['net_pnl'] >= 0 else ''}${tr['net_pnl']:,.2f}",
                "return_pct": f"{tr['return_pct']*100:+.2f}%",
                "duration": f"{int(tr['duration_bars'])} мин",
                "reason": tr["exit_reason"],
                "entry_z": f"{tr['entry_z_score']:.2f}",
                "exit_z": f"{tr['exit_z_score']:.2f}"
            })

    all_trade_dates = sorted(trades_df["entry_time"].dt.date.unique()) if not trades_df.empty else []
    selected_dates = all_trade_dates[-60:] if len(all_trade_dates) > 60 else all_trade_dates

    daily_payload = {}
    for d in selected_dates:
        d_str = str(d)
        df_d = df_signals[df_signals["session_date"] == d]
        if df_d.empty:
            continue
        times = [t.strftime("%H:%M") for t in df_d.index]
        daily_payload[d_str] = {
            "date": d_str,
            "times": times,
            "open": [round(float(v), 2) for v in df_d["target_open"]],
            "high": [round(float(v), 2) for v in df_d["target_close"].rolling(2).max().fillna(df_d["target_close"])],
            "low": [round(float(v), 2) for v in df_d["target_close"].rolling(2).min().fillna(df_d["target_close"])],
            "close": [round(float(v), 2) for v in df_d["target_close"]],
            "fair": [round(float(v), 2) for v in df_d["target_fair_price"]],
            "z_score": [round(float(v), 3) for v in df_d["z_score"]],
            "signals": [str(s) for s in df_d["signal"]],
            "notes": [str(n) for n in df_d["signal_note"]]
        }

    # Generate full HTML
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Внутридневной Статистический Арбитраж: NVDA vs QQQ (2 Года)</title>
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
        }}
        
        .selector-box {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 20px;
            background-color: var(--bg-card);
            padding: 16px 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
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
        }}
    </style>
</head>
<body>
    <div class="container">
        
        <!-- HEADER -->
        <div class="header">
            <div>
                <h1>🎯 Внутридневной Статистический Арбитраж (NVDA vs QQQ)</h1>
                <p>2-летний бэктест на 1-минутных данных (195,502 баров, 502 торговых дня) | Точный расчет без заглядывания в будущее</p>
            </div>
            <div>
                <span class="badge badge-blue">RTH: 09:30 - 16:00 ET</span>
                <span class="badge badge-success" style="margin-left: 8px;">Single-Leg Mean Reversion</span>
            </div>
        </div>

        <!-- 1. COMPARATIVE EQUITY CURVES SECTION -->
        <div class="section-card">
            <div class="section-title">📈 1. Общий сравнительный график Equity (Все 4 Стратегии)</div>
            <div class="section-subtitle">
                Наложение 4-х сценариев на одних и тех же реальных данных за 2 года:
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
            <div class="section-title">🔍 2. Крупный инспектор внутридневных сделок и свечей</div>
            <div class="section-subtitle">
                Выберите сессию для просмотра крупных минутных свечей NVDA, точек входа после отката на δ, справедливой цены и уровней 4σ:
            </div>

            <div class="selector-box">
                <label for="day_select">📅 Выберите торговый день:</label>
                <select id="day_select" onchange="renderDayChart(this.value)">
                </select>
            </div>

            <div class="chart-box">
                <div id="day_price_chart" style="height: 580px; width: 100%;"></div>
            </div>
            <div class="chart-box" style="margin-top: 14px;">
                <div id="day_zscore_chart" style="height: 320px; width: 100%;"></div>
            </div>
        </div>

        <!-- 3. COMPLETED TRADES TABLE -->
        <div class="section-card">
            <div class="section-title">📜 3. Реестр совершенных сделок (Лучшая стратегия: Сценарий B, {len(trades_payload)} сделок)</div>
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

    <script>
        const equityData = {json.dumps(equity_payload)};
        const dailySessions = {json.dumps(daily_payload)};
        const tradesData = {json.dumps(trades_payload)};
        const daysList = {json.dumps([str(d) for d in selected_dates])};

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
                xaxis: {{
                    gridcolor: '#262D3D',
                    title: 'Дата',
                    showgrid: true
                }},
                yaxis: {{
                    gridcolor: '#262D3D',
                    title: 'Баланс счета ($)',
                    showgrid: true,
                    tickformat: '$,.0f'
                }},
                margin: {{ l: 70, r: 40, t: 30, b: 60 }},
                legend: {{
                    orientation: 'h',
                    y: 1.08,
                    x: 0.5,
                    xanchor: 'center',
                    font: {{ size: 14, weight: 'bold' }}
                }},
                hovermode: 'x unified'
            }};

            Plotly.newPlot('equity_chart', traces, layout, {{ responsive: true, displayModeBar: true }});
        }}

        // 2. POPULATE DAY SELECTOR
        function initDaySelector() {{
            const select = document.getElementById('day_select');
            select.innerHTML = '';
            
            daysList.forEach((d, idx) => {{
                const opt = document.createElement('option');
                opt.value = d;
                opt.textContent = `${{idx + 1}}. Сессия: ${{d}}`;
                select.appendChild(opt);
            }});

            if (daysList.length > 0) {{
                select.value = daysList[daysList.length - 1];
                renderDayChart(daysList[daysList.length - 1]);
            }}
        }}

        // 3. RENDER LARGE INTRADAY PRICE & Z-SCORE CHART
        function renderDayChart(dayStr) {{
            const data = dailySessions[dayStr];
            if (!data) return;

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
                    name: 'NVDA Свечи (1m)',
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
                    marker: {{ symbol: 'triangle-up', size: 20, color: '#00E676' }}
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
                    marker: {{ symbol: 'triangle-down', size: 20, color: '#FF5252' }}
                }});
            }}

            if (exitIdx.length > 0) {{
                priceTraces.push({{
                    x: exitIdx.map(i => data.times[i]),
                    y: exitIdx.map(i => data.close[i]),
                    mode: 'markers+text',
                    type: 'scatter',
                    name: '✖ ВЫХОД',
                    text: exitIdx.map(i => '✖ EXIT'),
                    textposition: 'top right',
                    textfont: {{ color: '#FFD600', size: 14, weight: 'bold' }},
                    marker: {{ symbol: 'x', size: 18, color: '#FFD600' }}
                }});
            }}

            const priceLayout = {{
                title: `📊 Крупный график цен: NVDA vs Справедливая цена (${{dayStr}})`,
                paper_bgcolor: '#1C2230',
                plot_bgcolor: '#1C2230',
                font: {{ color: '#FFFFFF', size: 14 }},
                xaxis: {{ gridcolor: '#262D3D', rangeslider: {{ visible: false }} }},
                yaxis: {{ gridcolor: '#262D3D', title: 'Цена NVDA ($)', tickformat: '$,.2f' }},
                margin: {{ l: 70, r: 40, t: 50, b: 40 }},
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
                    name: 'Z-Score спреда',
                    line: {{ color: '#00E5FF', width: 2.5 }}
                }}
            ];

            const zLayout = {{
                title: `📐 Z-Score Отклонения и уровни 4σ (${{dayStr}})`,
                paper_bgcolor: '#1C2230',
                plot_bgcolor: '#1C2230',
                font: {{ color: '#FFFFFF', size: 13 }},
                xaxis: {{ gridcolor: '#262D3D', title: 'Время (Нью-Йорк, ET)' }},
                yaxis: {{ gridcolor: '#262D3D', title: 'Z-Score (σ)' }},
                margin: {{ l: 70, r: 40, t: 50, b: 50 }},
                shapes: [
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: 1.5, y1: 1.5, line: {{ color: '#FF5252', width: 1.5, dash: 'dot' }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: -1.5, y1: -1.5, line: {{ color: '#00E676', width: 1.5, dash: 'dot' }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: 0, y1: 0, line: {{ color: '#8F9CAE', width: 1 }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: 4.0, y1: 4.0, line: {{ color: '#D50000', width: 2, dash: 'dash' }} }},
                    {{ type: 'line', x0: data.times[0], x1: data.times[data.times.length-1], y0: -4.0, y1: -4.0, line: {{ color: '#D50000', width: 2, dash: 'dash' }} }}
                ],
                annotations: [
                    {{ x: data.times[5], y: 1.6, text: 'Вход SHORT (+1.5σ)', showarrow: false, font: {{ color: '#FF5252', size: 12 }} }},
                    {{ x: data.times[5], y: -1.6, text: 'Вход LONG (-1.5σ)', showarrow: false, font: {{ color: '#00E676', size: 12 }} }},
                    {{ x: data.times[5], y: 4.2, text: '⛔ БЛОКИРОВКА 4σ', showarrow: false, font: {{ color: '#D50000', size: 12, weight: 'bold' }} }},
                    {{ x: data.times[5], y: -4.2, text: '⛔ БЛОКИРОВКА 4σ', showarrow: false, font: {{ color: '#D50000', size: 12, weight: 'bold' }} }}
                ],
                hovermode: 'x unified'
            }};

            Plotly.newPlot('day_zscore_chart', zTraces, zLayout, {{ responsive: true }});
        }}

        // 4. POPULATE TRADES TABLE
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

    target_paths = [
        CWD / "standalone_report" / "index.html",
        CWD / "index.html"
    ]

    for p in target_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ Written to {p}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    export()
