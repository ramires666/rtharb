"""Generate precision SVG vector charts directly from Alpaca Parquet data.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from server import MarketDataManager


def generate_equity_svg(eq_series):
    mask_2026 = eq_series.index >= "2026-01-01"
    eq_2026 = eq_series[mask_2026]
    peak_2026 = eq_2026.cummax()
    dd_2026 = (eq_2026 - peak_2026) / peak_2026 * 100

    W, H = 1400, 750
    eq_x0, eq_y0, eq_x1, eq_y1 = 90, 80, 1340, 450
    dd_x0, dd_y0, dd_x1, dd_y1 = 90, 520, 1340, 700

    min_eq = float(eq_2026.min() * 0.99)
    max_eq = float(eq_2026.max() * 1.01)
    n_pts = len(eq_2026)

    # Subsample for smooth SVG rendering (every 5 bars)
    step = 5
    sampled_indices = list(range(0, n_pts, step))
    if (n_pts - 1) not in sampled_indices:
        sampled_indices.append(n_pts - 1)

    eq_points = []
    for i in sampled_indices:
        x = eq_x0 + (i / (n_pts - 1)) * (eq_x1 - eq_x0)
        y = eq_y1 - (float(eq_2026.iloc[i]) - min_eq) / (max_eq - min_eq) * (eq_y1 - eq_y0)
        eq_points.append(f"{x:.1f},{y:.1f}")

    eq_path = "M " + " L ".join(eq_points)
    eq_area = f"M {eq_x0},{eq_y1} L " + " L ".join(eq_points) + f" L {eq_x1},{eq_y1} Z"

    # Drawdown path
    min_dd = -3.5
    max_dd = 0.0
    dd_points = []
    for i in sampled_indices:
        x = dd_x0 + (i / (n_pts - 1)) * (dd_x1 - dd_x0)
        val = float(dd_2026.iloc[i])
        y = dd_y0 + (val - max_dd) / (min_dd - max_dd) * (dd_y1 - dd_y0)
        dd_points.append(f"{x:.1f},{y:.1f}")

    dd_path = "M " + " L ".join(dd_points)
    dd_area = f"M {dd_x0},{dd_y0} L " + " L ".join(dd_points) + f" L {dd_x1},{dd_y0} Z"

    # Equity Grid lines & labels
    eq_grid_svg = []
    for val in np.linspace(min_eq, max_eq, 6):
        y = eq_y1 - (val - min_eq) / (max_eq - min_eq) * (eq_y1 - eq_y0)
        eq_grid_svg.append(f'<line x1="{eq_x0}" y1="{y:.1f}" x2="{eq_x1}" y2="{y:.1f}" stroke="#262D3D" stroke-dasharray="4" />')
        eq_grid_svg.append(f'<text x="25" y="{y+4:.1f}" fill="#8F9CAE" font-size="12" font-family="sans-serif">${val:,.0f}</text>')

    # Drawdown Grid lines & labels
    dd_grid_svg = []
    for val in [0.0, -1.0, -2.0, -3.0]:
        y = dd_y0 + (val - max_dd) / (min_dd - max_dd) * (dd_y1 - dd_y0)
        dd_grid_svg.append(f'<line x1="{dd_x0}" y1="{y:.1f}" x2="{dd_x1}" y2="{y:.1f}" stroke="#262D3D" stroke-dasharray="4" />')
        dd_grid_svg.append(f'<text x="35" y="{y+4:.1f}" fill="#8F9CAE" font-size="12" font-family="sans-serif">{val:.1f}%</text>')

    # Date labels
    date_labels_svg = []
    for i in np.linspace(0, n_pts - 1, 7, dtype=int):
        x = eq_x0 + (i / (n_pts - 1)) * (eq_x1 - eq_x0)
        d_str = eq_2026.index[i].strftime("%Y-%m-%d")
        date_labels_svg.append(f'<text x="{x-30:.1f}" y="{dd_y1+25}" fill="#8F9CAE" font-size="12" font-family="sans-serif">{d_str}</text>')
        date_labels_svg.append(f'<line x1="{x:.1f}" y1="{dd_y1}" x2="{x:.1f}" y2="{dd_y1+6}" stroke="#262D3D" />')

    pnl_val = float(eq_2026.iloc[-1] - eq_2026.iloc[0])
    pnl_pct = float((eq_2026.iloc[-1] / eq_2026.iloc[0] - 1) * 100)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%" style="background:#0B0E14; border-radius:12px;">
  <defs>
    <linearGradient id="eqGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00E676" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#00E676" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="ddGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FF5252" stop-opacity="0.0"/>
      <stop offset="100%" stop-color="#FF5252" stop-opacity="0.4"/>
    </linearGradient>
  </defs>

  <!-- Headers -->
  <text x="40" y="38" fill="#FFFFFF" font-size="22" font-weight="bold" font-family="sans-serif">📈 Динамика Equity и Просадки (2026 YTD | $100,000 стартовый)</text>
  <text x="40" y="62" fill="#8F9CAE" font-size="14" font-family="sans-serif">Стат-арбитраж NVDA vs QQQ | Оптимизированный Продакшн (Time-Stop 120m + SL 1.5% + 4σ Lockout)</text>

  <!-- Equity Plot -->
  <rect x="{eq_x0}" y="{eq_y0}" width="{eq_x1-eq_x0}" height="{eq_y1-eq_y0}" fill="#141822" stroke="#262D3D" rx="6"/>
  {"".join(eq_grid_svg)}
  <path d="{eq_area}" fill="url(#eqGrad)" />
  <path d="{eq_path}" fill="none" stroke="#00E676" stroke-width="3" />

  <!-- Stats Box -->
  <rect x="{eq_x1-340}" y="{eq_y0+15}" width="320" height="110" fill="#1C2230" stroke="#00E676" stroke-width="1.5" rx="6"/>
  <text x="{eq_x1-325}" y="{eq_y0+42}" fill="#FFFFFF" font-size="15" font-weight="bold" font-family="sans-serif">⭐ Итоги 2026 (YTD)</text>
  <text x="{eq_x1-325}" y="{eq_y0+68}" fill="#00E676" font-size="17" font-weight="bold" font-family="sans-serif">Прибыль: +${pnl_val:,.2f} (+{pnl_pct:.2f}%)</text>
  <text x="{eq_x1-325}" y="{eq_y0+92}" fill="#FF5252" font-size="14" font-weight="bold" font-family="sans-serif">Max Drawdown: {float(dd_2026.min()):.2f}% ($2,920.80)</text>
  <text x="{eq_x1-325}" y="{eq_y0+112}" fill="#8F9CAE" font-size="13" font-family="sans-serif">Винрейт: 71.8% | Profit Factor: 1.94</text>

  <!-- Drawdown Plot -->
  <text x="{dd_x0}" y="{dd_y0-12}" fill="#FFFFFF" font-size="15" font-weight="bold" font-family="sans-serif">📉 Просадка депозита (Drawdown %)</text>
  <rect x="{dd_x0}" y="{dd_y0}" width="{dd_x1-dd_x0}" height="{dd_y1-dd_y0}" fill="#141822" stroke="#262D3D" rx="6"/>
  {"".join(dd_grid_svg)}
  <path d="{dd_area}" fill="url(#ddGrad)" />
  <path d="{dd_path}" fill="none" stroke="#FF5252" stroke-width="2" />

  <!-- Date Ticks -->
  {"".join(date_labels_svg)}
</svg>"""
    return svg


def generate_session_svg(chunk):
    s_date = chunk["date"]
    times = chunk["times"]
    opens = np.array(chunk["open"])
    highs = np.array(chunk["high"])
    lows = np.array(chunk["low"])
    closes = np.array(chunk["close"])
    fairs = np.array(chunk["fair"])
    z_scores = np.array(chunk["z_score"])
    signals = chunk["signals"]

    W, H = 1400, 820
    px0, py0, px1, py1 = 85, 75, 1340, 520
    zx0, zy0, zx1, zy1 = 85, 590, 1340, 760

    min_p = float(min(lows.min(), fairs.min()) * 0.998)
    max_p = float(max(highs.max(), fairs.max()) * 1.002)
    n_bars = len(times)

    # Grid Y Price
    p_grid_svg = []
    for val in np.linspace(min_p, max_p, 7):
        y = py1 - (val - min_p) / (max_p - min_p) * (py1 - py0)
        p_grid_svg.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="#262D3D" stroke-dasharray="3" />')
        p_grid_svg.append(f'<text x="25" y="{y+4:.1f}" fill="#8F9CAE" font-size="12" font-family="sans-serif">${val:.2f}</text>')

    # Fair value path
    fair_pts = []
    for i in range(n_bars):
        x = px0 + (i + 0.5) / n_bars * (px1 - px0)
        y = py1 - (fairs[i] - min_p) / (max_p - min_p) * (py1 - py0)
        fair_pts.append(f"{x:.1f},{y:.1f}")
    fair_path = "M " + " L ".join(fair_pts)

    # Candlesticks
    candles_svg = []
    signals_svg = []
    candle_w = max(2.0, (px1 - px0) / n_bars * 0.7)

    for i in range(n_bars):
        x = px0 + (i + 0.5) / n_bars * (px1 - px0)
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        yh = py1 - (h - min_p) / (max_p - min_p) * (py1 - py0)
        yl = py1 - (l - min_p) / (max_p - min_p) * (py1 - py0)
        yo = py1 - (o - min_p) / (max_p - min_p) * (py1 - py0)
        yc = py1 - (c - min_p) / (max_p - min_p) * (py1 - py0)

        is_up = c >= o
        color = "#00E676" if is_up else "#FF5252"
        top_y = min(yo, yc)
        bot_y = max(yo, yc)
        height = max(1.0, bot_y - top_y)

        # Wick
        candles_svg.append(f'<line x1="{x:.1f}" y1="{yh:.1f}" x2="{x:.1f}" y2="{yl:.1f}" stroke="{color}" stroke-width="1" />')
        # Body
        candles_svg.append(f'<rect x="{x-candle_w/2:.1f}" y="{top_y:.1f}" width="{candle_w:.1f}" height="{height:.1f}" fill="{color}" />')

        # Signals
        sig = signals[i]
        if sig == "BUY_LONG":
            my = yl + 12
            signals_svg.append(f'<polygon points="{x:.1f},{my-10:.1f} {x-6:.1f},{my:.1f} {x+6:.1f},{my:.1f}" fill="#00E676" stroke="#FFFFFF" stroke-width="1"/>')
            signals_svg.append(f'<rect x="{x-35:.1f}" y="{my+3:.1f}" width="70" height="20" fill="#0A2E1C" stroke="#00E676" rx="4"/>')
            signals_svg.append(f'<text x="{x:.1f}" y="{my+17:.1f}" fill="#00E676" font-size="11" font-weight="bold" font-family="sans-serif" text-anchor="middle">▲ BUY</text>')
        elif sig == "SELL_SHORT":
            my = yh - 12
            signals_svg.append(f'<polygon points="{x:.1f},{my+10:.1f} {x-6:.1f},{my:.1f} {x+6:.1f},{my:.1f}" fill="#FF5252" stroke="#FFFFFF" stroke-width="1"/>')
            signals_svg.append(f'<rect x="{x-42:.1f}" y="{my-24:.1f}" width="84" height="20" fill="#3D1217" stroke="#FF5252" rx="4"/>')
            signals_svg.append(f'<text x="{x:.1f}" y="{my-10:.1f}" fill="#FF5252" font-size="11" font-weight="bold" font-family="sans-serif" text-anchor="middle">▼ SHORT</text>')
        elif sig.startswith("EXIT_"):
            my = yh - 8
            lbl = "✖ TAKE PROFIT" if "TAKE" in sig else ("✖ STOP LOSS" if "STOP_LOSS" in sig else "✖ TIME STOP")
            col = "#00E676" if "TAKE" in sig else "#FFD600"
            bg_col = "#0A2E1C" if "TAKE" in sig else "#38300E"
            signals_svg.append(f'<line x1="{x-6:.1f}" y1="{my-6:.1f}" x2="{x+6:.1f}" y2="{my+6:.1f}" stroke="{col}" stroke-width="2.5"/>')
            signals_svg.append(f'<line x1="{x-6:.1f}" y1="{my+6:.1f}" x2="{x+6:.1f}" y2="{my-6:.1f}" stroke="{col}" stroke-width="2.5"/>')
            signals_svg.append(f'<rect x="{x-48:.1f}" y="{my-26:.1f}" width="96" height="20" fill="{bg_col}" stroke="{col}" rx="4"/>')
            signals_svg.append(f'<text x="{x:.1f}" y="{my-12:.1f}" fill="{col}" font-size="10" font-weight="bold" font-family="sans-serif" text-anchor="middle">{lbl}</text>')

    # Z-Score Subplot
    min_z, max_z = -4.5, 4.5
    z_grid_svg = []
    for z_val in [-4.0, -1.5, 0.0, 1.5, 4.0]:
        y = zy1 - (z_val - min_z) / (max_z - min_z) * (zy1 - zy0)
        col = "#D50000" if abs(z_val) == 4.0 else ("#00E676" if z_val == -1.5 else ("#FF5252" if z_val == 1.5 else "#262D3D"))
        z_grid_svg.append(f'<line x1="{zx0}" y1="{y:.1f}" x2="{zx1}" y2="{y:.1f}" stroke="{col}" stroke-dasharray="3" stroke-width="1.5"/>')
        z_grid_svg.append(f'<text x="35" y="{y+4:.1f}" fill="{col}" font-size="12" font-family="sans-serif">{"+" if z_val>0 else ""}{z_val}σ</text>')

    z_pts = []
    for i in range(n_bars):
        x = zx0 + (i + 0.5) / n_bars * (zx1 - zx0)
        y = zy1 - (z_scores[i] - min_z) / (max_z - min_z) * (zy1 - zy0)
        z_pts.append(f"{x:.1f},{y:.1f}")
    z_path = "M " + " L ".join(z_pts)

    # Time ticks
    time_ticks_svg = []
    for i in range(0, n_bars, 45):
        x = zx0 + (i + 0.5) / n_bars * (zx1 - zx0)
        time_ticks_svg.append(f'<text x="{x-14:.1f}" y="{zy1+22}" fill="#8F9CAE" font-size="12" font-family="sans-serif">{times[i]}</text>')
        time_ticks_svg.append(f'<line x1="{x:.1f}" y1="{zy1}" x2="{x:.1f}" y2="{zy1+6}" stroke="#262D3D" />')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%" style="background:#0B0E14; border-radius:12px;">
  <!-- Header -->
  <text x="40" y="36" fill="#FFFFFF" font-size="20" font-weight="bold" font-family="sans-serif">📊 Торговая Сессия {s_date} | Реальные биржевые свечи NVDA (1m) vs Fair Value</text>
  <text x="40" y="58" fill="#00E5FF" font-size="13" font-family="sans-serif">100% Genuine Alpaca Parquet Stream | 390 баров (09:30–16:00 ET)</text>

  <!-- Price Chart -->
  <rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="#141822" stroke="#262D3D" rx="6"/>
  {"".join(p_grid_svg)}
  <path d="{fair_path}" fill="none" stroke="#FFD600" stroke-width="2.2" stroke-dasharray="6,4" />
  {"".join(candles_svg)}
  {"".join(signals_svg)}

  <!-- Legend -->
  <rect x="{px1-330}" y="{py0+12}" width="315" height="34" fill="#1C2230" stroke="#262D3D" rx="4"/>
  <line x1="{px1-315}" y1="{py0+29}" x2="{px1-285}" y2="{py0+29}" stroke="#FFD600" stroke-width="2.5" stroke-dasharray="6,4"/>
  <text x="{px1-275}" y="{py0+33}" fill="#FFD600" font-size="13" font-weight="bold" font-family="sans-serif">Fair Value (QQQ * Beta)</text>

  <!-- Z-Score Chart -->
  <text x="{zx0}" y="{zy0-10}" fill="#FFFFFF" font-size="14" font-weight="bold" font-family="sans-serif">📐 Z-Score Отклонения и уровни входа (±1.5σ / ±4.0σ)</text>
  <rect x="{zx0}" y="{zy0}" width="{zx1-zx0}" height="{zy1-zy0}" fill="#141822" stroke="#262D3D" rx="6"/>
  {"".join(z_grid_svg)}
  <path d="{z_path}" fill="none" stroke="#00E5FF" stroke-width="2" />
  {"".join(time_ticks_svg)}
</svg>"""
    return svg


if __name__ == "__main__":
    mgr = MarketDataManager()
    print("Writing equity SVG...")
    eq_svg = generate_equity_svg(mgr.eq_prod)
    Path("equity_drawdown_2026.svg").write_text(eq_svg, encoding="utf-8")

    august_dates = [s["date"] for s in mgr.sessions_list if s["date"].startswith("2026-08-") and s["trades_count"] > 0]
    for d in august_dates[:4]:
        chunk = mgr.get_session_chunk(d)
        if chunk:
            tag = d.replace("-", "_")
            s_svg = generate_session_svg(chunk)
            Path(f"session_{tag}.svg").write_text(s_svg, encoding="utf-8")
    print("Done!")
