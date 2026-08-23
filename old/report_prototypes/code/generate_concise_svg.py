"""Precision concise SVG generator for August sessions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from server import MarketDataManager


def generate_concise_session_svg(chunk):
    s_date = chunk["date"]
    times = chunk["times"]
    opens = np.array(chunk["open"])
    highs = np.array(chunk["high"])
    lows = np.array(chunk["low"])
    closes = np.array(chunk["close"])
    fairs = np.array(chunk["fair"])
    z_scores = np.array(chunk["z_score"])
    signals = chunk["signals"]

    W, H = 1200, 700
    px0, py0, px1, py1 = 70, 70, 1140, 440
    zx0, zy0, zx1, zy1 = 70, 500, 1140, 650

    min_p = float(min(lows.min(), fairs.min()) * 0.998)
    max_p = float(max(highs.max(), fairs.max()) * 1.002)
    n_bars = len(times)

    # Grid Y Price
    p_grid = []
    for val in np.linspace(min_p, max_p, 6):
        y = py1 - (val - min_p) / (max_p - min_p) * (py1 - py0)
        p_grid.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="#262D3D" stroke-dasharray="3"/>')
        p_grid.append(f'<text x="15" y="{y+4:.1f}" fill="#8F9CAE" font-size="11" font-family="sans-serif">${val:.2f}</text>')

    # Fair value path
    fair_pts = []
    for i in range(n_bars):
        x = px0 + (i + 0.5) / n_bars * (px1 - px0)
        y = py1 - (fairs[i] - min_p) / (max_p - min_p) * (py1 - py0)
        fair_pts.append(f"{x:.1f},{y:.1f}")
    fair_path = "M " + " L ".join(fair_pts)

    # Candlesticks
    candles = []
    sigs_svg = []
    candle_w = max(1.5, (px1 - px0) / n_bars * 0.7)

    for i in range(n_bars):
        x = px0 + (i + 0.5) / n_bars * (px1 - px0)
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        yh = py1 - (h - min_p) / (max_p - min_p) * (py1 - py0)
        yl = py1 - (l - min_p) / (max_p - min_p) * (py1 - py0)
        yo = py1 - (o - min_p) / (max_p - min_p) * (py1 - py0)
        yc = py1 - (c - min_p) / (max_p - min_p) * (py1 - py0)

        is_up = c >= o
        col = "#00E676" if is_up else "#FF5252"
        top_y = min(yo, yc)
        bot_y = max(yo, yc)
        h_body = max(1.0, bot_y - top_y)

        candles.append(f'<line x1="{x:.1f}" y1="{yh:.1f}" x2="{x:.1f}" y2="{yl:.1f}" stroke="{col}"/>')
        candles.append(f'<rect x="{x-candle_w/2:.1f}" y="{top_y:.1f}" width="{candle_w:.1f}" height="{h_body:.1f}" fill="{col}"/>')

        sig = signals[i]
        if sig == "BUY_LONG":
            my = yl + 12
            sigs_svg.append(f'<polygon points="{x:.1f},{my-8:.1f} {x-5:.1f},{my:.1f} {x+5:.1f},{my:.1f}" fill="#00E676" stroke="#FFF"/>')
            sigs_svg.append(f'<rect x="{x-30:.1f}" y="{my+2:.1f}" width="60" height="18" fill="#0A2E1C" stroke="#00E676" rx="3"/>')
            sigs_svg.append(f'<text x="{x:.1f}" y="{my+14:.1f}" fill="#00E676" font-size="10" font-weight="bold" font-family="sans-serif" text-anchor="middle">▲ BUY</text>')
        elif sig == "SELL_SHORT":
            my = yh - 12
            sigs_svg.append(f'<polygon points="{x:.1f},{my+8:.1f} {x-5:.1f},{my:.1f} {x+5:.1f},{my:.1f}" fill="#FF5252" stroke="#FFF"/>')
            sigs_svg.append(f'<rect x="{x-35:.1f}" y="{my-20:.1f}" width="70" height="18" fill="#3D1217" stroke="#FF5252" rx="3"/>')
            sigs_svg.append(f'<text x="{x:.1f}" y="{my-8:.1f}" fill="#FF5252" font-size="10" font-weight="bold" font-family="sans-serif" text-anchor="middle">▼ SHORT</text>')
        elif sig.startswith("EXIT_"):
            my = yh - 8
            lbl = "✖ TP" if "TAKE" in sig else ("✖ SL" if "STOP_LOSS" in sig else "✖ TIME")
            c_sig = "#00E676" if "TAKE" in sig else "#FFD600"
            bg_sig = "#0A2E1C" if "TAKE" in sig else "#38300E"
            sigs_svg.append(f'<rect x="{x-25:.1f}" y="{my-20:.1f}" width="50" height="18" fill="{bg_sig}" stroke="{c_sig}" rx="3"/>')
            sigs_svg.append(f'<text x="{x:.1f}" y="{my-8:.1f}" fill="{c_sig}" font-size="10" font-weight="bold" font-family="sans-serif" text-anchor="middle">{lbl}</text>')

    # Z-Score Subplot
    min_z, max_z = -4.5, 4.5
    z_grid = []
    for z_val in [-4.0, -1.5, 0.0, 1.5, 4.0]:
        y = zy1 - (z_val - min_z) / (max_z - min_z) * (zy1 - zy0)
        col = "#D50000" if abs(z_val) == 4.0 else ("#00E676" if z_val == -1.5 else ("#FF5252" if z_val == 1.5 else "#262D3D"))
        z_grid.append(f'<line x1="{zx0}" y1="{y:.1f}" x2="{zx1}" y2="{y:.1f}" stroke="{col}" stroke-dasharray="3"/>')
        z_grid.append(f'<text x="25" y="{y+4:.1f}" fill="{col}" font-size="11" font-family="sans-serif">{"+" if z_val>0 else ""}{z_val}σ</text>')

    z_pts = []
    for i in range(n_bars):
        x = zx0 + (i + 0.5) / n_bars * (zx1 - zx0)
        y = zy1 - (z_scores[i] - min_z) / (max_z - min_z) * (zy1 - zy0)
        z_pts.append(f"{x:.1f},{y:.1f}")
    z_path = "M " + " L ".join(z_pts)

    # Time ticks
    t_ticks = []
    for i in range(0, n_bars, 60):
        x = zx0 + (i + 0.5) / n_bars * (zx1 - zx0)
        t_ticks.append(f'<text x="{x-14:.1f}" y="{zy1+20}" fill="#8F9CAE" font-size="11" font-family="sans-serif">{times[i]}</text>')
        t_ticks.append(f'<line x1="{x:.1f}" y1="{zy1}" x2="{x:.1f}" y2="{zy1+5}" stroke="#262D3D"/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%" style="background:#0B0E14; border-radius:10px;">
  <!-- Header -->
  <text x="30" y="32" fill="#FFFFFF" font-size="18" font-weight="bold" font-family="sans-serif">📊 Сессия {s_date} | Свечи NVDA (1m, 390 баров) vs Fair Value</text>
  <text x="30" y="52" fill="#00E5FF" font-size="12" font-family="sans-serif">100% Real Alpaca Parquet Data | NASDAQ RTH (09:30–16:00 ET)</text>

  <!-- Price Chart -->
  <rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="#141822" stroke="#262D3D" rx="5"/>
  {"".join(p_grid)}
  <path d="{fair_path}" fill="none" stroke="#FFD600" stroke-width="2" stroke-dasharray="5,3"/>
  {"".join(candles)}
  {"".join(sigs_svg)}

  <!-- Legend -->
  <rect x="{px1-260}" y="{py0+10}" width="250" height="28" fill="#1C2230" stroke="#262D3D" rx="4"/>
  <line x1="{px1-245}" y1="{py0+24}" x2="{px1-220}" y2="{py0+24}" stroke="#FFD600" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="{px1-210}" y="{py0+28}" fill="#FFD600" font-size="11" font-weight="bold" font-family="sans-serif">Fair Value (QQQ * Beta)</text>

  <!-- Z-Score Chart -->
  <text x="{zx0}" y="{zy0-8}" fill="#FFFFFF" font-size="13" font-weight="bold" font-family="sans-serif">📐 Z-Score Отклонения (±1.5σ пороги, ±4.0σ блок)</text>
  <rect x="{zx0}" y="{zy0}" width="{zx1-zx0}" height="{zy1-zy0}" fill="#141822" stroke="#262D3D" rx="5"/>
  {"".join(z_grid)}
  <path d="{z_path}" fill="none" stroke="#00E5FF" stroke-width="1.8"/>
  {"".join(t_ticks)}
</svg>"""
    return svg


if __name__ == "__main__":
    mgr = MarketDataManager()
    for d in ['2026-08-21', '2026-08-19', '2026-08-18', '2026-08-14']:
        chunk = mgr.get_session_chunk(d)
        svg = generate_concise_session_svg(chunk)
        print(f"===SESSION_{d.replace('-', '_')}_START===")
        print(svg)
        print(f"===SESSION_{d.replace('-', '_')}_END===")
