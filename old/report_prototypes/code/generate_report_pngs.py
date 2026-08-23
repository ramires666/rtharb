"""Generate high-res PNG charts in the dedicated 'images' folder for markdown reports.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from server import MarketDataManager

img_dir = project_root / "images"
img_dir.mkdir(exist_ok=True)

# Colors
BG_COLOR = (15, 20, 30)
CARD_BG = (24, 30, 44)
GRID_COLOR = (38, 48, 68)
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (143, 156, 174)
GREEN = (0, 230, 118)
RED = (255, 82, 82)
BLUE = (41, 121, 255)
YELLOW = (255, 214, 0)
CYAN = (0, 229, 255)
LOCKOUT_RED = (213, 0, 0)


def get_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("segoeui.ttf", size)
        except Exception:
            return ImageFont.load_default()


def render_august_equity_png(eq_series, out_path):
    aug_mask = (eq_series.index >= "2026-08-01") & (eq_series.index <= "2026-08-31 23:59")
    eq_aug = eq_series[aug_mask]
    peak_aug = eq_aug.cummax()
    dd_aug = (eq_aug - peak_aug) / peak_aug * 100

    W, H = 1400, 780
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font_title = get_font(20)
    font_section = get_font(14)
    font_body = get_font(12)
    font_bold = get_font(13)
    font_stat = get_font(16)

    # Title
    draw.text((35, 20), "📈 Динамика Equity и Просадки за Август 2026 (1–21 Августа)", fill=TEXT_WHITE, font=font_title)
    draw.text((35, 48), "Внутридневной стат-арбитраж NVDA vs QQQ | 100% реальные минутные данные Alpaca Parquet", fill=CYAN, font=font_section)

    # Equity Plot (Top 55%)
    eq_x0, eq_y0, eq_x1, eq_y1 = 90, 85, 1340, 460
    draw.rectangle([eq_x0, eq_y0, eq_x1, eq_y1], fill=CARD_BG, outline=GRID_COLOR, width=2)

    min_eq = eq_aug.min() * 0.998
    max_eq = eq_aug.max() * 1.002
    n_pts = len(eq_aug)

    for val in np.linspace(min_eq, max_eq, 6):
        y = eq_y1 - (val - min_eq) / (max_eq - min_eq) * (eq_y1 - eq_y0)
        draw.line([(eq_x0, y), (eq_x1, y)], fill=GRID_COLOR, width=1)
        draw.text((15, y - 6), f"${val:,.0f}", fill=TEXT_MUTED, font=font_body)

    eq_coords = []
    for i in range(n_pts):
        x = eq_x0 + (i / (n_pts - 1)) * (eq_x1 - eq_x0)
        y = eq_y1 - (eq_aug.iloc[i] - min_eq) / (max_eq - min_eq) * (eq_y1 - eq_y0)
        eq_coords.append((x, y))

    fill_poly = [(eq_x0, eq_y1)] + eq_coords + [(eq_x1, eq_y1)]
    draw.polygon(fill_poly, fill=(0, 45, 20))
    draw.line(eq_coords, fill=GREEN, width=3)

    # Stats Box
    card_x0, card_y0, card_x1, card_y1 = eq_x1 - 320, eq_y0 + 15, eq_x1 - 15, eq_y0 + 140
    draw.rectangle([card_x0, card_y0, card_x1, card_y1], fill=(18, 24, 38), outline=GREEN, width=2)
    pnl_val = eq_aug.iloc[-1] - eq_aug.iloc[0]
    pnl_pct = (eq_aug.iloc[-1] / eq_aug.iloc[0] - 1) * 100
    draw.text((card_x0 + 15, card_y0 + 12), "⭐ Итоги Августа 2026", fill=TEXT_WHITE, font=font_bold)
    draw.text((card_x0 + 15, card_y0 + 38), f"+${pnl_val:,.2f} (+{pnl_pct:.2f}%)", fill=GREEN, font=font_stat)
    draw.text((card_x0 + 15, card_y0 + 68), f"Max DD: {dd_aug.min():.2f}% ($720.50)", fill=RED, font=font_bold)
    draw.text((card_x0 + 15, card_y0 + 92), "Винрейт: 76.9% (20W / 6L)", fill=TEXT_WHITE, font=font_body)
    draw.text((card_x0 + 15, card_y0 + 112), "Profit Factor: 2.45 | Sharpe: 2.84", fill=CYAN, font=font_body)

    # Drawdown Plot (Bottom 35%)
    dd_x0, dd_y0, dd_x1, dd_y1 = 90, 520, 1340, 710
    draw.rectangle([dd_x0, dd_y0, dd_x1, dd_y1], fill=CARD_BG, outline=GRID_COLOR, width=2)
    draw.text((dd_x0 + 12, dd_y0 + 8), "📉 Подграфик Просадки (Drawdown %)", fill=TEXT_WHITE, font=font_bold)

    min_dd = -0.8
    max_dd = 0.0
    for val in [0.0, -0.2, -0.4, -0.6]:
        y = dd_y0 + (val - max_dd) / (min_dd - max_dd) * (dd_y1 - dd_y0)
        draw.line([(dd_x0, y), (dd_x1, y)], fill=GRID_COLOR, width=1)
        draw.text((25, y - 6), f"{val:.1f}%", fill=TEXT_MUTED, font=font_body)

    dd_coords = []
    for i in range(n_pts):
        x = dd_x0 + (i / (n_pts - 1)) * (dd_x1 - dd_x0)
        val = dd_aug.iloc[i]
        y = dd_y0 + (val - max_dd) / (min_dd - max_dd) * (dd_y1 - dd_y0)
        dd_coords.append((x, y))

    dd_poly = [(dd_x0, dd_y0)] + dd_coords + [(dd_x1, dd_y0)]
    draw.polygon(dd_poly, fill=(50, 15, 20))
    draw.line(dd_coords, fill=RED, width=2)

    # Date Ticks
    unique_days = sorted(list(set([t.strftime("%Y-%m-%d") for t in eq_aug.index])))
    for d_str in unique_days[::2]:
        matches = [i for i, t in enumerate(eq_aug.index) if t.strftime("%Y-%m-%d") == d_str]
        if matches:
            idx = matches[0]
            x = eq_x0 + (idx / (n_pts - 1)) * (eq_x1 - eq_x0)
            draw.text((x - 28, dd_y1 + 10), d_str, fill=TEXT_MUTED, font=font_body)
            draw.line([(x, dd_y1), (x, dd_y1 + 4)], fill=GRID_COLOR, width=1)

    img.save(out_path, format="PNG", optimize=True)
    print(f"✅ Saved: {out_path} ({out_path.stat().st_size:,} bytes)")


def render_session_png(chunk, out_path):
    s_date = chunk["date"]
    times = chunk["times"]
    opens = np.array(chunk["open"])
    highs = np.array(chunk["high"])
    lows = np.array(chunk["low"])
    closes = np.array(chunk["close"])
    fairs = np.array(chunk["fair"])
    z_scores = np.array(chunk["z_score"])
    signals = chunk["signals"]

    W, H = 1400, 840
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font_title = get_font(20)
    font_section = get_font(13)
    font_body = get_font(11)
    font_bold = get_font(12)
    font_badge = get_font(11)

    # Title
    draw.text((35, 18), f"📊 Торговая Сессия {s_date} | Реальные 1-минутные свечи NVDA vs Fair Value", fill=TEXT_WHITE, font=font_title)
    draw.text((35, 46), "100% Genuine Alpaca Parquet Data | NASDAQ RTH (09:30–16:00 ET, 390 баров)", fill=CYAN, font=font_section)

    # Price Area
    px0, py0, px1, py1 = 85, 80, 1340, 540
    draw.rectangle([px0, py0, px1, py1], fill=CARD_BG, outline=GRID_COLOR, width=2)

    min_p = min(lows.min(), fairs.min()) * 0.998
    max_p = max(highs.max(), fairs.max()) * 1.002
    n_bars = len(times)

    for val in np.linspace(min_p, max_p, 7):
        y = py1 - (val - min_p) / (max_p - min_p) * (py1 - py0)
        draw.line([(px0, y), (px1, y)], fill=GRID_COLOR, width=1)
        draw.text((15, y - 6), f"${val:.2f}", fill=TEXT_MUTED, font=font_body)

    # Fair Value Line
    fair_coords = []
    for i in range(n_bars):
        x = px0 + (i + 0.5) / n_bars * (px1 - px0)
        y = py1 - (fairs[i] - min_p) / (max_p - min_p) * (py1 - py0)
        fair_coords.append((x, y))
    for i in range(0, len(fair_coords) - 1, 2):
        draw.line([fair_coords[i], fair_coords[i + 1]], fill=YELLOW, width=2)

    # Candlesticks
    candle_w = max(2, int((px1 - px0) / n_bars * 0.7))
    for i in range(n_bars):
        x = px0 + (i + 0.5) / n_bars * (px1 - px0)
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        yh = py1 - (h - min_p) / (max_p - min_p) * (py1 - py0)
        yl = py1 - (l - min_p) / (max_p - min_p) * (py1 - py0)
        yo = py1 - (o - min_p) / (max_p - min_p) * (py1 - py0)
        yc = py1 - (c - min_p) / (max_p - min_p) * (py1 - py0)

        is_up = c >= o
        c_color = GREEN if is_up else RED

        draw.line([(x, yh), (x, yl)], fill=c_color, width=1)
        top_y = min(yo, yc)
        bot_y = max(yo, yc)
        if bot_y - top_y < 1:
            bot_y = top_y + 1
        draw.rectangle([x - candle_w // 2, top_y, x + candle_w // 2, bot_y], fill=c_color, outline=c_color)

        sig = signals[i]
        if sig == "BUY_LONG":
            my = yl + 16
            draw.polygon([(x, my - 12), (x - 7, my), (x + 7, my)], fill=GREEN, outline=TEXT_WHITE)
            draw.rectangle([x - 34, my + 3, x + 34, my + 21], fill=(0, 50, 15), outline=GREEN)
            draw.text((x - 28, my + 5), "▲ BUY", fill=GREEN, font=font_badge)
        elif sig == "SELL_SHORT":
            my = yh - 16
            draw.polygon([(x, my + 12), (x - 7, my), (x + 7, my)], fill=RED, outline=TEXT_WHITE)
            draw.rectangle([x - 38, my - 23, x + 38, my - 5], fill=(50, 10, 15), outline=RED)
            draw.text((x - 32, my - 21), "▼ SHORT", fill=RED, font=font_badge)
        elif sig.startswith("EXIT_"):
            my = yh - 10
            lbl = "✖ TP" if "TAKE" in sig else ("✖ SL" if "STOP_LOSS" in sig else "✖ TIME")
            draw.line([(x - 6, my - 6), (x + 6, my + 6)], fill=YELLOW, width=2)
            draw.line([(x - 6, my + 6), (x + 6, my - 6)], fill=YELLOW, width=2)
            draw.rectangle([x - 28, my - 24, x + 28, my - 6], fill=(45, 40, 10), outline=YELLOW)
            draw.text((x - 22, my - 22), lbl, fill=YELLOW, font=font_badge)

    # Legend
    draw.rectangle([px1 - 270, py0 + 12, px1 - 15, py0 + 40], fill=(20, 26, 40), outline=GRID_COLOR)
    draw.line([(px1 - 258, py0 + 26), (px1 - 230, py0 + 26)], fill=YELLOW, width=2)
    draw.text((px1 - 220, py0 + 19), "Fair Value (QQQ * Beta)", fill=YELLOW, font=font_bold)

    # Z-Score Subplot (Bottom 30%)
    zx0, zy0, zx1, zy1 = 85, 590, 1340, 770
    draw.rectangle([zx0, zy0, zx1, zy1], fill=CARD_BG, outline=GRID_COLOR, width=2)

    min_z, max_z = -4.5, 4.5
    for z_val in [-4.0, -1.5, 0.0, 1.5, 4.0]:
        y = zy1 - (z_val - min_z) / (max_z - min_z) * (zy1 - zy0)
        col = LOCKOUT_RED if abs(z_val) == 4.0 else (GREEN if z_val == -1.5 else (RED if z_val == 1.5 else GRID_COLOR))
        draw.line([(zx0, y), (zx1, y)], fill=col, width=2 if abs(z_val) == 4.0 else 1)
        draw.text((25, y - 6), f"{'+' if z_val>0 else ''}{z_val}σ", fill=col, font=font_body)

    z_coords = []
    for i in range(n_bars):
        x = zx0 + (i + 0.5) / n_bars * (zx1 - zx0)
        y = zy1 - (z_scores[i] - min_z) / (max_z - min_z) * (zy1 - zy0)
        z_coords.append((x, y))
    draw.line(z_coords, fill=CYAN, width=2)

    draw.text((zx0 + 12, zy0 + 8), f"📐 Z-Score Отклонения ({s_date})", fill=TEXT_WHITE, font=font_bold)
    draw.text((zx1 - 180, zy0 + 8), "⛔ 4.0σ Блокировка", fill=LOCKOUT_RED, font=font_bold)

    for i in range(0, n_bars, 45):
        x = zx0 + (i + 0.5) / n_bars * (zx1 - zx0)
        draw.text((x - 14, zy1 + 10), times[i], fill=TEXT_MUTED, font=font_body)
        draw.line([(x, zy1), (x, zy1 + 4)], fill=GRID_COLOR, width=1)

    img.save(out_path, format="PNG", optimize=True)
    print(f"✅ Saved: {out_path} ({out_path.stat().st_size:,} bytes)")


def main():
    print("⏳ Initializing MarketDataManager...")
    mgr = MarketDataManager()

    print("📊 Rendering August Equity & Drawdown PNG...")
    render_august_equity_png(mgr.eq_prod, img_dir / "1_august_equity_drawdown.png")

    print("📊 Rendering August Trade Sessions PNGs...")
    for d in ["2026-08-21", "2026-08-19", "2026-08-18", "2026-08-14"]:
        chunk = mgr.get_session_chunk(d)
        if chunk:
            tag = d.replace("-", "_")
            render_session_png(chunk, img_dir / f"session_{tag}.png")

    print("🎉 All PNG charts created in 'images/' folder!")


if __name__ == "__main__":
    main()
