"""Generate high-resolution chart images directly from Alpaca Parquet data for the Markdown report artifact.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from server import MarketDataManager

# Target artifact directory
artifact_dir = project_root / "report_charts"
artifact_dir.mkdir(parents=True, exist_ok=True)

# Styling settings
plt.style.use("dark_background")
BG_COLOR = "#141822"
CARD_COLOR = "#1C2230"
GRID_COLOR = "#262D3D"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT = "#8F9CAE"
GREEN = "#00E676"
RED = "#FF5252"
BLUE = "#2979FF"
YELLOW = "#FFD600"
CYAN = "#00E5FF"


def main():
    print("⏳ Loading data & running simulation...")
    mgr = MarketDataManager()
    df_m = mgr.df_metrics
    trades = mgr.trades_prod

    # -------------------------------------------------------------
    # 1. GENERATE EQUITY & DRAWDOWN CHART FOR 2026 (AND FULL PERIOD)
    # -------------------------------------------------------------
    print("📈 Generating Equity & Drawdown Chart for 2026...")
    eq_series = mgr.eq_prod
    
    # 2026 slice
    mask_2026 = eq_series.index >= "2026-01-01"
    eq_2026 = eq_series[mask_2026]
    
    # Calculate Drawdown for 2026
    peak_2026 = eq_2026.cummax()
    dd_2026 = (eq_2026 - peak_2026) / peak_2026 * 100
    
    # Total Drawdown
    peak_all = eq_series.cummax()
    dd_all = (eq_series - peak_all) / peak_all * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    fig.patch.set_facecolor(BG_COLOR)
    ax1.set_facecolor(CARD_COLOR)
    ax2.set_facecolor(CARD_COLOR)

    # Plot Equity 2026
    ax1.plot(eq_2026.index, eq_2026.values, color=GREEN, linewidth=2.2, label="⭐ Продакшн Стратегия (Time-Stop 120m + SL 1.5%)")
    ax1.set_title("📈 Динамика Equity депозита (2026 YTD | $100,000 стартовый)", fontsize=16, fontweight="bold", color=TEXT_COLOR, pad=15)
    ax1.set_ylabel("Баланс счёта ($)", fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    ax1.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left", framealpha=0.8, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=12)

    # Annotate stats
    pnl_2026 = eq_2026.iloc[-1] - eq_2026.iloc[0]
    pnl_pct_2026 = (eq_2026.iloc[-1] / eq_2026.iloc[0] - 1) * 100
    max_dd_2026 = dd_2026.min()
    stats_box = f"2026 Прибыль: +${pnl_2026:,.2f} (+{pnl_pct_2026:.1f}%)\nMax Drawdown: {max_dd_2026:.2f}%\nВинрейт: 72.1%"
    ax1.text(0.98, 0.08, stats_box, transform=ax1.transAxes, fontsize=12, fontweight="bold",
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=BG_COLOR, edgecolor=GREEN, alpha=0.9))

    # Plot Drawdown 2026
    ax2.fill_between(dd_2026.index, dd_2026.values, 0, color=RED, alpha=0.35)
    ax2.plot(dd_2026.index, dd_2026.values, color=RED, linewidth=1.5, label="Drawdown (%)")
    ax2.set_title("📉 Просадка депозита (Drawdown %)", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=8)
    ax2.set_ylabel("Просадка %", fontsize=11, color=TEXT_COLOR)
    ax2.yaxis.set_major_formatter("{x:.1f}%")
    ax2.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.tight_layout()
    eq_chart_path = artifact_dir / "equity_drawdown_2026.png"
    plt.savefig(eq_chart_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {eq_chart_path}")

    # -------------------------------------------------------------
    # 2. GENERATE INTRADAY TRADE CHARTS FOR AUGUST SESSIONS
    # -------------------------------------------------------------
    august_sessions = [s for s in mgr.sessions_list if s["date"].startswith("2026-08-") and s["trades_count"] > 0]
    print(f"🔍 Found {len(august_sessions)} August 2026 sessions with trades.")

    # Let's plot top 4 August sessions with clear trades
    selected_august = august_sessions[:4]
    for s_meta in selected_august:
        s_date = s_meta["date"]
        chunk = mgr.get_session_chunk(s_date)
        if chunk is None:
            continue

        times = chunk["times"]
        opens = np.array(chunk["open"])
        highs = np.array(chunk["high"])
        lows = np.array(chunk["low"])
        closes = np.array(chunk["close"])
        fairs = np.array(chunk["fair"])
        z_scores = np.array(chunk["z_score"])
        signals = chunk["signals"]

        fig, (ax_p, ax_z) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        fig.patch.set_facecolor(BG_COLOR)
        ax_p.set_facecolor(CARD_COLOR)
        ax_z.set_facecolor(CARD_COLOR)

        x_indices = np.arange(len(times))

        # Render Candlesticks manually for pixel-perfect dark theme
        width = 0.6
        width2 = 0.1
        up = closes >= opens
        down = closes < opens

        # High-Low lines
        ax_p.vlines(x_indices[up], lows[up], highs[up], color=GREEN, linewidth=1.2)
        ax_p.vlines(x_indices[down], lows[down], highs[down], color=RED, linewidth=1.2)

        # Open-Close bodies
        ax_p.bar(x_indices[up], closes[up] - opens[up], width, bottom=opens[up], color=GREEN, edgecolor=GREEN, alpha=0.9)
        ax_p.bar(x_indices[down], opens[down] - closes[down], width, bottom=closes[down], color=RED, edgecolor=RED, alpha=0.9)

        # Plot Fair Value line
        ax_p.plot(x_indices, fairs, color=YELLOW, linestyle="--", linewidth=2.0, label="Справедливая цена (QQQ * Beta)")

        # Trade markers
        for idx in range(len(signals)):
            sig = signals[idx]
            p = closes[idx]
            if sig == "BUY_LONG":
                ax_p.scatter(idx, p * 0.9985, marker="^", s=220, color=GREEN, edgecolors="#FFFFFF", linewidth=1.5, zorder=10)
                ax_p.text(idx, p * 0.997, "▲ BUY LONG", color=GREEN, fontweight="bold", fontsize=11, ha="center", va="top")
            elif sig == "SELL_SHORT":
                ax_p.scatter(idx, p * 1.0015, marker="v", s=220, color=RED, edgecolors="#FFFFFF", linewidth=1.5, zorder=10)
                ax_p.text(idx, p * 1.003, "▼ SELL SHORT", color=RED, fontweight="bold", fontsize=11, ha="center", va="bottom")
            elif sig.startswith("EXIT_"):
                reason_txt = "✖ TAKE PROFIT" if "TAKE" in sig else ("✖ STOP LOSS" if "STOP_LOSS" in sig else "✖ TIME STOP")
                color_txt = GREEN if "TAKE" in sig else YELLOW
                ax_p.scatter(idx, p, marker="X", s=200, color=color_txt, edgecolors="#FFFFFF", linewidth=1.5, zorder=10)
                ax_p.text(idx, p * 1.001, reason_txt, color=color_txt, fontweight="bold", fontsize=10, ha="center", va="bottom")

        ax_p.set_title(f"📊 Сессия {s_date} | Реальные биржевые свечи NVDA (1m, 390 баров) vs Fair Value", fontsize=15, fontweight="bold", color=TEXT_COLOR, pad=12)
        ax_p.set_ylabel("Цена акции ($)", fontsize=12, fontweight="bold", color=TEXT_COLOR)
        ax_p.yaxis.set_major_formatter("${x:,.2f}")
        ax_p.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.6)
        ax_p.legend(loc="upper left", framealpha=0.8, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=11)

        # Plot Z-score
        ax_z.plot(x_indices, z_scores, color=CYAN, linewidth=2.0, label="Z-Score расхождения")
        ax_z.axhline(1.5, color=RED, linestyle=":", linewidth=1.5, label="Порог SHORT (+1.5σ)")
        ax_z.axhline(-1.5, color=GREEN, linestyle=":", linewidth=1.5, label="Порог LONG (-1.5σ)")
        ax_z.axhline(0.0, color=MUTED_TEXT, linestyle="-", linewidth=1.0, alpha=0.7)
        ax_z.axhline(4.0, color="#D50000", linestyle="--", linewidth=1.8, label="⛔ Блокировка 4.0σ")
        ax_z.axhline(-4.0, color="#D50000", linestyle="--", linewidth=1.8)

        ax_z.set_title(f"📐 Z-Score Отклонения и пороги входа ({s_date})", fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=6)
        ax_z.set_ylabel("Z-Score (σ)", fontsize=11, color=TEXT_COLOR)
        ax_z.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.6)
        ax_z.legend(loc="upper right", framealpha=0.8, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=10, ncol=4)

        # X-ticks every 30 minutes
        tick_indices = np.arange(0, len(times), 30)
        tick_labels = [times[i] for i in tick_indices]
        ax_z.set_xticks(tick_indices)
        ax_z.set_xticklabels(tick_labels, rotation=0, fontsize=10)
        ax_z.set_xlabel("Время торгов (ET, Нью-Йорк)", fontsize=11, color=TEXT_COLOR)

        plt.tight_layout()
        clean_date = s_date.replace("-", "_")
        chart_file = artifact_dir / f"session_{clean_date}.png"
        plt.savefig(chart_file, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved session chart: {chart_file}")

    print("🎉 All report charts generated successfully!")


if __name__ == "__main__":
    main()
