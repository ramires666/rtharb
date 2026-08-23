"""Interactive Streamlit Web Dashboard for RTH Statistical Arbitrage and A/B/C/D Matrix Analysis."""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

from rtharb.config import AppConfig, StrategyConfig, BacktestConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator, SignalType
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics
from rtharb.analysis.matrix_comparator import MatrixComparator
from rtharb.analysis.optimizer import ParameterOptimizer

load_dotenv()

st.set_page_config(
    page_title="RTH Stat-Arb Platform (NVDA vs QQQ)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #2962FF;
    }
    .metric-title { color: #8F9CAE; font-size: 13px; font-weight: 500; }
    .metric-val { color: #FFFFFF; font-size: 22px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and synchronizing market data...")
def load_market_data(lead_sym: str, target_sym: str, source: str, days_back: int):
    loader = DataLoader(cache_dir="data_cache", source=source)
    df_lead, df_target = loader.get_synchronized_pair(
        ticker_lead=lead_sym,
        ticker_target=target_sym,
        days_back=days_back,
        source=source
    )
    return df_lead, df_target


def main():
    st.title("🎯 RTH Intraday Statistical Arbitrage & Lead-Lag Platform")
    st.caption("One-Legged Mean Reversion Strategy (NVDA vs QQQ) with Reversal Confirmation & 4σ Outlier Protections")

    # ==================== SIDEBAR CONTROLS ====================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            lead_ticker = st.text_input("Leader (Index/ETF)", value="QQQ").upper()
        with col_t2:
            target_ticker = st.text_input("Target (Traded)", value="NVDA").upper()

        data_source = st.selectbox("Data Source", options=["alpaca", "yfinance"], index=0)
        days_back = st.slider("Lookback (Days)", min_value=5, max_value=730, value=730 if data_source == "alpaca" else 28, step=5)

        st.subheader("📐 Fair Value & Model")
        beta_mode = st.selectbox("Beta Calculation", options=["dynamic_rolling", "fixed_1.0", "historical_daily"], index=0)
        rolling_w = st.slider("Rolling Spread Window W (min)", min_value=15, max_value=120, value=30, step=5)
        
        st.subheader("🎯 Entry & Reversal Hook")
        z_entry = st.slider("Divergence Entry Threshold (Z_entry)", min_value=0.8, max_value=3.0, value=1.5, step=0.1)
        reversal_type = st.selectbox("Reversal Base", options=["z_score_hook", "price_pct_rebound"], index=0)
        reversal_delta = st.slider("Reversal Confirmation Delta (δ)", min_value=0.02, max_value=0.50, value=0.15, step=0.01)
        reversal_timeout = st.slider("Reversal Timeout (bars)", min_value=3, max_value=20, value=10)

        st.subheader("🛡️ 4-Sigma Extreme Dislocation")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            enable_lockout = st.checkbox("Entry Lockout", value=True, help="Block new entries when divergence >= Z_max")
        with col_f2:
            enable_emerg = st.checkbox("Emergency Exit", value=False, help="Force liquidate active position when divergence >= Z_max")
        
        z_max = st.slider("Max Allowed Divergence (Z_max)", min_value=3.0, max_value=5.0, value=4.0, step=0.1)
        lockout_mode = st.selectbox("Lockout Mode", options=["day_lockout", "window_lockout"], index=0)

        st.subheader("💰 Execution & Fees")
        init_cap = st.number_input("Capital ($)", min_value=1000.0, max_value=10000000.0, value=100000.0, step=10000.0)
        pos_size = st.number_input("Position Size ($)", min_value=1000.0, max_value=1000000.0, value=20000.0, step=5000.0)
        comm_share = st.number_input("Commission ($/share)", min_value=0.0, max_value=0.05, value=0.0035, step=0.0005, format="%.4f")
        slippage_bps = st.slider("Slippage (bps)", min_value=0, max_value=10, value=2, step=1)

    # Load Data
    try:
        df_lead, df_target = load_market_data(lead_ticker, target_ticker, data_source, days_back)
        st.success(f"✅ Loaded {len(df_target):,} 1-minute bars across {df_target['session_date'].nunique()} trading sessions ({df_target.index[0].date()} to {df_target.index[-1].date()})")
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        st.stop()

    # Compute Fair Value & Spread
    fv_model = FairValueModel(
        beta_mode=beta_mode,
        beta_rolling_days=10,
        rolling_window_w=rolling_w,
        min_session_warmup_bars=15
    )
    df_metrics = fv_model.compute_intraday_metrics(df_lead, df_target)

    # Build App Config
    cfg = AppConfig(
        strategy=StrategyConfig(
            ticker_lead=lead_ticker,
            ticker_target=target_ticker,
            data_source=data_source,
            beta_mode=beta_mode,
            rolling_window_w=rolling_w,
            z_entry=z_entry,
            reversal_type=reversal_type,
            reversal_delta=reversal_delta,
            reversal_timeout_bars=reversal_timeout,
            enable_extreme_entry_lockout=enable_lockout,
            enable_extreme_emergency_exit=enable_emerg,
            z_max_allowed=z_max,
            lockout_mode=lockout_mode
        ),
        backtest=BacktestConfig(
            initial_capital=init_cap,
            position_size_usd=pos_size,
            commission_per_share=comm_share,
            slippage_pct=slippage_bps / 10000.0,
            allow_short=True
        )
    )

    # UI Tabs
    tab_matrix, tab_backtest, tab_optimizer = st.tabs([
        "🔬 4-Scenario Matrix (A/B/C/D)",
        "📊 Deep Dive Backtest & Charts",
        "⚡ Parameter Optimizer & Heatmap"
    ])

    # ==================== TAB 1: MATRIX COMPARISON ====================
    with tab_matrix:
        st.subheader("🔬 4-Scenario Comparative Analysis: Impact of 4σ Protections")
        st.markdown("""
        Compare all 4 combinations on the exact same 2-year intraday market data:
        - **A (Pure Reversion):** No 4σ limits — trade all hooks, hold to mean or EOD.
        - **B (Smart Lockout - Recommended):** Block new entries on 4σ regime shifts, but don't cut active trades.
        - **C (Emergency Exit Only):** Enter everywhere, but force exit immediately if 4σ breached.
        - **D (Conservative):** Both 4σ entry lockout and emergency exit enabled.
        """)

        matrix_comp = MatrixComparator(cfg)
        with st.spinner("Running 4-scenario simulation..."):
            matrix_res = matrix_comp.run_all_scenarios(df_metrics)

        # Plot 4 Equity Curves
        fig_equity = go.Figure()
        colors = {
            "A: Pure Reversion (No 4σ caps)": "#787B86",
            "B: Entry Lockout Only (Recommended)": "#00E676",
            "C: Emergency Exit Only": "#FF5252",
            "D: Conservative (Lockout + Exit)": "#2979FF"
        }
        for s_name, s_equity in matrix_res["equity_curves"].items():
            fig_equity.add_trace(go.Scatter(
                x=matrix_res["equity_curves"].index,
                y=s_equity,
                mode="lines",
                name=s_name,
                line=dict(color=colors.get(s_name, "#FFFFFF"), width=2)
            ))

        fig_equity.update_layout(
            title="Portfolio Equity Curve Comparison ($)",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template="plotly_dark",
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        st.subheader("📋 Performance Comparison Matrix")
        st.dataframe(matrix_res["comparison_df"], use_container_width=True, hide_index=True)

    # ==================== TAB 2: DEEP DIVE BACKTEST ====================
    with tab_backtest:
        sig_gen = SignalGenerator(
            z_entry=cfg.strategy.z_entry,
            reversal_type=cfg.strategy.reversal_type,
            reversal_delta=cfg.strategy.reversal_delta,
            reversal_timeout_bars=cfg.strategy.reversal_timeout_bars,
            enable_extreme_entry_lockout=cfg.strategy.enable_extreme_entry_lockout,
            enable_extreme_emergency_exit=cfg.strategy.enable_extreme_emergency_exit,
            z_max_allowed=cfg.strategy.z_max_allowed,
            lockout_mode=cfg.strategy.lockout_mode,
            z_exit=cfg.strategy.z_exit,
            forced_close_time=cfg.strategy.forced_close_time,
            min_session_warmup_bars=cfg.strategy.min_session_warmup_bars
        )
        df_signals = sig_gen.generate_signals(df_metrics)

        engine = BacktestEngine(
            initial_capital=cfg.backtest.initial_capital,
            position_size_usd=cfg.backtest.position_size_usd,
            commission_per_share=cfg.backtest.commission_per_share,
            slippage_pct=cfg.backtest.slippage_pct,
            allow_short=cfg.backtest.allow_short
        )
        bt_out = engine.run(df_signals, ticker_target=cfg.strategy.ticker_target)
        metrics = calculate_performance_metrics(bt_out["df_results"], bt_out["trades_df"], cfg.backtest.initial_capital)

        # Top KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("Total Net PnL", f"${metrics.total_pnl:,.2f}", f"{metrics.total_return_pct:+.2f}%")
        kpi2.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        kpi3.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%", f"-${metrics.max_drawdown_usd:,.2f}", delta_color="inverse")
        kpi4.metric("Win Rate", f"{metrics.win_rate_pct:.1f}%")
        kpi5.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
        kpi6.metric("Total Trades", f"{metrics.total_trades}")

        # Filter by specific date for granular view
        available_dates = df_metrics["session_date"].unique()
        selected_date = st.selectbox("Inspect Specific Session Day", options=available_dates, index=len(available_dates) - 1)

        day_mask = df_signals["session_date"] == selected_date
        df_day = df_signals.loc[day_mask]

        # Multi-panel intraday plot
        fig_day = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f"{target_ticker} Price Action & Implied Fair Value ({selected_date})", "Spread Z-Score & Signals"),
            row_heights=[0.6, 0.4]
        )

        # Panel 1: Price and Fair Value
        fig_day.add_trace(go.Candlestick(
            x=df_day.index,
            open=df_day["target_open"],
            high=df_day["target_close"].rolling(2).max().fillna(df_day["target_close"]),
            low=df_day["target_close"].rolling(2).min().fillna(df_day["target_close"]),
            close=df_day["target_close"],
            name=f"{target_ticker} Actual",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ), row=1, col=1)

        fig_day.add_trace(go.Scatter(
            x=df_day.index,
            y=df_day["target_fair_price"],
            mode="lines",
            name=f"Implied Fair Value ({lead_ticker} * Beta)",
            line=dict(color="#FFD600", width=1.5, dash="dash")
        ), row=1, col=1)

        # Panel 2: Z-Score & Bands
        fig_day.add_trace(go.Scatter(
            x=df_day.index,
            y=df_day["z_score"],
            mode="lines",
            name="Z-Score",
            line=dict(color="#00E5FF", width=2)
        ), row=2, col=1)

        # Bands
        fig_day.add_hline(y=z_entry, line=dict(color="#FF5252", dash="dot"), row=2, col=1)
        fig_day.add_hline(y=-z_entry, line=dict(color="#00E676", dash="dot"), row=2, col=1)
        fig_day.add_hline(y=z_max, line=dict(color="#D50000", dash="dash"), row=2, col=1)
        fig_day.add_hline(y=-z_max, line=dict(color="#D50000", dash="dash"), row=2, col=1)
        fig_day.add_hline(y=0.0, line=dict(color="#787B86", width=1), row=2, col=1)

        # Entry and Exit Markers
        buys = df_day[df_day["signal"] == SignalType.BUY_LONG]
        shorts = df_day[df_day["signal"] == SignalType.SELL_SHORT]
        exits = df_day[df_day["signal"].isin([SignalType.EXIT_TAKE_PROFIT, SignalType.EXIT_EMERGENCY, SignalType.EXIT_FORCED_EOD])]

        if not buys.empty:
            fig_day.add_trace(go.Scatter(
                x=buys.index, y=buys["target_close"], mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#00E676"),
                name="BUY (Long Entry)"
            ), row=1, col=1)

        if not shorts.empty:
            fig_day.add_trace(go.Scatter(
                x=shorts.index, y=shorts["target_close"], mode="markers",
                marker=dict(symbol="triangle-down", size=14, color="#FF5252"),
                name="SHORT (Sell Entry)"
            ), row=1, col=1)

        if not exits.empty:
            fig_day.add_trace(go.Scatter(
                x=exits.index, y=exits["target_close"], mode="markers",
                marker=dict(symbol="x", size=12, color="#FFD600"),
                name="EXIT"
            ), row=1, col=1)

        fig_day.update_layout(
            template="plotly_dark",
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode="x unified"
        )
        st.plotly_chart(fig_day, use_container_width=True)

        st.subheader("📜 Completed Trades Log")
        if not bt_out["trades_df"].empty:
            st.dataframe(bt_out["trades_df"].sort_values(by="entry_time", ascending=False), use_container_width=True)
            csv = bt_out["trades_df"].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Trades CSV", data=csv, file_name="stat_arb_trades.csv", mime="text/csv")
        else:
            st.info("No trades generated with current parameters.")

    # ==================== TAB 3: OPTIMIZER ====================
    with tab_optimizer:
        st.subheader("⚡ Grid Search Parameter Optimizer")
        st.markdown("Search for optimal combinations of `Z_entry`, `Reversal Delta δ`, and `Z_max`.")
        
        if st.button("🚀 Run Parameter Grid Search", type="primary"):
            opt = ParameterOptimizer(cfg)
            with st.spinner("Evaluating parameter combinations..."):
                opt_df = opt.grid_search(
                    df_metrics,
                    z_entries=[1.2, 1.5, 1.8, 2.0],
                    reversal_deltas=[0.05, 0.10, 0.15, 0.20],
                    z_max_alloweds=[3.5, 4.0, 4.5],
                    enable_lockouts=[True, False],
                    enable_emergency_exits=[False, True]
                )
            st.success(f"Evaluated {len(opt_df)} parameter combinations!")
            st.dataframe(opt_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
