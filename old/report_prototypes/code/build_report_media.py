"""Generate all artifact charts directly in the artifacts directory.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from server import MarketDataManager
from generate_pillow_charts import render_equity_drawdown_chart, render_intraday_session_chart
from generate_svg_charts import generate_equity_svg, generate_session_svg

artifact_dir = project_root / "report_charts"
artifact_dir.mkdir(parents=True, exist_ok=True)


def main():
    print("⏳ Initializing MarketDataManager...")
    mgr = MarketDataManager()

    print("📊 1. Rendering Equity & Drawdown Chart for 2026...")
    png_eq = artifact_dir / "equity_drawdown_2026.png"
    render_equity_drawdown_chart(mgr.eq_prod, png_eq)
    svg_eq = artifact_dir / "equity_drawdown_2026.svg"
    svg_eq.write_text(generate_equity_svg(mgr.eq_prod), encoding="utf-8")
    print(f"   ✅ Saved: {png_eq.name} ({png_eq.stat().st_size:,} bytes)")
    print(f"   ✅ Saved: {svg_eq.name} ({svg_eq.stat().st_size:,} bytes)")

    print("📊 2. Rendering August 2026 Trade Sessions...")
    august_dates = [s["date"] for s in mgr.sessions_list if s["date"].startswith("2026-08-") and s["trades_count"] > 0]
    print(f"   Found active August sessions: {august_dates}")

    for d in august_dates[:5]:
        chunk = mgr.get_session_chunk(d)
        if chunk:
            tag = d.replace("-", "_")
            png_file = artifact_dir / f"session_{tag}.png"
            render_intraday_session_chart(chunk, png_file)
            svg_file = artifact_dir / f"session_{tag}.svg"
            svg_file.write_text(generate_session_svg(chunk), encoding="utf-8")
            print(f"   ✅ Saved: {png_file.name} ({png_file.stat().st_size:,} bytes), {svg_file.name}")

    print("\n🎉 ALL ARTIFACT CHARTS SUCCESSFULLY GENERATED!")


if __name__ == "__main__":
    main()
