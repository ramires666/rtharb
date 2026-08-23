import json

with open("standalone_report/real_data.json", "r", encoding="utf-8") as f:
    d = json.load(f)

trades = d["trades"]
print("Total real trades in report:", len(trades))
wins = [t for t in trades if t["pnl"] > 0]
losses = [t for t in trades if t["pnl"] < 0]
print(f"Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%), Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")

print("\nSample 5 Losing Trades:")
for tr in losses[:5]:
    print(f"  #{tr['id']} {tr['dir']} {tr['entry_time']} @ {tr['entry_price']} -> {tr['exit_time']} @ {tr['exit_price']} PnL: {tr['pnl_str']} ({tr['return_pct']}) [{tr['reason']}]")

print(f"\nTotal real trading sessions: {len(d['days'])}")
first_day = d["days"][0]
day_data = d["daily"][first_day]
print(f"Session {first_day}: {len(day_data['times'])} bars")
print(f"Real NVDA Closes (first 5): {day_data['close'][:5]}")
print(f"Real Fair Prices (first 5): {day_data['fair'][:5]}")
print(f"Real Z-Scores (first 5): {day_data['z_score'][:5]}")
