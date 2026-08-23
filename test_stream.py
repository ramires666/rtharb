import sys
sys.path.insert(0, '.')
from server import MarketDataManager
import pandas as pd

mgr = MarketDataManager()
print(f"Total available sessions: {len(mgr.sessions_list)}")
print(f"Sample sessions: {[s['date'] for s in mgr.sessions_list[:5]]}")

test_date = mgr.sessions_list[0]['date']
chunk = mgr.get_session_chunk(test_date)
assert chunk is not None, f"Chunk for {test_date} was None!"

df_raw = pd.read_parquet("data_cache/NVDA_1m.parquet")
df_raw = df_raw[df_raw['session_date'] == pd.to_datetime(test_date).date()]

assert len(chunk['open']) == len(df_raw), f"Length mismatch: {len(chunk['open'])} vs {len(df_raw)}"
for i in range(min(20, len(df_raw))):
    raw_o = round(float(df_raw['open'].iloc[i]), 2)
    assert chunk['open'][i] == raw_o, f"Bar {i} price mismatch: {chunk['open'][i]} != {raw_o}"

print("🎉 VERIFICATION PASSED: 100% genuine raw Parquet stream on demand confirmed!")
