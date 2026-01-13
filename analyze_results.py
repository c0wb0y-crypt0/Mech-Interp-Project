import pandas as pd
import sys
from pathlib import Path

# Flexible input
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_files = list(Path('.').glob('results_*.csv'))
    if not csv_files:
        print("No results CSV found!")
        sys.exit(1)
    csv_path = csv_files[-1]  # Latest
    print(f"Using: {csv_path}")

df = pd.read_csv(csv_path)

# Clean
df['defensive_words'] = pd.to_numeric(df['defensive_words'], errors='coerce')
df['aggressive_words'] = pd.to_numeric(df['aggressive_words'], errors='coerce')
df['tokens_used'] = pd.to_numeric(df['tokens_used'], errors='coerce')

print(f"\n=== Analysis for {Path(csv_path).name} ===")
print(f"Rows: {len(df)} | Unique statements: {df['statement'].nunique()}\n")

# Stats
print("Overall:")
print(f"Avg Def Words: {df['defensive_words'].mean():.1f}")
print(f"Avg Agg Words: {df['aggressive_words'].mean():.1f}")
print(f"Avg Diff: {(df['defensive_words'] - df['aggressive_words']).mean():.1f}")
print(f"Agg Shorter %: {(df['aggressive_words'] < df['defensive_words']).mean()*100:.1f}%\n")

# Cost—auto-detect model
stem = Path(csv_path).stem
is_fast = 'fast' in stem.lower()
input_rate = 0.0000002 if is_fast else 0.000003  # ~$0.20/M fast input, $3/M flagship
output_rate = 0.0000005 if is_fast else 0.000015  # ~$0.50/M fast output, $15/M flagship
avg_rate = (input_rate + output_rate) / 2
total_tokens = df['tokens_used'].sum()
est_cost = total_tokens * avg_rate
print(f"Total tokens: {total_tokens:,}")
print(f"Est cost: ~${est_cost:.3f} ({'fast' if is_fast else 'flagship'} rates)\n")

# Per-statement
print("Per-Statement:")
for stmt in df['statement'].unique():
    sub = df[df['statement'] == stmt]
    repeats = len(sub)
    def_unique = sub['defensive'].nunique()
    agg_unique = sub['aggressive'].nunique()
    def_id_pct = (repeats - def_unique) / repeats * 100 if repeats else 0
    agg_id_pct = (repeats - agg_unique) / repeats * 100 if repeats else 0
    print(f"\n{stmt[:80]}...")
    print(f"  Repeats: {repeats} | Def unique: {def_unique}/{repeats} ({def_id_pct:.0f}% identical)")
    print(f"  Agg unique: {agg_unique}/{repeats} ({agg_id_pct:.0f}% identical)")
    print(f"  Avg words — Def: {sub['defensive_words'].mean():.0f} | Agg: {sub['aggressive_words'].mean():.0f}")

print("\nDone!")