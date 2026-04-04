import sys
import os
import json
from datetime import datetime

# Add heatmaps dir to path so we can import the existing module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'heatmaps'))

from sp500_heatmap import InteractiveSP500Heatmap

def main():
    heatmap = InteractiveSP500Heatmap()
    df = heatmap.prepare_data()

    # Compute market_value the same way as create_heatmap
    if 'market_cap' in df.columns and df['market_cap'].notna().any():
        df['market_value'] = df['market_cap'].fillna(df['current_price'] * df['volume'])
    else:
        df['market_value'] = df['current_price'] * df['volume']

    # Convert numpy types to native Python for JSON serialization
    records = []
    for _, row in df.iterrows():
        records.append({
            'symbol': str(row['symbol']),
            'company': str(row['company']),
            'sector': str(row['sector']),
            'change_pct': float(row['change_pct']),
            'current_price': float(row['current_price']),
            'volume': float(row['volume']),
            'market_value': float(row['market_value']),
        })

    output = {
        'generated_at': datetime.now().isoformat(),
        'companies': records,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'data.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(records)} companies to {out_path}")

if __name__ == '__main__':
    main()
