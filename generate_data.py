import sys
import os
import json
from datetime import datetime

import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))

from sp500_heatmap import InteractiveSP500Heatmap

def get_latest_trading_date():
    """Use SPY as the canonical reference for the most recent US trading day."""
    spy = yf.Ticker('SPY').history(period='5d')
    if len(spy) == 0:
        return None
    return spy.index[-1].strftime('%Y-%m-%d')

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
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'trading_date': get_latest_trading_date(),
        'companies': records,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'data.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(records)} companies to {out_path}")

if __name__ == '__main__':
    main()
