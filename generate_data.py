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

# (label, ticker, unit-suffix shown in the marquee — empty = none)
COMMODITY_TICKERS = [
    ('Silver',      'SI=F', '/oz'),
    ('Gold',        'GC=F', '/oz'),
    ('Brent Crude', 'BZ=F', '/bbl'),
    ('WTI Crude',   'CL=F', '/bbl'),
    ('Corn',        'ZC=F', '¢/bu'),
    ('Soybean',     'ZS=F', '¢/bu'),
    ('Sugar',       'SB=F', '¢/lb'),
    ('Live Cattle', 'LE=F', '¢/lb'),
    ('Lean Hogs',   'HE=F', '¢/lb'),
]

def get_commodity_data():
    """Fetch spot prices + daily change for the marquee tickers. Skip any that fail."""
    out = []
    for label, ticker, unit in COMMODITY_TICKERS:
        try:
            h = yf.Ticker(ticker).history(period='5d')
            if len(h) < 2:
                print(f"  skip {ticker}: only {len(h)} rows")
                continue
            last = float(h['Close'].iloc[-1])
            prev = float(h['Close'].iloc[-2])
            chg = (last - prev) / prev * 100 if prev else 0.0
            out.append({
                'label': label,
                'symbol': ticker,
                'unit': unit,
                'price': last,
                'change_pct': chg,
            })
        except Exception as e:
            print(f"  skip {ticker}: {e}")
    return out

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

    print("Fetching commodity prices...")
    commodities = get_commodity_data()
    print(f"  got {len(commodities)} commodities")

    output = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'trading_date': get_latest_trading_date(),
        'commodities': commodities,
        'companies': records,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'data.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(records)} companies to {out_path}")

if __name__ == '__main__':
    main()
