import json
import os
from datetime import datetime, date

import yfinance as yf

YTD_START = date(2026, 1, 1)

# User's holdings across Acorns, Fidelity, and Vanguard. Yahoo Finance tickers.
#   (yahoo_ticker, display_name, asset_class)
HOLDINGS = [
    ("SI=F",    "Silver (spot, COMEX continuous)",         "Commodity"),
    ("BTC-USD", "Bitcoin",                                  "Crypto"),
    ("ETH-USD", "Ethereum",                                 "Crypto"),
    ("VEIPX",   "Vanguard Equity-Income Fund",              "Mutual Fund"),
    ("VDIGX",   "Vanguard Dividend Growth Fund",            "Mutual Fund"),
    ("VYMI",    "Vanguard International High Div Yield",    "ETF"),
    ("NUDV",    "Nuveen ESG Dividend ETF",                  "ETF"),
    ("DIV",     "Global X SuperDividend U.S. ETF",          "ETF"),
    ("SGMO",    "Sangamo Therapeutics",                     "Stock"),
    ("PROK",    "ProKidney Corp.",                          "Stock"),
    ("HST",     "Host Hotels & Resorts",                    "Stock"),
    ("ET",      "Energy Transfer LP",                       "Stock"),
    ("EPM",     "Evolution Petroleum",                      "Stock"),
    ("EGY",     "VAALCO Energy",                            "Stock"),
    ("FXAIX",   "Fidelity 500 Index Fund",                  "Mutual Fund"),
    ("FBCGX",   "Fidelity Blue Chip Growth Fund K",         "Mutual Fund"),
    ("FSPSX",   "Fidelity International Index Fund",        "Mutual Fund"),
    ("AALRX",   "American Funds AMCAP Fund R-1",            "Mutual Fund"),
    ("TRRKX",   "T. Rowe Price Retirement 2065 Fund",       "Mutual Fund"),
    ("CSRSX",   "Cohen & Steers Realty Shares",             "Mutual Fund"),
    ("RERAX",   "American Funds EuroPacific Growth R-1",    "Mutual Fund"),
    ("VOO",     "Vanguard S&P 500 ETF",                     "ETF"),
    ("IJH",     "iShares Core S&P Mid-Cap ETF",             "ETF"),
    ("IJR",     "iShares Core S&P Small-Cap ETF",           "ETF"),
    ("IXUS",    "iShares Core MSCI Total Intl Stock ETF",   "ETF"),
]


def fetch_holding(ticker_symbol, display_name, asset_class):
    t = yf.Ticker(ticker_symbol)

    # Pull YTD daily history. auto_adjust=False keeps Open/Close as raw quotes
    # (matters for dividend accounting; we account for dividends separately).
    hist = t.history(start=YTD_START.isoformat(), auto_adjust=False)
    if hist is None or len(hist) < 1:
        print(f"    skip: no YTD history")
        return None

    ytd_open = float(hist["Open"].iloc[0])
    current = float(hist["Close"].iloc[-1])
    period_end = hist.index[-1].date().isoformat()

    # Sum dividend distributions paid YTD. Cap-gain distributions for mutual
    # funds may not be included in yfinance's `dividends` series — accept that
    # limitation and label the column "Dividends YTD" rather than "Total Distros".
    divs = t.dividends
    div_total = 0.0
    last_div_amount = None
    last_div_date = None
    payout_schedule = None
    if divs is not None and len(divs) > 0:
        if divs.index.tz is not None:
            divs = divs.copy()
            divs.index = divs.index.tz_localize(None)
        start_ts = datetime(YTD_START.year, YTD_START.month, YTD_START.day)
        ytd_divs = divs[divs.index >= start_ts]
        div_total = float(ytd_divs.sum())
        last_div_amount = float(divs.iloc[-1])
        last_div_date = divs.index[-1].date().isoformat()
        payout_schedule = classify_payout_schedule(divs)

    change_dollar = current - ytd_open
    change_pct = (change_dollar / ytd_open * 100) if ytd_open else None

    return {
        "ticker": ticker_symbol,
        "name": display_name,
        "asset_class": asset_class,
        "current_price": round(current, 4),
        "ytd_open_price": round(ytd_open, 4),
        "ytd_change_dollar": round(change_dollar, 4),
        "ytd_change_pct": round(change_pct, 2) if change_pct is not None else None,
        "ytd_dividend_per_share": round(div_total, 4),
        "payout_schedule": payout_schedule,
        "last_dividend_amount": round(last_div_amount, 4) if last_div_amount is not None else None,
        "last_dividend_date": last_div_date,
        "as_of": period_end,
    }


def classify_payout_schedule(divs):
    """Infer how often this security distributes dividends.

    Looks at the last ~2 years of payout dates and bins the median gap into
    Monthly / Quarterly / Trimesterly (3x/yr) / Bi-annually / Annually.
    """
    if divs is None or len(divs) < 2:
        return None
    recent = divs.tail(12) if len(divs) >= 12 else divs
    deltas = recent.index.to_series().diff().dt.days.dropna().tolist()
    if not deltas:
        return None
    median_days = sorted(deltas)[len(deltas) // 2]
    if median_days <= 45:
        return "Monthly"
    if median_days <= 100:
        return "Quarterly"
    if median_days <= 150:
        return "Trimesterly"
    if median_days <= 240:
        return "Bi-annually"
    return "Annually"


def main():
    print(f"Fetching {len(HOLDINGS)} holdings (YTD baseline {YTD_START.isoformat()})...")
    rows = []
    for ticker, name, asset_class in HOLDINGS:
        print(f"  {ticker:8s}  {name}")
        try:
            rec = fetch_holding(ticker, name, asset_class)
            if rec:
                rows.append(rec)
        except Exception as e:
            print(f"    error: {e}")

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "ytd_start": YTD_START.isoformat(),
        "platforms": ["Acorns", "Fidelity", "Vanguard"],
        "strategy_note": (
            "My investment strategy is to go dividend-heavy so if the prices "
            "go down then I still get a return."
        ),
        "holdings": rows,
    }

    out_path = os.path.join(os.path.dirname(__file__), "portfolio.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(rows)}/{len(HOLDINGS)} holdings to {out_path}")


if __name__ == "__main__":
    main()
