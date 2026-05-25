import csv
import gzip
import io
import json
import os
import zipfile
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Zillow Research public CSVs (updated monthly).
#   ZHVI = Zillow Home Value Index, smoothed, seasonally adjusted, all homes
#          mid-tier (35th-65th percentile). Use as median-ish home value.
#   ZORI = Zillow Observed Rent Index, smoothed, all homes (SFR + condo + MFR).
#          Mean of asking rents in the 35th-65th percentile band.
# https://www.zillow.com/research/data/
# ---------------------------------------------------------------------------
ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
ZORI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv"
)

# ---------------------------------------------------------------------------
# Redfin Data Center monthly metro tracker (gzipped TSV, ~110 MB). Provides
# actual median sale price by metro, useful as a cross-check on Zillow ZHVI.
# https://www.redfin.com/news/data-center/
# ---------------------------------------------------------------------------
REDFIN_METRO_URL = (
    "https://redfin-public-data.s3.us-west-2.amazonaws.com/"
    "redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
)

# ---------------------------------------------------------------------------
# Census 2025 Gazetteer for CBSAs (Core Based Statistical Areas). Used only
# for centroid lat/lon so each metro can be plotted as a bubble on the map.
# https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
# ---------------------------------------------------------------------------
CBSA_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2025_Gazetteer/2025_Gaz_cbsa_national.zip"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
OUT_PATH = os.path.join(os.path.dirname(__file__), "rental.json")


def _cached_download(url, filename):
    """Download URL to CACHE_DIR/filename if missing. Returns local path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"  downloading {url}")
        resp = requests.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def fetch_zillow_metro(url, label):
    """Return {(city, state): latest_value} from a Zillow Metro_*.csv URL."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    reader = csv.reader(io.StringIO(resp.text))
    header = next(reader)
    month_cols = [(i, c) for i, c in enumerate(header) if c[:2] == "20" and "-" in c]
    name_i = header.index("RegionName")
    type_i = header.index("RegionType")
    state_i = header.index("StateName")
    out = {}
    latest_month = None
    for row in reader:
        if row[type_i] != "msa":
            continue
        name = row[name_i]
        if ", " not in name:
            continue
        city, state = name.rsplit(", ", 1)
        latest_val = None
        latest_col = None
        for i, col in reversed(month_cols):
            if i < len(row) and row[i]:
                try:
                    latest_val = float(row[i])
                    latest_col = col
                    break
                except ValueError:
                    continue
        if latest_val is None:
            continue
        out[(city.strip(), state.strip())] = latest_val
        if latest_month is None or (latest_col and latest_col > latest_month):
            latest_month = latest_col
    print(f"  {label}: {len(out)} metros, latest month {latest_month}")
    return out, latest_month


def fetch_redfin_metro():
    """Return {(city, state): (median_sale_price, period_end)} for the latest
    All Residential, 30-day duration row per metro."""
    path = _cached_download(REDFIN_METRO_URL, "redfin_metro.tsv.gz")
    out = {}
    latest_period = None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        for row in reader:
            if row["REGION_TYPE"].strip('"') != "metro":
                continue
            if row["PROPERTY_TYPE"].strip('"') != "All Residential":
                continue
            if row["PERIOD_DURATION"] != "30":
                continue
            try:
                price = float(row["MEDIAN_SALE_PRICE"])
            except (ValueError, TypeError):
                continue
            region = row["REGION"].strip('"')
            state = row["STATE_CODE"].strip('"')
            # REGION looks like "New York, NY metro area"; strip the suffix.
            city = region.replace(" metro area", "")
            if ", " in city:
                city = city.rsplit(", ", 1)[0]
            period_end = row["PERIOD_END"].strip('"')
            key = (city.strip(), state.strip())
            prev = out.get(key)
            if prev is None or period_end > prev[1]:
                out[key] = (price, period_end)
            if latest_period is None or period_end > latest_period:
                latest_period = period_end
    print(f"  Redfin: {len(out)} metros, latest period {latest_period}")
    return out, latest_period


def fetch_cbsa_centroids():
    """Return {(first_principal_city, first_state): (lat, lon, full_name)}."""
    path = _cached_download(CBSA_GAZETTEER_URL, "cbsa.zip")
    with zipfile.ZipFile(path) as z:
        inner = [n for n in z.namelist() if n.endswith(".txt")][0]
        text = z.read(inner).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    out = {}
    for row in reader:
        name = row["NAME"]  # e.g. "New York-Newark-Jersey City, NY-NJ Metro Area"
        if " Metro Area" not in name:
            continue
        core = name.replace(" Metro Area", "")
        if ", " not in core:
            continue
        cities_part, states_part = core.rsplit(", ", 1)
        first_city = cities_part.split("-")[0].strip()
        first_state = states_part.split("-")[0].strip()
        try:
            lat = float(row["INTPTLAT"])
            lon = float(row["INTPTLONG"])
        except ValueError:
            continue
        out[(first_city, first_state)] = (lat, lon, name.replace(" Metro Area", ""))
    print(f"  CBSA gazetteer: {len(out)} metros")
    return out


def build_records():
    print("Fetching Zillow ZHVI (home value)...")
    zhvi, zhvi_month = fetch_zillow_metro(ZHVI_URL, "ZHVI")
    print("Fetching Zillow ZORI (rent)...")
    zori, zori_month = fetch_zillow_metro(ZORI_URL, "ZORI")
    print("Fetching Redfin metro tracker (median sale price)...")
    redfin, redfin_period = fetch_redfin_metro()
    print("Fetching Census CBSA centroids...")
    cbsa = fetch_cbsa_centroids()

    keys = sorted(set(zhvi) & set(zori))
    records = []
    for key in keys:
        city, state = key
        zhvi_val = zhvi[key]
        zori_val = zori[key]
        annual_rent = zori_val * 12
        gross_yield = annual_rent / zhvi_val if zhvi_val else None
        redfin_price, redfin_end = redfin.get(key, (None, None))
        lat, lon, full_name = cbsa.get(key, (None, None, f"{city}, {state}"))
        records.append({
            "metro": f"{city}, {state}",
            "cbsa_name": full_name,
            "city": city,
            "state": state,
            "lat": lat,
            "lon": lon,
            "zhvi_home_value": round(zhvi_val),
            "zori_rent_monthly": round(zori_val),
            "annual_rent": round(annual_rent),
            "gross_yield_pct": round(gross_yield * 100, 2) if gross_yield else None,
            "redfin_median_sale_price": round(redfin_price) if redfin_price else None,
            "redfin_period_end": redfin_end,
        })
    # Sort by yield descending; metros with missing yield go to the end.
    records.sort(key=lambda r: (r["gross_yield_pct"] is None, -(r["gross_yield_pct"] or 0)))
    return records, zhvi_month, zori_month, redfin_period


def main():
    records, zhvi_month, zori_month, redfin_period = build_records()
    print(f"\nJoined {len(records)} metros (intersection of Zillow ZHVI + ZORI)")

    # Need lat/lon to plot; only include geo-resolved metros in top/bottom.
    geo_records = [r for r in records if r["lat"] is not None and r["gross_yield_pct"] is not None]
    top = geo_records[:15]
    bottom = geo_records[-15:]

    print("\nTop 15 by gross rental yield:")
    for r in top:
        print(f"  {r['gross_yield_pct']:5.2f}%  {r['metro']:<35} rent ${r['zori_rent_monthly']:>5}/mo  home ${r['zhvi_home_value']:>9,}")
    print("\nBottom 15 by gross rental yield:")
    for r in bottom:
        print(f"  {r['gross_yield_pct']:5.2f}%  {r['metro']:<35} rent ${r['zori_rent_monthly']:>5}/mo  home ${r['zhvi_home_value']:>9,}")

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sources": {
            "zhvi": {
                "label": f"Zillow Home Value Index (mid-tier, smoothed, SA) — {zhvi_month}",
                "url": ZHVI_URL,
                "units": "USD (typical home value)",
            },
            "zori": {
                "label": f"Zillow Observed Rent Index (all homes, smoothed) — {zori_month}",
                "url": ZORI_URL,
                "units": "USD per month",
            },
            "redfin": {
                "label": f"Redfin Data Center, metro median sale price (All Residential, 30-day) — {redfin_period}",
                "url": REDFIN_METRO_URL,
                "units": "USD (actual median sale price)",
            },
            "cbsa_gazetteer": {
                "label": "Census 2025 Gazetteer (CBSA centroids)",
                "url": CBSA_GAZETTEER_URL,
                "units": "decimal degrees",
            },
        },
        "note": (
            "Gross rental yield = (ZORI monthly rent x 12) / ZHVI home value. "
            "HUD Fair Market Rents were considered as a fourth source but excluded: "
            "FMR is a 40th-percentile voucher-determination figure, not a market median, "
            "and would systematically understate yields if mixed with median-based sources."
        ),
        "top_15_by_yield": top,
        "bottom_15_by_yield": bottom,
        "all_metros": records,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(records)} metros ({len(top)} top, {len(bottom)} bottom) to {OUT_PATH}")


if __name__ == "__main__":
    main()
