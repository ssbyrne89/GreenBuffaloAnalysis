import csv
import io
import json
import os
from datetime import datetime

import requests

# Census Population Estimates Program, Vintage 2024 (year ending July 1, 2024).
# Authoritative net-domestic-migration figures by state.
# Source: https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
CENSUS_CSV_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2024/state/totals/NST-EST2024-ALLDATA.csv"
)
CENSUS_VINTAGE_YEAR = 2024
CENSUS_SOURCE_LABEL = "Census PEP Vintage 2024 (year ending Jul 1, 2024)"

# U-Haul Growth Index 2025 (calendar year 2025), released January 2026.
# Behavioral signal from one-way truck rentals. U-Haul publishes the full
# 1-50 ranking but not raw flow numbers per state.
# Source: https://www.uhaul.com/Articles/About/U-Haul-Growth-Index-Texas-Back-ON-Top-As-No-1-Growth-State-Of-2025-36556/
UHAUL_SOURCE_URL = (
    "https://www.uhaul.com/Articles/About/"
    "U-Haul-Growth-Index-Texas-Back-ON-Top-As-No-1-Growth-State-Of-2025-36556/"
)
UHAUL_SOURCE_LABEL = "U-Haul Growth Index 2025"
UHAUL_RANKINGS_2025 = [
    "Texas", "Florida", "North Carolina", "Tennessee", "South Carolina",
    "Washington", "Arizona", "Idaho", "Alabama", "Georgia",
    "Oregon", "Montana", "Arkansas", "Oklahoma", "Maine",
    "Utah", "Kentucky", "South Dakota", "Minnesota", "Nevada",
    "Mississippi", "New Mexico", "Colorado", "Vermont", "Indiana",
    "Wisconsin", "Alaska", "Hawaii", "Missouri", "Wyoming",
    "Louisiana", "New Hampshire", "Delaware", "West Virginia", "Iowa",
    "Virginia", "Kansas", "North Dakota", "Nebraska", "Rhode Island",
    "Michigan", "Connecticut", "Ohio", "Pennsylvania", "Maryland",
    "Massachusetts", "New York", "New Jersey", "Illinois", "California",
]

STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}


def fetch_census_migration():
    """Pull NST-EST2024-ALLDATA.csv and return {state_name: net_domestic_migration_2024}."""
    resp = requests.get(CENSUS_CSV_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    col = f"DOMESTICMIG{CENSUS_VINTAGE_YEAR}"
    out = {}
    for row in reader:
        # SUMLEV 040 = state-level rows (excludes national/region/division rollups).
        if row.get("SUMLEV") != "040":
            continue
        name = row["NAME"]
        if name not in STATE_CODES:
            continue
        out[name] = int(row[col])
    return out


def build_records(census_migration):
    uhaul_rank = {state: i + 1 for i, state in enumerate(UHAUL_RANKINGS_2025)}
    records = []
    for state, code in STATE_CODES.items():
        records.append({
            "state": state,
            "state_code": code,
            "census_net_migration": census_migration.get(state),
            "uhaul_rank_2025": uhaul_rank.get(state),
        })
    records.sort(key=lambda r: (r["census_net_migration"] is None, -(r["census_net_migration"] or 0)))
    return records


def main():
    print(f"Fetching Census PEP Vintage {CENSUS_VINTAGE_YEAR}...")
    census = fetch_census_migration()
    print(f"  got {len(census)} state rows")

    records = build_records(census)
    missing_census = [r["state"] for r in records if r["census_net_migration"] is None]
    missing_uhaul = [r["state"] for r in records if r["uhaul_rank_2025"] is None]
    if missing_census:
        print(f"  WARNING: no Census value for {missing_census}")
    if missing_uhaul:
        print(f"  WARNING: no U-Haul rank for {missing_uhaul}")

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sources": {
            "census": {
                "label": CENSUS_SOURCE_LABEL,
                "url": CENSUS_CSV_URL,
                "field": f"DOMESTICMIG{CENSUS_VINTAGE_YEAR}",
                "units": "persons (net domestic migration)",
            },
            "uhaul": {
                "label": UHAUL_SOURCE_LABEL,
                "url": UHAUL_SOURCE_URL,
                "units": "rank 1-50 (1 = highest net inflow of one-way rentals)",
            },
        },
        "states": records,
    }

    out_path = os.path.join(os.path.dirname(__file__), "migration.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(records)} states to {out_path}")


if __name__ == "__main__":
    main()
