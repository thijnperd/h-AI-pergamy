"""Build measured couple-market priors from the American Community Survey.

The speed-dating file has decisions but no partner income and no ethnicity beyond
``samerace``.  This builder adds the other half of the picture: who actually lives
with whom, from the 2023 ACS 1-year Public Use Microdata Sample.

Scope and honesty:

* Five large states (CA, TX, FL, NY, IL), 2023 1-year PUMS.
* Opposite-sex married and unmarried couples where both partners are 18-80.
* One row per partner, so ``partner_age`` and ``partner_income`` are directional.
* These are **realised couples**, not dating decisions.  They answer "what does the
  couple market look like", never "would this person say yes on a date".
* Counts are unweighted sample counts.  Medians are unweighted; the PUMS weights are
  supplied in the raw files but this file does not claim to be a national estimate.

The raw files live in ``data/raw_acs/`` (gitignored).  The output is small:

  data/couple_priors.json

  python data/build_couples.py
  python data/build_couples.py --states ca,tx
"""

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_acs")
OUT = os.path.join(HERE, "couple_priors.json")

STATE_FILES = {
    "ca": ("csv_pca.zip", "psam_p06.csv"),
    "tx": ("csv_ptx.zip", "psam_p48.csv"),
    "fl": ("csv_pfl.zip", "psam_p12.csv"),
    "ny": ("csv_pny.zip", "psam_p36.csv"),
    "il": ("csv_pil.zip", "psam_p17.csv"),
}
COLUMNS = ["SERIALNO", "SPORDER", "RELSHIPP", "SEX", "AGEP", "PINCP",
           "SCHL", "RAC1P", "HISP", "MAR", "PWGTP", "STATE"]
REF = 20
PARTNER_CODES = {21: "married", 22: "cohabiting", 23: "married", 24: "cohabiting"}

AGE_BANDS = [(18, 24), (25, 29), (30, 34), (35, 44), (45, 54), (55, 64), (65, 80)]
INCOME_BANDS = [(0, 14999), (15000, 29999), (30000, 49999),
                (50000, 79999), (80000, 10 ** 9)]
EDU_LABELS = {1: "high school or less", 2: "some college",
              3: "bachelor", 4: "graduate"}
ETH_LABELS = ["white", "black", "latino", "asian", "pacific", "other"]


def age_band(age):
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "65-80"


def income_band(income):
    for lo, hi in INCOME_BANDS:
        if lo <= income <= hi:
            return f"{lo // 1000}-{hi // 1000}k" if hi < 10 ** 8 else f"{lo // 1000}k+"
    return "0-15k"


def ethnicity(hisp, race):
    if hisp != 1:
        return "latino"
    if race == 1:
        return "white"
    if race == 2:
        return "black"
    if race == 6:
        return "asian"
    if race == 7:
        return "pacific"
    return "other"


def education(schl):
    if schl <= 16:
        return 1
    if schl <= 20:
        return 2
    if schl == 21:
        return 3
    return 4


def as_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def read_state(path, member):
    with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
        df = pd.read_csv(fh, usecols=COLUMNS, dtype={"SERIALNO": str},
                         low_memory=False)
    # The PUMS columns are numeric codes; keep them numeric and small.
    for col in ("RELSHIPP", "SEX", "AGEP", "SCHL", "RAC1P", "HISP", "MAR"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("PINCP", "PWGTP"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def couples_from_state(df, state):
    """Return one row per partner in an opposite-sex reference/spouse pair."""
    ref = df[df["RELSHIPP"] == REF].drop_duplicates("SERIALNO")
    part = df[df["RELSHIPP"].isin(PARTNER_CODES)].drop_duplicates("SERIALNO")
    if ref.empty or part.empty:
        return pd.DataFrame()

    pair = ref.merge(part, on="SERIALNO", suffixes=("_own", "_p"))
    pair = pair[pair["SEX_own"] != pair["SEX_p"]]
    if pair.empty:
        return pd.DataFrame()

    pair["state"] = state.upper()
    pair["couple_type"] = pair["RELSHIPP_p"].map(PARTNER_CODES)
    pair["married"] = (pair["couple_type"] == "married").astype(int)

    # Both directions: each partner is a respondent and the other is the partner.
    a = pd.DataFrame({
        "own_sex": pair["SEX_own"], "own_age": pair["AGEP_own"],
        "own_income": pair["PINCP_own"], "own_educ": pair["SCHL_own"],
        "own_race": pair["RAC1P_own"], "own_hisp": pair["HISP_own"],
        "partner_age": pair["AGEP_p"], "partner_income": pair["PINCP_p"],
        "partner_educ": pair["SCHL_p"], "partner_race": pair["RAC1P_p"],
        "partner_hisp": pair["HISP_p"], "weight": pair["PWGTP_own"],
        "state": pair["state"], "couple_type": pair["couple_type"],
        "married": pair["married"],
    })
    b = pd.DataFrame({
        "own_sex": pair["SEX_p"], "own_age": pair["AGEP_p"],
        "own_income": pair["PINCP_p"], "own_educ": pair["SCHL_p"],
        "own_race": pair["RAC1P_p"], "own_hisp": pair["HISP_p"],
        "partner_age": pair["AGEP_own"], "partner_income": pair["PINCP_own"],
        "partner_educ": pair["SCHL_own"], "partner_race": pair["RAC1P_own"],
        "partner_hisp": pair["HISP_own"], "weight": pair["PWGTP_p"],
        "state": pair["state"], "couple_type": pair["couple_type"],
        "married": pair["married"],
    })
    return pd.concat([a, b], ignore_index=True)


def clean(rows):
    for key in ("own_age", "partner_age"):
        rows = rows[(rows[key] >= 18) & (rows[key] <= 80)]
    rows = rows.dropna(subset=["own_age", "partner_age", "own_sex"])
    rows = rows.copy()
    rows["own_income"] = rows["own_income"].fillna(0).clip(lower=0, upper=1_000_000)
    rows["partner_income"] = rows["partner_income"].fillna(0).clip(lower=0, upper=1_000_000)
    rows["age_gap"] = rows["partner_age"] - rows["own_age"]
    # +1000 keeps ratios finite when both partners report no personal income.
    rows["income_ratio"] = ((rows["partner_income"] + 1000.0)
                            / (rows["own_income"] + 1000.0))
    rows["own_eth"] = [ethnicity(h, r) for h, r in zip(rows["own_hisp"], rows["own_race"])]
    rows["partner_eth"] = [ethnicity(h, r) for h, r in zip(rows["partner_hisp"],
                                                            rows["partner_race"])]
    rows["own_edu"] = rows["own_educ"].map(education)
    rows["partner_edu"] = rows["partner_educ"].map(education)
    rows["same_eth"] = (rows["own_eth"] == rows["partner_eth"]).astype(float)
    rows["same_edu"] = (rows["own_edu"] == rows["partner_edu"]).astype(float)
    rows["gender"] = rows["own_sex"].map({1: "men", 2: "women"})
    rows["age_band"] = rows["own_age"].map(age_band)
    rows["income_band"] = rows["own_income"].map(income_band)
    return rows.dropna(subset=["gender", "own_edu", "partner_edu"])


def q(values, p):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, p)) if len(values) else None


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "median": q(values, 50),
        "p25": q(values, 25),
        "p75": q(values, 75),
        "mean": float(values.mean()),
    }


def group_context(group):
    top = Counter(group["partner_eth"]).most_common(1)
    return {
        "n": int(len(group)),
        "partner_age": describe(group["partner_age"]),
        "partner_income": describe(group["partner_income"]),
        "age_gap": describe(group["age_gap"]),
        "income_ratio": describe(group["income_ratio"]),
        "same_eth_pct": round(100.0 * float(group["same_eth"].mean()), 2),
        "same_edu_pct": round(100.0 * float(group["same_edu"].mean()), 2),
        "top_partner_eth": top[0][0] if top else None,
        "top_partner_eth_pct": round(100.0 * top[0][1] / len(group), 2) if top else None,
    }


def build(rows):
    out = {
        "source": "ACS 2023 1-year PUMS, CA/TX/FL/NY/IL, opposite-sex couples 18-80",
        "url": "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/",
        "n_rows": int(len(rows)),
        "n_couples": int(len(rows) // 2),
        "states": sorted(rows["state"].unique().tolist()),
        "definition": "realised married/cohabiting couples, not dating decisions",
        "overall": {},
        "market": {},
        "ethnicity": {},
        "education": {},
        "age_gap": {},
    }
    for gender, g in rows.groupby("gender"):
        out["overall"][gender] = {
            "n": int(len(g)),
            "median_age_gap": q(g["age_gap"], 50),
            "same_eth_pct": round(100.0 * float(g["same_eth"].mean()), 2),
            "same_edu_pct": round(100.0 * float(g["same_edu"].mean()), 2),
            "married_pct": round(100.0 * float(g["married"].mean()), 2),
        }
        out["age_gap"][gender] = describe(g["age_gap"])

        market = {}
        for band, bg in g.groupby("age_band"):
            market[band] = {}
            for ib, ig in bg.groupby("income_band"):
                if len(ig) >= 50:
                    market[band][ib] = group_context(ig)
        out["market"][gender] = market

        eth = {}
        for name, eg in g.groupby("own_eth"):
            if len(eg) >= 50:
                eth[name] = group_context(eg)
        out["ethnicity"][gender] = eth

        edu = {}
        for level, eg in g.groupby("own_edu"):
            if len(eg) >= 50:
                edu[EDU_LABELS[int(level)]] = group_context(eg)
        out["education"][gender] = edu

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--states", default="ca,tx,fl,ny,il",
                        help="comma-separated state keys, e.g. ca,tx")
    args = parser.parse_args()
    states = [s.strip().lower() for s in args.states.split(",") if s.strip()]

    frames = []
    for state in states:
        if state not in STATE_FILES:
            raise SystemExit(f"unknown state {state!r}; choose from {sorted(STATE_FILES)}")
        archive, member = STATE_FILES[state]
        path = os.path.join(RAW, archive)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; download the 2023 1-year PUMS file first")
        print(f"reading {state}: {archive}/{member} ...")
        df = read_state(path, member)
        part = couples_from_state(df, state)
        print(f"  {state}: {len(df):,} person rows -> {len(part) // 2:,} couples")
        frames.append(part)

    rows = clean(pd.concat(frames, ignore_index=True))
    out = build(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}")
    print(f"  {out['n_couples']:,} couples / {out['n_rows']:,} partner rows")
    print(f"  states: {', '.join(out['states'])}")
    for gender in ("men", "women"):
        g = out["overall"][gender]
        print(f"  {gender:<6} n={g['n']:,}  median age gap {g['median_age_gap']:+.1f}  "
              f"same ethnicity {g['same_eth_pct']:.1f}%  same education {g['same_edu_pct']:.1f}%")


if __name__ == "__main__":
    main()
