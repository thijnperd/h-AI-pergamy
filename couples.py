"""Measured couple-market context from the 2023 ACS.

This module deliberately does **not** feed a date-acceptance probability.  The ACS
records who is married or cohabiting, not who said yes on a date.  What it can do is
put the speed-dating estimate next to the real couple market: typical partner age,
partner income, age gap, same-ethnicity share, same-education share, and a measured
height percentile from NHANES.

The data are built by:

  python data/build_couples.py
  python data/build_height.py

and are small enough to commit.  If the JSON is missing, `context()` returns None and
the GUI simply omits the panel.
"""

import json
import os
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
COUPLE_PATH = os.path.join(HERE, "data", "couple_priors.json")
HEIGHT_PATH = os.path.join(HERE, "data", "height_priors.json")

AGE_BANDS = [(18, 24), (25, 29), (30, 34), (35, 44), (45, 54), (55, 64), (65, 80)]
INCOME_BANDS = [(0, 14999), (15000, 29999), (30000, 49999),
                (50000, 79999), (80000, 10 ** 9)]
ETHNICITIES = ["white", "black", "latino", "asian", "pacific", "other"]


@lru_cache(maxsize=1)
def priors():
    try:
        with open(COUPLE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def height_priors():
    try:
        with open(HEIGHT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


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


def _gender_key(gender):
    return "men" if gender else "women"


def _stat(cell, key):
    value = cell.get(key) if cell else None
    if not isinstance(value, dict):
        return None
    return {
        "n": int(value.get("n", 0)),
        "median": value.get("median"),
        "p25": value.get("p25"),
        "p75": value.get("p75"),
    }


def _height_percentile_cm(height_cm, gender, age=None):
    p = height_priors()
    if not p or not height_cm:
        return None
    key = _gender_key(gender)
    values = p.get("height_cm", {}).get(key, {})
    if age is not None:
        band = age_band(age)
        age_values = p.get("by_age", {}).get(key, {}).get(band)
        if age_values and age_values.get("n", 0) >= 30:
            values = age_values
    if not values:
        return None
    points = [(values.get("p05"), 5.0), (values.get("p25"), 25.0),
              (values.get("median"), 50.0), (values.get("p75"), 75.0),
              (values.get("p95"), 95.0)]
    points = [(float(x), float(y)) for x, y in points
              if x is not None and y is not None]
    if len(points) < 2:
        return None
    points.sort()
    if height_cm <= points[0][0]:
        return 1.0
    if height_cm >= points[-1][0]:
        return 99.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= height_cm <= x1:
            span = (height_cm - x0) / max(x1 - x0, 1e-9)
            return y0 + span * (y1 - y0)
    return 50.0


def height_context(profile, gender):
    p = height_priors()
    if not p:
        return None
    key = _gender_key(gender)
    values = p.get("height_cm", {}).get(key, {})
    h = profile.get("height_cm")
    if not values:
        return None
    pct = _height_percentile_cm(float(h), gender, profile.get("age")) if h else None
    return {
        "n": int(values.get("n", 0)),
        "median_cm": values.get("median"),
        "p05_cm": values.get("p05"),
        "p95_cm": values.get("p95"),
        "your_percentile": pct,
        "income_correlation": p.get("income_correlation", {}).get(
            key, {}).get("pearson_r_height_vs_poverty_ratio"),
        "source": p.get("source"),
    }


def _fallback_cell(gender_key, age, income):
    """Exact age x income cell, else the nearest income cell in that age band."""
    p = priors()
    market = p.get("market", {}).get(gender_key, {})
    band = age_band(age)
    cells = market.get(band, {})
    exact = cells.get(income_band(income))
    if exact:
        return exact, band, income_band(income), "exact age x income cell"
    # Pick the income cell with the largest n instead of inventing an average.
    candidates = [(k, v) for k, v in cells.items() if v.get("n", 0) >= 50]
    if candidates:
        key, cell = max(candidates, key=lambda kv: kv[1].get("n", 0))
        return cell, band, key, "nearest income cell with n >= 50"
    # Last fallback: the pooled gender distribution, explicitly labelled.
    return None, band, income_band(income), "no age/income cell; using overall"



def _education_cell(table, name):
    """ACS education table keyed by group; tolerate either label spelling."""
    if not table or not name:
        return None
    if name in table:
        return table[name]
    first = str(name).split()[0].lower()
    for key, value in table.items():
        if first and (first in key.lower() or key.lower() in str(name).lower()):
            return value
    return None

def context(profile, gender, partner_gender):
    """Measured ACS/NHANES context for one profile, or None when data is absent."""
    p = priors()
    if not p:
        return None
    if partner_gender == gender:
        return {
            "available": False,
            "same_gender": True,
            "note": "ACS couple priors here are opposite-sex only; no same-gender "
                    "market benchmark is shown.",
        }

    key = _gender_key(gender)
    age = float(profile.get("age", 24.0))
    income = float(profile.get("income", 32000.0))
    cell, band, ib, match = _fallback_cell(key, age, income)
    overall = p.get("overall", {}).get(key, {})
    eth_name = profile.get("ethnicity")
    if eth_name not in ETHNICITIES:
        eth_name = None
    eth_cell = p.get("ethnicity", {}).get(key, {}).get(eth_name) if eth_name else None

    def put(source, prefix=""):
        if not source:
            return None
        return {
            "n": int(source.get("n", 0)),
            "partner_age": _stat(source, "partner_age"),
            "partner_income": _stat(source, "partner_income"),
            "age_gap": _stat(source, "age_gap"),
            "income_ratio": _stat(source, "income_ratio"),
            "same_eth_pct": source.get("same_eth_pct"),
            "same_edu_pct": source.get("same_edu_pct"),
            "top_partner_eth": source.get("top_partner_eth"),
            "top_partner_eth_pct": source.get("top_partner_eth_pct"),
        }

    edu_name = profile.get("education")
    edu_cell = _education_cell(p.get("education", {}).get(key, {}), edu_name)

    out = {
        "available": True,
        "same_gender": False,
        "source": p.get("source"),
        "states": p.get("states"),
        "n_total": p.get("n_couples"),
        "definition": p.get("definition"),
        "age_band": band,
        "income_band": ib,
        "match": match,
        "cell": put(cell),
        "overall": put(overall),
        "ethnicity": eth_name,
        "ethnicity_cell": put(eth_cell),
        "education": edu_name,
        "education_cell": put(edu_cell),
        "height": height_context(profile, gender),
    }
    return out
