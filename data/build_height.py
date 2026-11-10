"""Build measured height priors from NHANES 2021-2023.

The existing height percentiles came from self-reported OKCupid profiles.  NHANES
measures height with a stadiometer and also records age, sex, race/ethnicity,
education and family income-to-poverty ratio.  This builder produces a small JSON
with measured percentiles and the measured height-income gradient.

NHANES is a complex, stratified sample.  The percentiles below are unweighted
descriptive statistics from the public files; they are not a national prevalence
estimate.  The sample weights are kept in the source, not in the output.

  data/height_priors.json

  python data/build_height.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "raw_nhanes", "DEMO_L.xpt")
BMX = os.path.join(HERE, "raw_nhanes", "BMX_L.xpt")
OUT = os.path.join(HERE, "height_priors.json")

AGE_BANDS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 80)]
RACE_LABELS = {
    1: "mexican american", 2: "other hispanic", 3: "white",
    4: "black", 5: "asian", 6: "other", 7: "multi",
}
EDU_LABELS = {
    1: "less than 9th grade", 2: "9-11th grade", 3: "high school",
    4: "some college", 5: "college graduate",
}
INCOME_BANDS = [(0.0, 1.0, "under 1x poverty"),
                (1.0, 2.0, "1-2x poverty"),
                (2.0, 3.0, "2-3x poverty"),
                (3.0, 4.0, "3-4x poverty"),
                (4.0, 5.0, "4-5x poverty"),
                (5.0, 99.0, "5x+ poverty")]


def age_band(age):
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "65-80"


def income_band(pir):
    if not np.isfinite(pir):
        return None
    for lo, hi, label in INCOME_BANDS:
        if lo <= pir < hi:
            return label
    return None


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
        "p05": q(values, 5),
        "p25": q(values, 25),
        "p75": q(values, 75),
        "p95": q(values, 95),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def main():
    if not (os.path.exists(DEMO) and os.path.exists(BMX)):
        raise SystemExit("missing NHANES DEMO_L.xpt or BMX_L.xpt in data/raw_nhanes")
    demo = pd.read_sas(DEMO, format="xport", encoding="utf-8")
    bmx = pd.read_sas(BMX, format="xport", encoding="utf-8")
    df = demo.merge(bmx[["SEQN", "BMXHT"]], on="SEQN", how="inner")
    df = df[(df["RIDAGEYR"] >= 18) & (df["RIDAGEYR"] <= 80)]
    df = df[df["BMXHT"].between(120, 220)]
    df = df.dropna(subset=["BMXHT", "RIAGENDR", "RIDAGEYR"])
    df["gender"] = df["RIAGENDR"].map({1: "men", 2: "women"})
    df = df.dropna(subset=["gender"])
    df["age_band"] = df["RIDAGEYR"].map(age_band)
    df["height_in"] = df["BMXHT"] / 2.54
    df["race"] = df["RIDRETH3"].map(RACE_LABELS)
    df["education"] = df["DMDEDUC2"].map(EDU_LABELS)
    df["income_band"] = df["INDFMPIR"].map(income_band)

    out = {
        "source": "NHANES August 2021-August 2023, DEMO_L + BMX_L",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/",
        "n": int(len(df)),
        "definition": "measured standing height, adults 18-80; unweighted descriptive",
        "height_in": {},
        "height_cm": {},
        "by_age": {},
        "by_age_unit": "cm",
        "by_race": {},
        "by_education": {},
        "by_income": {},
        "income_correlation": {},
    }
    for gender, g in df.groupby("gender"):
        out["height_in"][gender] = describe(g["height_in"])
        out["height_cm"][gender] = describe(g["BMXHT"])
        out["by_age"][gender] = {
            band: describe(bg["BMXHT"]) for band, bg in g.groupby("age_band")
        }
        out["by_race"][gender] = {
            race: describe(rg["height_in"])
            for race, rg in g.groupby("race") if len(rg) >= 30
        }
        out["by_education"][gender] = {
            edu: describe(eg["height_in"])
            for edu, eg in g.groupby("education") if len(eg) >= 30
        }
        out["by_income"][gender] = {
            band: describe(ig["height_in"])
            for band, ig in g.groupby("income_band") if len(ig) >= 30
        }
        ok = g[["height_in", "INDFMPIR"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ok) > 2:
            out["income_correlation"][gender] = {
                "n": int(len(ok)),
                "pearson_r_height_vs_poverty_ratio": float(
                    np.corrcoef(ok["height_in"], ok["INDFMPIR"])[0, 1]),
            }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {OUT}  n={out['n']:,}")
    for gender in ("men", "women"):
        h = out["height_in"][gender]
        r = out["income_correlation"].get(gender, {}).get(
            "pearson_r_height_vs_poverty_ratio")
        print(f"  {gender:<6} median {h['median']:.1f} in "
              f"({h['median'] * 2.54:.1f} cm), n={h['n']:,}, "
              f"height-income r={r:+.3f}" if r is not None else
              f"  {gender:<6} median {h['median']:.1f} in, n={h['n']:,}")


if __name__ == "__main__":
    sys.exit(main())
