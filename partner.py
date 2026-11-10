"""Who you would most plausibly end up with -- measured, not forecast.

The Columbia speed-dating study records whether two people wanted a second date.
It does not follow anyone to a partner, a marriage or a breakup, so no model
trained on it can answer "who will I end up with".  No public dataset in this
repository follows a random person to a lifetime partner.  HCMST is the right
public panel for that question (see `research.SOURCES["hcmst"]`), but it needs
its own registration and is deliberately not shipped or trained here.

What is left is a measured answer, and it is a good one: the ACS 2023
realised-couple distribution for people with your age and income -- median
partner age, partner income, age gap, same-ethnicity and same-education shares.
Those are couples who actually formed, not a model's opinion.

Why the dyad networks are **not** used here
-------------------------------------------
The dyad models need the partner's own self-ratings and stated importance.  The
ACS records neither.  An earlier version of this module imputed the population
medians, scored the candidates and reported a 52% mutual chance for a typical
partner -- against a 16.5% base rate and an 18.4% best cell in the app's own
map.  That is what out-of-distribution imputation looks like, so the scoring was
removed rather than dressed up.  "Who likes you most" stays where it belongs:
the dyad panel, scored on the package ladder the models were trained for.

The candidate grid is deterministic: the measured 25th, 50th and 75th
percentiles of partner age crossed with the same percentiles of partner income.
No sampling, so the same profile always gives the same answer.

    python partner.py --demo
    python partner.py --payload '{"gender":"man","age":18,"income":4500}'
"""

import argparse
import json

import numpy as np

import couples
from profile import DEFAULTS, have_data

QUANTILE = ((("p25"), "25th percentile"), ("median", "median"), ("p75", "75th percentile"))


def _quantiles(stat, low, high):
    """The measured p25/median/p75 of a distribution, clipped and de-duplicated."""
    if not stat:
        return []
    out = []
    for key, label in QUANTILE:
        value = stat.get(key)
        if value is None:
            continue
        value = float(min(max(float(value), low), high))
        if any(abs(value - seen[0]) < 1e-9 for seen in out):
            continue
        out.append((value, label))
    return out


def _median_self(gender):
    """Median self-ratings of real participants of that gender (measured)."""
    if not have_data():
        return None
    try:
        from train import FEATURE_SETS, load_dataset
        X, _ = load_dataset(gender=gender, features=FEATURE_SETS["market"])
    except (SystemExit, OSError):
        return None
    return [float(v) for v in np.nanmedian(X[:, 3:8], axis=0)]


def _ladder_level(selfs):
    """Where those self-ratings sit on the app's 3-to-10 package ladder."""
    if not selfs:
        return None
    mean = float(np.mean(selfs))
    return float(min(max(1.0 + 9.0 * (mean - 3.0) / 7.0, 1.0), 10.0))


def who_you_end_up_with(profile, gender, partner_gender):
    """The measured partner distribution for this profile."""
    context = couples.context(profile, gender, partner_gender)
    if not context:
        return {"available": False,
                "note": "couple priors are missing; run data/build_couples.py."}
    if not context.get("available"):
        return {"available": False, "same_gender": True,
                "note": context.get("note", "")}
    cell = context.get("cell") or context.get("overall")
    if not cell:
        return {"available": False, "note": "no ACS cell for this age and income."}

    ages = _quantiles(cell.get("partner_age"), 18.0, 80.0)
    incomes = _quantiles(cell.get("partner_income"), 0.0, 1_000_000.0)
    if not ages or not incomes:
        return {"available": False, "note": "the ACS cell has no partner quantiles."}

    your_age = float(profile.get("age", 24.0))
    candidates = []
    for age, age_label in ages:
        for income, income_label in incomes:
            candidates.append({
                "age": round(age, 1),
                "income": float(round(income / 1000.0) * 1000.0),
                "age_gap": round(age - your_age, 1),
                "age_quantile": age_label,
                "income_quantile": income_label,
                "most_typical": age_label == "median" and income_label == "median",
            })

    selfs = _median_self(partner_gender)
    typical = next((c for c in candidates if c["most_typical"]), candidates[len(candidates) // 2])

    def stat(source, key):
        return (source or {}).get(key)

    return {
        "available": True,
        "source": context.get("source"),
        "states": context.get("states"),
        "n_couples": context.get("n_total"),
        "cell": {
            "age_band": context.get("age_band"),
            "income_band": context.get("income_band"),
            "n": cell.get("n"),
            "match": context.get("match"),
            "education": context.get("education"),
            "same_education_pct": cell.get("same_edu_pct"),
        },
        "typical": {
            "age": stat(cell.get("partner_age"), "median"),
            "income": stat(cell.get("partner_income"), "median"),
            "age_gap": stat(cell.get("age_gap"), "median"),
            "income_ratio": stat(cell.get("income_ratio"), "median"),
            "same_ethnicity_pct": cell.get("same_eth_pct"),
            "same_education_pct": cell.get("same_edu_pct"),
            "top_partner_ethnicity": cell.get("top_partner_eth"),
            "top_partner_ethnicity_pct": cell.get("top_partner_eth_pct"),
        },
        "candidates": candidates,
        "most_typical": typical,
        "self_median": [round(v, 2) for v in selfs] if selfs else None,
        "self_basis": ("median self-ratings of real participants" if selfs
                       else "dataset missing; no self-ratings"),
        "ladder_level": _ladder_level(selfs),
        "stability_note": "research direction only: larger age gaps are associated with "
                          "more dissolution, while the education effect is weak or absent "
                          "in some samples; this does not predict your relationship.",
        "note": "ACS 2023 realised couples: who people with your age and income actually "
                "live with, not a forecast. The ACS records no partner self-ratings or "
                "stated importance, so these candidates are shown as measured profiles and "
                "are not scored by the dyad networks. No data shipped here follows a person "
                "to a lifetime partner, so 'lifetime' is not claimed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--payload", metavar="JSON")
    args = parser.parse_args()

    profile = dict(DEFAULTS)
    profile.update(mode="adult", age=18.0, income=4500.0)
    gender, partner = 1, 0
    if args.payload:
        raw = json.loads(args.payload)
        for key in ("age", "income", "goal", "height_cm", "age_gap"):
            if raw.get(key) is not None:
                profile[key] = float(raw[key])
        for key, width in (("self", 5), ("want", 6), ("importance", 6)):
            if isinstance(raw.get(key), list) and len(raw[key]) == width:
                profile[key] = [float(v) for v in raw[key]]
        if raw.get("education"):
            profile["education"] = raw["education"]
        gender = 1 if raw.get("gender") == "man" else 0
        partner = 0 if raw.get("partner_gender", "women") == "women" else 1

    out = who_you_end_up_with(profile, gender, partner)
    print(json.dumps(out, indent=1))
    if out.get("available"):
        t = out["typical"]
        print("\ntypical realised partner: age %.1f, income $%s, gap %s"
              % (t["age"], format(int(t["income"]), ",") if t["income"] else "n/a",
                 t["age_gap"]))
        if out["ladder_level"] is not None:
            print("self-ratings on the 3-10 package ladder: %.1f / 10" % out["ladder_level"])


if __name__ == "__main__":
    main()
