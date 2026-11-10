"""Pick the right trained model for a profile.

The short version:

  age 18-25 (or the 18-25 mode) -> the gender model *and* the model trained only on
                                   18-25 year olds, averaged
  any other age                  -> the gender model only
  the other person's decision    -> always the gender market model (there is no
                                   age-specific market model, and pretending there
                                   is would be a lie)

Averaging two probabilities is a blend, not a new network: both models see the same
16 inputs, so there is nothing to concatenate. The GUI shows which files were used
and their cross-validated accuracy so the blend is never hidden.

  python route.py            # a small table of what would be used for which age
"""

import os

import numpy as np

from profile import MODELS, forward, load, market_row, taste_row

HERE = os.path.dirname(os.path.abspath(__file__))
YOUNG = "weights_young.json"
YOUNG_RANGE = (18, 25)
YOUNG_MODE = "18-25"


def _exists(name):
    return os.path.exists(os.path.join(HERE, name))


def uses_young(age, mode):
    """True when the 18-25 model should join the average."""
    if not _exists(YOUNG):
        return False
    if mode == YOUNG_MODE:
        return True
    return YOUNG_RANGE[0] <= float(age) <= YOUNG_RANGE[1]


def taste_models(age, mode, gender):
    paths = [MODELS["taste"][gender]]
    if uses_young(age, mode):
        paths.append(YOUNG)
    return paths


def market_models(age, mode, gender):
    # No age-specific market model exists; the gender model is the honest choice.
    return [MODELS["market"][gender]]


def _average(paths, row):
    probabilities, used = [], []
    for path in paths:
        model, layers = load(path)
        probabilities.append(forward(model, layers, row))
        used.append({"path": path, "cv": model.get("accuracy_cv"),
                     "arch": model.get("arch")})
    return float(np.mean(probabilities)), used


def predict_you(profile, gender):
    age = float(profile["age"])
    mode = profile.get("mode", "regular")
    return _average(taste_models(age, mode, gender), taste_row(profile))


def predict_them(profile, gender):
    age = float(profile["age"])
    mode = profile.get("mode", "regular")
    return _average(market_models(age, mode, gender), market_row(profile))


def explain(age, mode, gender):
    taste = taste_models(age, mode, gender)
    market = market_models(age, mode, gender)
    if len(taste) == 2:
        why = f"age {age:.0f} is inside 18-25, so the 18-25 model is blended in"
    else:
        why = f"age {age:.0f} is outside 18-25, so the full-sample model is used"
    return {"taste": taste, "market": market, "reason": why}


def main():
    print(f"{'age':>4}  {'mode':<8}  {'gender':<6}  models")
    for age in (17, 18, 22, 25, 26, 45):
        for mode in ("regular", "18-25"):
            for gender in (0, 1):
                plan = explain(age, mode, gender)
                label = "woman" if gender == 0 else "man"
                print(f"{age:>4}  {mode:<8}  {label:<6}  "
                      f"taste: {' + '.join(plan['taste'])}   market: {plan['market'][0]}")
    print("\n" + explain(18, "18-25", 1)["reason"])


if __name__ == "__main__":
    main()
