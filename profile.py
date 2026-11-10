"""Your chance, your hypergamy, and the market you are competing in.

Three modes, because the honest answer depends on how much you are willing to say:

  regular    ten ratings: how you rate yourself and the date you want.
  advanced   adds height, body type, race, age gap and income expectations --
             effects that are either measured in the data or drawn from a real
             population of 59,946 profiles.
  adult      adds orientation, kink and libido compatibility, experience, and
             partner build. **None of those have public outcome data**, so they run
             on a documented prior (`adult.py`) and are labelled as such.

The two trained networks answer the two halves separately:

  * their decision (`weights_market_*.json`, trained on `dec_o`) -> "a date says yes
    to you", from things you know about yourself;
  * your decision (`weights_deep_*.json`, trained on `dec` with the preferences you
    stated) -> "you say yes to them".

Multiplying them gives the chance of a mutual yes. Independence is an assumption,
but the data nearly agrees with it: 47.4% x 36.5% = 17.3% against an observed 16.5%.

  python profile.py --demo
  python profile.py --mode adult --gender 1 --age 24 --income 30000 \
      --self 7,7,7,7,5 --want 8,6,7,8,9,7 --importance 30,10,15,15,20,10
  python profile.py --demo --chart        # also writes hypergamy_scale.png
"""

import argparse
import json
import os

import numpy as np

import adult
from train import FEATURE_SETS, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "market": {0: "weights_market_women.json", 1: "weights_market_men.json"},
    "taste": {0: "weights_deep_women.json", 1: "weights_deep_men.json"},
}
FALLBACK = {
    "market": {0: "weights_women.json", 1: "weights_men.json"},
    "taste": {0: "weights_women.json", 1: "weights_men.json"},
}
LABEL = {0: "women", 1: "men"}
SINGULAR = {0: "woman", 1: "man"}

ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
PRETTY = ["looks", "sincerity", "intelligence", "fun", "ambition", "shared taste"]

BASE_RATE = {0: 0.365, 1: 0.474}
MATCH_RATE = 0.165

DEFAULTS = {
    "age": 24.0,
    "income": 32000.0,
    "goal": 4.0,
    "self": [7.0, 7.0, 7.0, 7.0, 5.0],
    "want": [8.0, 6.0, 7.0, 8.0, 9.0, 7.0],
    "importance": [30.0, 10.0, 15.0, 15.0, 20.0, 10.0],
    # advanced
    "height_in": 70.0,
    "body_type": "average",
    "same_race": True,
    "age_gap": 3.0,
    "partner_income": 50000.0,
    # adult
    "orientation_match": True,
    "kink_match": 3.0,
    "libido_match": 3.0,
    "genital": 3.0,
    "cup": 3.0,
    "experience": 3.0,
}


def model_path(kind, gender):
    for name in (MODELS[kind][gender], FALLBACK[kind][gender]):
        if os.path.exists(os.path.join(HERE, name)):
            return name
    raise SystemExit(f"no {kind} weights found -- run train.py first")


def load(path):
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        model = json.load(f)
    layers = model.get("layers")
    if layers is None:
        layers = [{"W": model["W1"], "b": model["b1"]},
                  {"W": model["W2"], "b": model["b2"]}]
    return model, [(np.asarray(l["W"], float), np.asarray(l["b"], float)) for l in layers]


def forward(model, layers, row):
    a = (np.asarray(row, float) - np.asarray(model["mu"], float)) / np.asarray(model["sigma"], float)
    for i, (W, b) in enumerate(layers):
        z = a @ W + b
        if i == len(layers) - 1:
            t = float(model.get("temperature", 1.0))
            return float(np.ravel(1.0 / (1.0 + np.exp(-z / t)))[0])
        a = np.tanh(z)
    raise ValueError("model has no layers")


def taste_row(profile):
    want = profile["want"]
    date = [profile["age"], profile["income"], profile["goal"]] + list(want) + [float(np.mean(want))]
    return date + list(profile["importance"])


def market_row(profile):
    return [profile["age"], profile["income"], profile["goal"]] + list(profile["self"])


def dataset_profile(gender, features):
    X, y = load_dataset(gender=gender, features=features)
    impute = np.nanmedian(X, axis=0)
    impute = np.where(np.isnan(impute), 0.0, impute)
    return np.where(np.isnan(X), impute, X), y.ravel()


_REVEALED = {}


def have_data():
    """A clone without the CSVs can still score a profile from the saved weights."""
    try:
        from train import data_path
        data_path()
        return True
    except SystemExit:
        return False


def revealed(profile, gender):
    if gender in _REVEALED:
        return _REVEALED[gender]
    model, layers = load(model_path("taste", gender))
    X, _ = dataset_profile(gender, FEATURE_SETS["taste"])

    def mean_at(feature, value):
        Z = X.copy()
        Z[:, FEATURE_SETS["taste"].index(feature)] = value
        return 100.0 * float(np.mean([forward(model, layers, row) for row in Z]))

    result = (mean_at("amb", 9.0) - mean_at("amb", 5.0),
              mean_at("attr", 9.0) - mean_at("attr", 5.0))
    _REVEALED[gender] = result
    return result


def stated_hypergamy(profile, gender):
    amb, attr = profile["importance"][4], profile["importance"][0]
    mine = amb / (amb + attr) if (amb + attr) > 0 else float("nan")
    X, _ = dataset_profile(gender, FEATURE_SETS["taste"])
    attr_pop, amb_pop = X[:, 10], X[:, 14]
    ok = (attr_pop + amb_pop) > 0
    ratios = amb_pop[ok] / (attr_pop[ok] + amb_pop[ok])
    return mine, float(np.mean(ratios < mine) * 100.0), float(np.nanmedian(ratios))


def competitors(profile, gender):
    """Rank the user against the opposite-gender rows who want the same partner."""
    market, layers = load(model_path("market", gender))
    Xm, _ = load_dataset(gender=1 - gender, features=FEATURE_SETS["market"])
    Xt, _ = load_dataset(gender=1 - gender, features=FEATURE_SETS["taste"])
    Xm = np.where(np.isnan(Xm), np.asarray(market["impute"], float), Xm)
    scores = np.array([
        forward(market, layers, [Xm[i, 0], Xm[i, 1], Xm[i, 2],
                                 Xm[i, 3], Xm[i, 4], Xm[i, 5], Xm[i, 6], Xm[i, 7]])
        for i in range(len(Xm))
    ])
    cols = [FEATURE_SETS["taste"].index(k) for k in ATTRS]
    mine = np.asarray(profile["want"], float)
    scale = np.array([30.0, 10.0, 15.0, 15.0, 20.0, 10.0])
    wanted = Xt[:, cols]
    wanted = np.where(np.isnan(wanted), np.nanmedian(wanted, axis=0), wanted)
    distance = np.sqrt(np.mean(((wanted - mine) / scale) ** 2, axis=1))
    close = distance <= np.percentile(distance, 25)
    my = forward(market, layers, market_row(profile)) * 100.0
    rank = 100.0 * float(np.mean(scores[close] < my / 100.0))
    return {
        "all": scores * 100.0,
        "close": scores[close] * 100.0,
        "rows": Xt[close],
        "rank": rank,
        "n": int(close.sum()),
        "my": my,
    }


def assess(profile, gender):
    market, market_layers = load(model_path("market", gender))
    taste, taste_layers = load(model_path("taste", gender))

    base_them = forward(market, market_layers, market_row(profile))
    base_you = forward(taste, taste_layers, taste_row(profile))
    d_them, d_you, rows = adult.adjustments(profile, gender)
    p_them = float(np.clip(base_them + d_them / 100.0, 0.01, 0.99))
    p_you = float(np.clip(base_you + d_you / 100.0, 0.01, 0.99))
    if have_data():
        _, pct, median_ratio = stated_hypergamy(profile, gender)
        d_amb, d_attr = revealed(profile, gender)
    else:
        pct = median_ratio = d_amb = d_attr = float("nan")

    return {
        "base_them": base_them,
        "base_you": base_you,
        "d_them": d_them,
        "d_you": d_you,
        "adjustments": rows,
        "p_them": p_them,
        "p_you": p_you,
        "mutual": p_them * p_you,
        "percentile": pct,
        "median_ratio": median_ratio,
        "d_amb": d_amb,
        "d_attr": d_attr,
        "market_acc": market.get("accuracy_cv", float("nan")),
        "taste_acc": taste.get("accuracy_cv", float("nan")),
        "market_path": model_path("market", gender),
        "taste_path": model_path("taste", gender),
    }


def bar(value, width=28):
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def report(profile, gender, name=""):
    r = assess(profile, gender)
    comp = competitors(profile, gender) if have_data() else None
    mode = profile.get("mode", "regular")
    print()
    print("=" * 66)
    print(f"  HYPERGAMY PROFILE{'  --  ' + name if name else ''}   [{mode}]")
    print("=" * 66)
    print(f"  you            {SINGULAR[gender]}, {profile['age']:.0f}, "
          f"${profile['income']:,.0f}/yr, dating goal {profile['goal']:.0f}")
    print(f"  you rate yourself  " + "  ".join(
        f"{p} {v:.0f}" for p, v in zip(["looks", "sinc", "intel", "fun", "amb"], profile["self"])))
    print(f"  you want           " + "  ".join(f"{p} {v:.0f}" for p, v in zip(PRETTY, profile["want"])))
    print(f"  stated importance  " + "  ".join(f"{p} {v:.0f}%" for p, v in zip(PRETTY, profile["importance"])))
    if mode in ("advanced", "adult"):
        print(f"  what you are       {profile.get('height_in', 0):.0f} in, "
              f"{profile.get('body_type', '?')}, age gap {profile.get('age_gap', 0):.0f} yrs, "
              f"{'same race' if profile.get('same_race') else 'different race'}")
    if mode == "adult":
        print(f"  adult layer        kink match {profile['kink_match']:.0f}/5, "
              f"libido {profile['libido_match']:.0f}/5, experience {profile['experience']:.0f}/5")
    print()
    print(f"  a date says yes to you    {r['p_them']:5.1%}  {bar(r['p_them'])}")
    print(f"  you say yes to them       {r['p_you']:5.1%}  {bar(r['p_you'])}")
    print(f"  both say yes              {r['mutual']:5.1%}  {bar(r['mutual'])}")
    print(f"    real match rate in the study: {MATCH_RATE:.1%}")
    print()
    if abs(r["d_them"]) > 0.05 or abs(r["d_you"]) > 0.05:
        print("  adjustments to those numbers:")
        for label, delta, basis in r["adjustments"]:
            if abs(delta) > 0.05:
                print(f"    {label:<42} {delta:>+5.1f} pp   [{basis}]")
        print()
    print(f"  HYPERGAMY  {r['percentile']:5.1f} / 100   percentile among {LABEL[gender]}"
          if r["percentile"] == r["percentile"] else "  HYPERGAMY  n/a (dataset missing)")
    print(f"    stated    ambition {profile['importance'][4]:.0f}% vs looks "
          f"{profile['importance'][0]:.0f}% of your importance")
    print(f"    revealed  +4 ambition -> {r['d_amb']:+.1f} pp   +4 looks -> {r['d_attr']:+.1f} pp")
    verdict = ("status moves you more than looks" if abs(r["d_amb"]) > abs(r["d_attr"])
               else "looks move you more than status")
    print(f"    verdict:  {verdict}")
    print()
    if comp is None:
        print("  THE MARKET   dataset not in this checkout -- run data/build.py for the")
        print("    competitor ranking and the population hypergamy percentile.")
    else:
        print(f"  THE MARKET   you score {comp['my']:.0f}/100 against the trained market model")
        print(f"    top {100 - comp['rank']:.0f}% of the {comp['n']:,} opposite-gender competitors "
              f"whose stated wants resemble your partner")
        you_tilt = 50.0 if profile["self"][4] + profile["self"][0] <= 0 else \
            100.0 * profile["self"][4] / (profile["self"][4] + profile["self"][0])
        partner_tilt = 50.0 if profile["want"][4] + profile["want"][0] <= 0 else \
            100.0 * profile["want"][4] / (profile["want"][4] + profile["want"][0])
        print(f"    hypergamy scale   you {you_tilt:.0f}/100   partner you want {partner_tilt:.0f}/100")
    print(f"    (models: their decision CV {r['market_acc']:.1%} via {r['market_path']}, "
          f"your decision CV {r['taste_acc']:.1%})")
    print("=" * 66)
    return r


def demo(chart=False):
    base = dict(DEFAULTS)
    report({**base, "mode": "regular", "age": 23.0, "income": 26000.0,
            "want": [9.0, 7.0, 6.0, 9.0, 6.0, 6.0],
            "importance": [40.0, 10.0, 5.0, 25.0, 10.0, 10.0]},
           0, "regular: the 23-year-old who wants a 10/10")
    report({**base, "mode": "advanced", "age": 27.0, "income": 58000.0, "height_in": 72.0,
            "body_type": "athletic", "age_gap": 4.0,
            "want": [7.0, 7.0, 7.0, 7.0, 8.0, 8.0],
            "importance": [25.0, 15.0, 15.0, 15.0, 20.0, 10.0]},
           1, "advanced: the 27-year-old who wants a peer")
    report({**base, "mode": "adult", "age": 25.0, "income": 42000.0, "height_in": 69.0,
            "body_type": "curvy", "age_gap": 2.0, "kink_match": 5.0, "libido_match": 4.0,
            "experience": 4.0, "same_race": False,
            "want": [8.0, 7.0, 8.0, 8.0, 7.0, 9.0],
            "importance": [30.0, 15.0, 15.0, 15.0, 15.0, 10.0]},
           0, "adult: the 25-year-old who knows what she wants")
    if chart:
        import visualize_scale
        visualize_scale.render({**base, "mode": "advanced", "age": 27.0, "income": 58000.0}, 1)


def numbers(text, n, default):
    if not text:
        return list(default)
    parts = [float(x) for x in text.replace(" ", "").split(",") if x]
    if len(parts) != n:
        raise SystemExit(f"expected {n} comma-separated numbers, got {len(parts)}")
    return parts


def main():
    p = argparse.ArgumentParser(description="your chance, your hypergamy, your market")
    p.add_argument("--mode", default="regular", choices=list(adult.MODES))
    p.add_argument("--gender", type=int, choices=[0, 1])
    p.add_argument("--age", type=float)
    p.add_argument("--income", type=float)
    p.add_argument("--goal", type=float)
    p.add_argument("--self", dest="self_")
    p.add_argument("--want")
    p.add_argument("--importance")
    p.add_argument("--height", type=float, help="inches")
    p.add_argument("--body", default="average", choices=adult.BODY_TYPES)
    p.add_argument("--age-gap", type=float, dest="age_gap")
    p.add_argument("--different-race", action="store_true")
    p.add_argument("--kink", type=float, help="kink compatibility 0-5")
    p.add_argument("--libido", type=float, help="libido compatibility 0-5")
    p.add_argument("--genital", type=float, help="self-reported 1-5")
    p.add_argument("--cup", type=float, help="partner build 1-5")
    p.add_argument("--experience", type=float, help="0-5")
    p.add_argument("--mismatch", action="store_true", help="orientation outside their pool")
    p.add_argument("--chart", action="store_true")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        demo(args.chart)
        return

    interactive = args.gender is None and os.isatty(0)
    gender = args.gender
    if interactive:
        raw = input("  your gender (0 = woman, 1 = man) [1]: ").strip()
        gender = int(raw) if raw else 1
    if gender is None:
        gender = 1

    profile = dict(DEFAULTS)
    profile["mode"] = args.mode
    for key in ("age", "income", "goal", "height", "age_gap", "kink", "libido",
                "genital", "cup", "experience"):
        if getattr(args, key) is not None:
            profile[key] = getattr(args, key)
    profile["body_type"] = args.body
    profile["same_race"] = not args.different_race
    profile["orientation_match"] = not args.mismatch
    profile["self"] = numbers(args.self_, 5, profile["self"])
    profile["want"] = numbers(args.want, 6, profile["want"])
    profile["importance"] = numbers(args.importance, 6, profile["importance"])

    report(profile, gender)
    if args.chart:
        import visualize_scale
        visualize_scale.render(profile, gender)


if __name__ == "__main__":
    main()
