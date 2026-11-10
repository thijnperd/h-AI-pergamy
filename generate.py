"""Generate the single person most likely to say yes to you.

This is a second, separate model from the dyad likes models. Those score a
candidate; this one goes the other way and *writes* a candidate from your
profile.

What the speed-dating data actually contains for the partner is: their age,
how they rated you on the six attributes, how much they liked you, and how
sure they were. It does **not** contain their income, body type, height, chest
size, kink or libido. So the generator is trained on the variables that exist,
and the remaining fields are filled from the documented population/research
priors in `adult.py` and marked as such in `basis`. Nothing is passed off as
trained when it is not.

The trained part is a multi-output MLP (14 inputs -> 16 -> 8 -> 9 outputs)
fitted on the rows where the partner said yes (`dec_o == 1`). At inference,
the predicted partner age is combined with the dyad model's 10 package
probabilities: the package is chosen by `log P(yes) - 0.5 * age_penalty`, so a
person who likes you but is wildly the wrong age is not chosen.

    python generate.py --train       # train weights_person_women/men.json
    python generate.py --demo        # a default profile, both genders
    python generate.py --payload '{"gender":"man",...}'
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

import adult
import hypergamy_map as hm
import likes
from profile import DEFAULTS
from train import PREF, SELF, data_path

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = ["age", "income", "goal"] + SELF + PREF
TARGETS = ["age_o", "attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o",
           "like_o", "prob_o"]
RATING_NAMES = ["looks", "sincerity", "intelligence", "fun", "ambition", "shared taste"]
OUT = {0: "weights_person_women.json", 1: "weights_person_men.json"}
PRIOR_GENDERS = {0: "women", 1: "men"}
LIMITS = np.array([[18.0, 80.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0],
                   [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0],
                   [0.0, 10.0]])


def _f(row, key):
    raw = (row.get(key) or "").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(gender):
    with open(data_path(), newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    X, Y = [], []
    for row in rows:
        if str(row.get("gender", "")).strip() != str(gender):
            continue
        if _f(row, "dec_o") != 1.0:
            continue
        ys = [_f(row, key) for key in TARGETS]
        if any(np.isnan(v) for v in ys):
            continue
        X.append([_f(row, key) for key in INPUTS])
        Y.append(ys)
    return np.asarray(X, float), np.asarray(Y, float)


class MLP:
    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, X):
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = np.tanh(z) if i < len(self.W) - 1 else z
        return a

    def loss(self, X, Y):
        return float(np.mean((self.forward(X) - Y) ** 2))

    def gradients(self, X, Y):
        a = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a[-1] @ W + b
            a.append(np.tanh(z) if i < len(self.W) - 1 else z)
        n = X.shape[0]
        delta = 2.0 * (a[-1] - Y) / (n * Y.shape[1])
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for layer in range(len(self.W) - 1, -1, -1):
            gW[layer] = a[layer].T @ delta
            gb[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.W[layer].T) * (1.0 - a[layer] ** 2)
        return gW, gb


def fit_model(X, Y, seed=7, hidden=(16, 8), epochs=600, lr=0.01, l2=1e-4):
    impute = np.nanmedian(X, axis=0)
    impute = np.where(np.isnan(impute), 0.0, impute)
    X = np.where(np.isnan(X), impute, X)
    mu_x, sigma_x = X.mean(axis=0), X.std(axis=0)
    sigma_x[sigma_x == 0] = 1.0
    mu_y, sigma_y = Y.mean(axis=0), Y.std(axis=0)
    sigma_y[sigma_y == 0] = 1.0
    Xs, Ys = (X - mu_x) / sigma_x, (Y - mu_y) / sigma_y

    net = MLP([X.shape[1]] + list(hidden) + [Y.shape[1]], np.random.default_rng(seed))
    m = [np.zeros_like(p) for p in net.W] + [np.zeros_like(p) for p in net.b]
    v = [np.zeros_like(p) for p in net.W] + [np.zeros_like(p) for p in net.b]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    params = net.W + net.b
    for epoch in range(1, epochs + 1):
        gW, gb = net.gradients(Xs, Ys)
        grads = [g + l2 * w for g, w in zip(gW, net.W)] + list(gb)
        for i, (param, grad) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * grad ** 2
            m_hat = m[i] / (1 - beta1 ** epoch)
            v_hat = v[i] / (1 - beta2 ** epoch)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return net, impute, mu_x, sigma_x, mu_y, sigma_y


def predict_raw(net, impute, mu_x, sigma_x, mu_y, sigma_y, x):
    x = np.asarray(x, float)
    x = np.where(np.isnan(x), impute, x)
    y = net.forward((x - mu_x) / sigma_x) * sigma_y + mu_y
    return y


def cross_validate(X, Y, folds=5, seed=7):
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(X))
    maes, bases = [], []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        net, impute, mu_x, sigma_x, mu_y, sigma_y = fit_model(X[train], Y[train], seed=seed)
        pred = np.array([predict_raw(net, impute, mu_x, sigma_x, mu_y, sigma_y, x)
                         for x in X[test]])
        maes.append(np.mean(np.abs(pred - Y[test]), axis=0))
        bases.append(np.mean(np.abs(Y[train].mean(axis=0) - Y[test]), axis=0))
    return np.mean(maes, axis=0), np.mean(bases, axis=0)


def train_one(gender):
    X, Y = load_rows(gender)
    if len(X) < 200:
        raise SystemExit(f"only {len(X)} yes-rows for gender {gender}")
    cv_mae, cv_base = cross_validate(X, Y)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    net, impute, mu_x, sigma_x, mu_y, sigma_y = fit_model(X[idx[:cut]], Y[idx[:cut]])
    pred = np.array([predict_raw(net, impute, mu_x, sigma_x, mu_y, sigma_y, x)
                     for x in X[idx[cut:]]])
    hold_mae = np.mean(np.abs(pred - Y[idx[cut:]]), axis=0)

    net, impute, mu_x, sigma_x, mu_y, sigma_y = fit_model(X, Y)
    model = {
        "inputs": INPUTS,
        "targets": TARGETS,
        "arch": [len(INPUTS), 16, 8, len(TARGETS)],
        "layers": [{"W": W.round(6).tolist(), "b": b.round(6).tolist()}
                   for W, b in zip(net.W, net.b)],
        "impute": impute.round(6).tolist(),
        "mu_x": mu_x.round(6).tolist(),
        "sigma_x": sigma_x.round(6).tolist(),
        "mu_y": mu_y.round(6).tolist(),
        "sigma_y": sigma_y.round(6).tolist(),
        "cv_mae": cv_mae.round(4).tolist(),
        "cv_baseline_mae": cv_base.round(4).tolist(),
        "holdout_mae": hold_mae.round(4).tolist(),
        "n": len(X),
        "epochs": 600,
        "lr": 0.01,
        "l2": 1e-4,
        "scope": {"gender": gender, "partner_gender": 1 - gender},
    }
    path = os.path.join(HERE, OUT[gender])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    print(f"{path}: n={len(X):,}  CV MAE {np.mean(cv_mae):.3f} "
          f"(baseline {np.mean(cv_base):.3f})")
    for name, mae, base in zip(TARGETS, cv_mae, cv_base):
        print(f"  {name:<8} {mae:5.3f}  baseline {base:5.3f}")


def load(gender):
    path = os.path.join(HERE, OUT[gender])
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    layers = [(np.asarray(l["W"], float), np.asarray(l["b"], float))
              for l in model["layers"]]
    return model, layers


def predict(profile, gender):
    model, layers = load(gender)
    x = ([profile["age"], profile["income"], profile["goal"]]
         + list(profile["self"]) + list(profile["importance"]))
    x = np.where(np.isnan(np.asarray(x, float)), np.asarray(model["impute"], float), x)
    a = (x - np.asarray(model["mu_x"], float)) / np.asarray(model["sigma_x"], float)
    for i, (W, b) in enumerate(layers):
        z = a @ W + b
        a = np.tanh(z) if i < len(layers) - 1 else z
    y = a * np.asarray(model["sigma_y"], float) + np.asarray(model["mu_y"], float)
    y = np.clip(y, LIMITS[:, 0], LIMITS[:, 1])
    return {name: float(v) for name, v in zip(TARGETS, y)}


def _body_prior(partner_gender):
    dist = adult.priors().get("body_type", {}).get(PRIOR_GENDERS[partner_gender], {})
    mapping = {"fit": "athletic", "skinny": "thin"}
    if not dist:
        return "average"
    best = max(dist.items(), key=lambda kv: kv[1])[0]
    return mapping.get(best, best)


def _height_prior(partner_gender):
    h = adult.priors().get("height_in", {}).get(PRIOR_GENDERS[partner_gender], {}).get("50")
    return round(float(h) * 2.54, 1) if h else adult.TYPICAL_HEIGHT[partner_gender]


def generate_person(profile, gender, partner_gender):
    """Return one complete partner profile, its chance, and the package range."""
    liked = likes.who_likes_you(profile, gender, partner_gender)
    gen = predict(profile, gender)
    packages = [hm.archetype(partner_gender, k) for k in range(1, likes.LEVELS + 1)]
    ages = np.array([p["age"] for p in packages], float)
    penalty = ((ages - gen["age_o"]) / 4.0) ** 2
    score = np.log(np.maximum(np.asarray(liked["their_yes"], float), 1e-9)) - 0.5 * penalty
    best = int(np.argmax(score)) + 1
    base = packages[best - 1]

    person = dict(base)
    person.update(
        gender="woman" if partner_gender == 0 else "man",
        age=round(float(np.clip(gen["age_o"], 18.0, 60.0)), 1),
        income=round(float(base["income"]), -2),
        goal=float(base["goal"]),
        body_type=_body_prior(partner_gender),
        height_cm=_height_prior(partner_gender),
        cup="C" if partner_gender == 0 else None,
        size_cm=13.1 if partner_gender == 1 else None,
        kink_match=3.0,
        libido_match=3.0,
        experience=3.0,
        package=best,
        age_gap=round(float(np.clip(gen["age_o"], 18.0, 60.0)) - float(profile["age"]), 1),
        rates_you={
            "looks": round(gen["attr_o"], 2),
            "sincerity": round(gen["sinc_o"], 2),
            "intelligence": round(gen["intel_o"], 2),
            "fun": round(gen["fun_o"], 2),
            "ambition": round(gen["amb_o"], 2),
            "shared taste": round(gen["shar_o"], 2),
            "interest": round(gen["like_o"], 2),
            "yes certainty": round(gen["prob_o"], 2),
        },
        basis={
            "age": "trained generator",
            "ratings of you": "trained generator",
            "income": "package template",
            "self ratings": "package template",
            "body type": "population prior",
            "height": "population prior",
            "chest/size": "population prior / research",
            "kink/libido": "research prior",
            "experience": "research prior",
        },
        same_gender=partner_gender == gender,
    )
    person["p_they"] = round(float(likes.p_yes(partner_gender, profile, person)), 4)
    person["p_you"] = round(float(likes.p_yes(gender, person, profile)), 4)
    person["mutual"] = round(person["p_they"] * person["p_you"], 4)
    person["range"] = {
        "low": liked["range_low"],
        "high": liked["range_high"],
        "packages": liked["range"],
        "low_chance": liked["range_low_chance"],
        "high_chance": liked["range_high_chance"],
    }
    person["score"] = round(float(score[best - 1]), 4)
    return person


def _profile_from_payload(raw):
    profile = dict(DEFAULTS)
    for key in ("age", "income", "goal"):
        if raw.get(key) is not None:
            profile[key] = float(raw[key])
    for key in ("self", "want", "importance"):
        if isinstance(raw.get(key), list) and len(raw[key]) == len(profile[key]):
            profile[key] = [float(v) for v in raw[key]]
    gender = 1 if raw.get("gender") == "man" else 0
    partner = 0 if raw.get("partner_gender", "women") == "women" else 1
    return profile, gender, partner


def _print_person(person):
    print(f"\n{person['gender']} {person['age']:.1f}  package {person['package']}")
    print(f"  income ${person['income']:,.0f}  goal {person['goal']:.0f}  "
          f"build {person['body_type']}  height {person['height_cm']:.0f} cm")
    if person["cup"]:
        print(f"  chest {person['cup']}  kink {person['kink_match']:.0f}/5  "
              f"libido {person['libido_match']:.0f}/5  experience {person['experience']:.0f}/5")
    else:
        print(f"  size {person['size_cm']:.1f} cm  kink {person['kink_match']:.0f}/5  "
              f"libido {person['libido_match']:.0f}/5  experience {person['experience']:.0f}/5")
    print("  rates you: " + ", ".join(
        f"{k} {v:.1f}" for k, v in person["rates_you"].items()))
    print(f"  they say yes {person['p_they']:.0%}  you say yes {person['p_you']:.0%}"
          f"  mutual {person['mutual']:.0%}")
    print(f"  packages most likely to say yes: {person['range']['low']}-"
          f"{person['range']['high']} "
          f"({person['range']['low_chance']:.0%}-{person['range']['high_chance']:.0%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", action="store_true", help="train both generator models")
    parser.add_argument("--demo", action="store_true", help="a default profile")
    parser.add_argument("--payload", metavar="JSON", help="the GUI payload")
    args = parser.parse_args()

    if args.train:
        for gender in (0, 1):
            train_one(gender)
        return
    if args.payload:
        import json as _json
        profile, gender, partner = _profile_from_payload(_json.loads(args.payload))
        _print_person(generate_person(profile, gender, partner))
        return

    profile = dict(DEFAULTS)
    profile.update(mode="regular", age=24.0, income=32000.0)
    _print_person(generate_person(profile, 1, 0))
    _print_person(generate_person(profile, 0, 1))


if __name__ == "__main__":
    sys.exit(main())
