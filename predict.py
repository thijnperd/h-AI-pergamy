"""Score one date through the trained network and show which neurons did it.

    python predict.py                 # asks for the ten inputs, Enter = default
    python predict.py attr=10 fun=9   # override any input on the command line
    python predict.py --demo          # three canned dates, no prompts

The logit is split exactly: b2 + sum(a_j * W2_j). Each row below is one hidden
neuron's real contribution, so "why did it say yes" has an answer.
"""

import sys

import numpy as np

from train import FEATURES, load_dataset
from visualize import SHORT

DEFAULTS = {
    "age": 27.0,
    "income": 50000.0,
    "goal": 2.0,
    "attr": 7.0,
    "sinc": 7.0,
    "intel": 7.0,
    "fun": 7.0,
    "amb": 7.0,
    "shar": 7.0,
    "like": 7.0,
}

PROMPTS = {
    "age": "age (years)",
    "income": "income ($/yr)",
    "goal": "wants kids? 1-6",
    "attr": "attractive 1-10",
    "sinc": "sincere 1-10",
    "intel": "intelligent 1-10",
    "fun": "fun 1-10",
    "amb": "ambitious 1-10",
    "shar": "shared taste 1-10",
    "like": "overall like 1-10",
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model(path="weights.json"):
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, path), encoding="utf-8") as f:
        m = json.load(f)
    if "mu" not in m or "sigma" not in m:
        raise SystemExit("weights.json has no mu/sigma -- re-run train.py first.")
    return (
        np.asarray(m["W1"], dtype=float),
        np.asarray(m["b1"], dtype=float),
        np.asarray(m["W2"], dtype=float).ravel(),
        float(np.asarray(m["b2"], dtype=float).ravel()[0]),
        np.asarray(m["mu"], dtype=float),
        np.asarray(m["sigma"], dtype=float),
    )


def parse_overrides(argv):
    values = {}
    for arg in argv:
        if "=" in arg:
            key, raw = arg.split("=", 1)
            key = key.strip()
            if key not in FEATURES:
                raise SystemExit(f"unknown input '{key}' -- choose from {', '.join(FEATURES)}")
            values[key] = float(raw)
    return values


def ask(values, interactive):
    out = []
    for name in FEATURES:
        default = values.get(name, DEFAULTS[name])
        if interactive:
            raw = input(f"  {PROMPTS[name]:<18}[{default:g}]: ").strip()
            try:
                out.append(float(raw) if raw else default)
            except ValueError:
                print(f"    not a number, using {default:g}")
                out.append(default)
        else:
            out.append(default)
    return np.asarray(out, dtype=float)


def explain(raw, W1, b1, W2, b2, mu, sigma):
    x = (raw - mu) / sigma
    a = np.tanh(x @ W1 + b1)
    logit = float(a @ W2 + b2)
    p = float(sigmoid(logit))
    contrib = a * W2

    # Same numbering as visualize.py: hidden units sorted by incoming weight mass.
    order = np.argsort(-np.abs(W1).sum(axis=0))
    label = {int(j): k + 1 for k, j in enumerate(order)}

    print(f"\n  P(second date) = {p:.1%}    ->  {'YES' if p > 0.5 else 'no'}")
    print(f"  logit {logit:+.3f}  =  bias {b2:+.3f}  +  neurons {logit - b2:+.3f}")
    print(f"\n  {'neuron':>7}  {'activation':>10}  {'readout':>8}  {'contribution':>12}  crowd")
    for j in np.argsort(-np.abs(contrib)):
        if contrib[j] >= 0.02:
            crowd = "push yes"
        elif contrib[j] <= -0.02:
            crowd = "push no"
        else:
            crowd = "indifferent"
        print(
            f"  {'#' + str(label[int(j)]):>7}  {a[j]:>+10.2f}  {W2[j]:>+8.2f}  "
            f"{contrib[j]:>+11.3f}  {crowd}"
        )
    top = label[int(np.argmax(contrib))]
    bottom = label[int(np.argmin(contrib))]
    print(f"\n  strongest push toward yes: #{top}"
          f"   strongest push toward no: #{bottom}")


def main():
    W1, b1, W2, b2, mu, sigma = load_model()

    if "--demo" in sys.argv or not sys.stdin.isatty():
        X, _ = load_dataset()
        dream = np.array([10.0] * len(FEATURES))
        dream[FEATURES.index("income")] = 69487.0
        mean = X.mean(axis=0)
        print("\n=== a date rated 10/10 on everything ===")
        explain(dream, W1, b1, W2, b2, mu, sigma)
        print("\n=== the average participant in the dataset ===")
        explain(mean, W1, b1, W2, b2, mu, sigma)
        print("\n=== attractive and fun, nothing else ===")
        plain = mean.copy()
        plain[FEATURES.index("attr")] = 10.0
        plain[FEATURES.index("fun")] = 10.0
        plain[FEATURES.index("like")] = 8.0
        explain(plain, W1, b1, W2, b2, mu, sigma)
        print()
        return

    print("rate the date (Enter keeps the default).")
    raw = ask(parse_overrides(sys.argv[1:]), interactive=True)
    explain(raw, W1, b1, W2, b2, mu, sigma)
    print()


if __name__ == "__main__":
    main()
