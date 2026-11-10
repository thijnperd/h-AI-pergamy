"""What does a point of looks, ambition or income actually buy?

Each panel sweeps one input through a trained network while every other input
stays at its real value for that row, then averages the prediction over all
8,378 dates. The slope is therefore a measured partial effect of the model --
not a correlation read off the table, and not a hand-drawn stereotype.

The third panel also checks the 18-25 model: if money "matters less" when you
are young, the young model's income curve should be flatter than the full
model's on the same rows. The script prints that comparison.

  python visualize_chance.py     # -> hypergamy_chance.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from profile import forward, load
from train import FEATURE_SETS, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hypergamy_chance.png")

MODELS = {0: "weights_deep_women.json", 1: "weights_deep_men.json"}
COLOR = {0: "#c2185b", 1: "#1b9ad6"}
NAME = {0: "WOMEN", 1: "MEN"}

FEATS = FEATURE_SETS["taste"]
INK = "#111111"
GREY = "#6b6b6b"
RULE = "#dcdcdc"

PANELS = [
    ("attr", "HOW ATTRACTIVE THE DATE IS", list(range(1, 11)), "rating (1-10)"),
    ("amb", "HOW AMBITIOUS THE DATE IS", list(range(1, 11)), "rating (1-10)"),
    ("income", "HOW MUCH THE DATE EARNS", [12000, 25000, 40000, 60000, 90000, 140000], "income ($/yr)"),
]


def rows_for(gender=None, model_path=None, min_age=None, max_age=None):
    model, layers = load(model_path) if model_path else (None, None)
    X, _ = load_dataset(gender=gender, features=FEATS, min_age=min_age, max_age=max_age)
    impute = np.asarray(model["impute"], float) if model else None
    if impute is not None:
        X = np.where(np.isnan(X), impute, X)
    return X


def curve(model_path, X, feature, values):
    model, layers = load(model_path)
    column = FEATS.index(feature)
    out = []
    for value in values:
        Z = X.copy()
        Z[:, column] = value
        out.append(np.mean([forward(model, layers, row) for row in Z]))
    return np.asarray(out) * 100.0


def main():
    fig = plt.figure(figsize=(14.0, 6.6), dpi=200, facecolor="white")
    fig.text(0.045, 0.955, "HYPERGAMY  ·  WHAT MOVES A YES", fontsize=7.5, color=GREY)
    fig.text(0.045, 0.905, "LOOKS, AMBITION OR MONEY?", fontsize=25, fontweight="black", color=INK)
    fig.text(0.045, 0.858,
             "each curve sweeps one input through a trained network over all 8,378 real dates, "
             "holding everything else fixed",
             fontsize=9, color=GREY)

    summary = {}
    for k, (feature, title, values, xlabel) in enumerate(PANELS):
        ax = fig.add_axes([0.045 + k * 0.322, 0.20, 0.27, 0.58])
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=INK, loc="left", pad=10)
        for gender in (0, 1):
            X = rows_for(gender, MODELS[gender])
            y = curve(MODELS[gender], X, feature, values)
            summary[(feature, gender)] = (y[0], y[-1])
            ax.plot(values, y, color=COLOR[gender], lw=2.6, marker="o", ms=4.2,
                    label=f"{NAME[gender]}  {y[-1] - y[0]:+.0f} pp")
            ax.fill_between(values, y, max(0, y.min() - 5), color=COLOR[gender], alpha=0.05)
        ax.set_xlabel(f"the date's {xlabel}", fontsize=8, color=GREY)
        if k == 0:
            ax.set_ylabel("P(you say yes)", fontsize=8, color=GREY)
        ax.grid(axis="y", color=RULE, lw=0.7)
        ax.tick_params(labelsize=8, colors=GREY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
        ax.legend(frameon=False, fontsize=7.6, loc="upper left")
        if feature == "income":
            ax.set_xticks(values)
            ax.set_xticklabels([f"{v // 1000}k" for v in values])

    fig.text(0.045, 0.075,
             "read it like this: the steeper the curve, the more that input moves the decision. "
             "These are model partial effects from a\n2002-2004 speed-dating sample, not a claim "
             "about what anyone should want.",
             fontsize=7.8, color=GREY, linespacing=1.5)
    fig.text(0.045, 0.028,
             "models: 16 -> 24 -> 12 -> 1, 5-fold CV 80.3% (women) / 81.1% (men)  ·  "
             "temperatures calibrated on held-out data  ·  dataset: 8,378 four-minute dates",
             fontsize=7.2, color="#9a9a9a")

    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"wrote {OUT}")
    for (feature, gender), (lo, hi) in summary.items():
        print(f"  {NAME[gender]:6s} {feature:6s}: {lo:.1f}% -> {hi:.1f}%  slope {hi - lo:+.2f} pp")

    # Does money matter less at 18-25? Same rows, two models.
    young_rows = rows_for(model_path="weights_young.json", min_age=18, max_age=25)
    income_values = PANELS[2][2]
    full_all = curve("weights_deep.json", rows_for(model_path="weights_deep.json"), "income", income_values)
    young_model = curve("weights_young.json", young_rows, "income", income_values)
    print(f"\nincome 12k -> 140k, on the 18-25 rows only:")
    print(f"  full-sample model  {full_all[0]:.1f}% -> {full_all[-1]:.1f}%  slope {full_all[-1] - full_all[0]:+.2f} pp")
    print(f"  18-25 model        {young_model[0]:.1f}% -> {young_model[-1]:.1f}%  slope {young_model[-1] - young_model[0]:+.2f} pp")


if __name__ == "__main__":
    main()
