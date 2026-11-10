"""Where you sit on the hypergamy scale, where your preferred partner sits, and what
you are competing against.

Panel 1 is the scale itself: a status-to-looks ratio built from self-ratings, with the
study population behind it, your position, and the position of the partner you said
you want. Panel 2 is the market: every opposite-gender participant is scored by the
trained market network, restricted to the ones whose stated wants look like the
partner you want, and your own score is placed among them.

Both panels read the real dataset and the trained models. Nothing here is decorative.

  python visualize_scale.py                 # a demo profile
  python visualize_scale.py --gender 0 --age 24 --income 30000 \
      --self 7,7,7,7,5 --want 8,6,7,8,9,7 --importance 30,10,15,15,20,10
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from profile import MODELS, assess, forward, load, market_row, taste_row
from train import FEATURE_SETS, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hypergamy_scale.png")

PINK = "#c2185b"
BLUE = "#1b9ad6"
INK = "#111111"
GREY = "#6b6b6b"
RULE = "#dcdcdc"

DEFAULT = {
    "age": 24.0, "income": 32000.0, "goal": 4.0,
    "self": [7.0, 7.0, 7.0, 7.0, 5.0],
    "want": [8.0, 6.0, 7.0, 8.0, 9.0, 7.0],
    "importance": [30.0, 10.0, 15.0, 15.0, 20.0, 10.0],
    "partner_income": 50000.0,
}


def tilt(ambition, looks):
    total = ambition + looks
    return 50.0 if total <= 0 else 100.0 * ambition / total


def population_tilt(gender):
    X, _ = load_dataset(gender=gender, features=FEATURE_SETS["taste"])
    amb, attr = X[:, 14], X[:, 10]
    ok = (amb + attr) > 0
    return 100.0 * amb[ok] / (amb[ok] + attr[ok])


def competitor_scores(profile, gender):
    """Every opposite-gender row, ranked by how likely they would be said yes to,
    restricted to the ones whose stated wants resemble the partner you want."""
    market, layers = load(MODELS["market"][gender])
    Xm, _ = load_dataset(gender=1 - gender, features=FEATURE_SETS["market"])
    Xt, _ = load_dataset(gender=1 - gender, features=FEATURE_SETS["taste"])
    Xm = np.where(np.isnan(Xm), np.asarray(market["impute"], float), Xm)

    scores = np.array([forward(market, layers, [
        Xm[i, 0], Xm[i, 1], Xm[i, 2], Xm[i, 3], Xm[i, 4], Xm[i, 5], Xm[i, 6], Xm[i, 7]
    ]) for i in range(len(Xm))])

    # "Also targeting your partner": their stated wants sit close to yours.
    want_columns = [FEATURE_SETS["taste"].index(k) for k in ("attr", "sinc", "intel", "fun", "amb", "shar")]
    mine = np.asarray(profile["want"], float)
    scale = np.array([30.0, 10.0, 15.0, 15.0, 20.0, 10.0])          # percentage points
    wanted = Xt[:, want_columns]
    wanted = np.where(np.isnan(wanted), np.nanmedian(wanted, axis=0), wanted)
    distance = np.sqrt(np.mean(((wanted - mine) / scale) ** 2, axis=1))
    close = distance <= np.percentile(distance, 25)                  # the nearest quartile

    return scores, scores[close], Xt[close], distance


def render(profile, gender, path=OUT):
    r = assess(profile, gender)
    pop = population_tilt(gender)
    you = tilt(profile["self"][4], profile["self"][0])
    partner = tilt(profile["want"][4] * 10.0, profile["want"][0] * 10.0)
    scores, close, rows, _ = competitor_scores(profile, gender)
    my_score = r["p_them"] * 100.0
    rank = 100.0 * float(np.mean(scores < my_score / 100.0))

    fig = plt.figure(figsize=(13.0, 7.2), dpi=200, facecolor="white")
    fig.text(0.045, 0.955, "HYPERGAMY  ·  THE SCALE AND THE MARKET", fontsize=7.5, color=GREY)
    fig.text(0.045, 0.905, "WHERE YOU SIT, AND WHO ELSE WANTS THEM", fontsize=24,
             fontweight="black", color=INK)

    # ---- panel 1: the scale -------------------------------------------------
    ax = fig.add_axes([0.055, 0.17, 0.42, 0.60])
    ax.hist(pop, bins=np.arange(0, 105, 5), color="#e6e6e6", edgecolor="white", lw=0.6)
    for value, label, color, dy in ((you, "YOU", PINK, 0.92), (partner, "THE PARTNER\nYOU WANT", BLUE, 0.62)):
        ax.axvline(value, color=color, lw=2.2, ls="--")
        ax.text(value, ax.get_ylim()[1] * dy, f" {label}\n {value:.0f}/100",
                color=color, fontsize=8.6, fontweight="bold", ha="left", va="top")
    ax.axvline(float(np.median(pop)), color=GREY, lw=1.0)
    ax.text(np.median(pop), ax.get_ylim()[1] * 0.34, f"median {np.median(pop):.0f}",
            color=GREY, fontsize=7.4, ha="center")
    ax.set_xlim(0, 100)
    ax.set_xlabel("status tilt:  0 = all looks, 100 = all status/ambition", fontsize=8, color=GREY)
    ax.set_ylabel(f"{'men' if gender else 'women'} in the study", fontsize=8, color=GREY)
    ax.set_title("THE HYPERGAMY SCALE", fontsize=10, fontweight="bold", loc="left", pad=10)
    ax.tick_params(labelsize=8, colors=GREY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ---- panel 2: the market ------------------------------------------------
    ax2 = fig.add_axes([0.555, 0.17, 0.40, 0.60])
    ax2.hist(close * 100.0, bins=np.arange(0, 105, 4), color="#f3d6e2", edgecolor="white", lw=0.6)
    ax2.axvline(my_score, color=INK, lw=2.0)
    ax2.text(my_score, ax2.get_ylim()[1] * 0.94,
             f"  you {my_score:.0f}%\n  top {100 - rank:.0f}% of {len(close):,}",
             fontsize=8.6, fontweight="bold", ha="left", va="top")
    ax2.set_xlabel("P(a date says yes), scored for every competitor", fontsize=8, color=GREY)
    ax2.set_ylabel("competitors with similar taste", fontsize=8, color=GREY)
    ax2.set_title("WHO ELSE IS TARGETING YOUR PARTNER", fontsize=10, fontweight="bold", loc="left", pad=10)
    ax2.tick_params(labelsize=8, colors=GREY)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.text(0.055, 0.085,
             f"your chance of a mutual yes {r['mutual']:.1%}   ·   their chance of yes {r['p_them']:.1%}   ·   "
             f"your chance of yes {r['p_you']:.1%}   ·   hypergamy {r['percentile']:.0f}/100",
             fontsize=9, color=INK)
    fig.text(0.055, 0.045,
             "left: self-rated ambition against looks, the same ratio for the population and for the partner you asked for.  "
             "right: every opposite-gender\ncompetitor scored by the trained market model, restricted to the quarter whose stated "
             "wants are closest to yours.",
             fontsize=7.6, color=GREY, linespacing=1.5)

    fig.savefig(path, dpi=200, facecolor="white")
    print(f"wrote {path}")
    print(f"  you {you:.0f}/100, partner {partner:.0f}/100, population median {np.median(pop):.0f}")
    print(f"  market score {my_score:.1f}%, top {100 - rank:.0f}% of {len(close):,} similar competitors")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gender", type=int, default=1, choices=[0, 1])
    p.add_argument("--age", type=float, default=DEFAULT["age"])
    p.add_argument("--income", type=float, default=DEFAULT["income"])
    p.add_argument("--self", dest="self_", default=None)
    p.add_argument("--want", default=None)
    p.add_argument("--importance", default=None)
    a = p.parse_args()
    profile = dict(DEFAULT)
    profile["age"], profile["income"] = a.age, a.income
    if a.self_:
        profile["self"] = [float(x) for x in a.self_.split(",")]
    if a.want:
        profile["want"] = [float(x) for x in a.want.split(",")]
    if a.importance:
        profile["importance"] = [float(x) for x in a.importance.split(",")]
    render(profile, a.gender)


if __name__ == "__main__":
    main()
