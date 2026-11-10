"""The fruit-fly test: switch one hidden neuron on, watch the network change its mind.

In 2024 the FlyWire consortium published the complete connectome of an adult
Drosophila -- every neuron and every synapse. A wiring diagram is a hypothesis
about what a neuron does; the experiment is to drive that neuron and watch the
animal. This script runs the same protocol on the speed-dating network, which
fits on one screen and therefore can actually be tested exhaustively.

Protocol, once per hidden neuron j, over all 3,581 dates:
  * stimulate -- force a_j = +1 (saturated firing), recompute every prediction
  * silence   -- force a_j = -1 (fully suppressed), recompute
  * trigger   -- quartile contrast: which input best separates the dates on which
                 this neuron is active?
  * flips     -- how many yes/no decisions the intervention actually reverses

Every number is measured on the trained weights. Nothing is drawn by hand.
Neuron numbering matches visualize.py: hidden units are sorted by incoming weight
mass, so #1 is the most-wanted neuron in the picture.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

from train import FEATURES, load_dataset
from visualize import SHORT

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights.json")
OUT = os.path.join(HERE, "fruitfly_neurons.png")

POS = "#a3144a"
NEG = "#12557f"
INK = "#111111"
GREY = "#6b6b6b"
BLUE = "#1b9ad6"
THORAX = "#12688f"
WING = "#cfe3f2"
RULE = "#dcdcdc"
DISH = "#fafafa"

FIGSIZE = (7.5, 10.0)
DPI = 200

COLS = (2.00, 5.55)
ROWS_Y = (8.02, 6.73, 5.44, 4.15, 2.86)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model():
    with open(WEIGHTS, encoding="utf-8") as f:
        m = json.load(f)
    if "mu" not in m or "sigma" not in m:
        raise SystemExit("weights.json has no mu/sigma -- re-run train.py first.")
    W1 = np.asarray(m["W1"], dtype=float)
    b1 = np.asarray(m["b1"], dtype=float)
    W2 = np.asarray(m["W2"], dtype=float).ravel()
    b2 = float(np.asarray(m["b2"], dtype=float).ravel()[0])
    mu = np.asarray(m["mu"], dtype=float)
    sigma = np.asarray(m["sigma"], dtype=float)
    return m, W1, b1, W2, b2, mu, sigma


def forward(X, W1, b1, W2, b2):
    A = np.tanh(X @ W1 + b1)
    p = sigmoid(A @ W2 + b2)
    return A, p.ravel()


def clamp(A, W2, b2, j, value):
    A2 = A.copy()
    A2[:, j] = value
    return sigmoid(A2 @ W2 + b2).ravel()


def quartile_trigger(activation, Xraw):
    """Which input most separates the dates where this neuron is switched on?

    High group = top quartile of that feature, low group = bottom quartile, and we
    report the standardised activation gap (Cohen's d). This is empirical: it never
    looks at the weights, only at what the neuron actually did.
    """
    hi_q = np.percentile(Xraw, 75, axis=0)
    lo_q = np.percentile(Xraw, 25, axis=0)
    ds = np.zeros(Xraw.shape[1])
    for f in range(Xraw.shape[1]):
        hi = activation[Xraw[:, f] >= hi_q[f]]
        lo = activation[Xraw[:, f] <= lo_q[f]]
        if len(hi) < 2 or len(lo) < 2:
            continue
        pooled = np.sqrt((hi.var() + lo.var()) / 2.0)
        if pooled > 1e-12:
            ds[f] = (hi.mean() - lo.mean()) / pooled
    k = int(np.argmax(np.abs(ds)))
    return k, float(ds[k])


def fly(ax, cx, cy, s):
    """A small Drosophila, drawn from ellipses and lines."""
    for sgn in (-1, 1):
        for y0, y1, x1 in ((-0.22, -0.44, 0.34), (-0.04, -0.16, 0.40), (0.12, 0.26, 0.34)):
            ax.plot(
                [cx + sgn * 0.14 * s, cx + sgn * x1 * s],
                [cy + y0 * s, cy + y1 * s],
                color=INK,
                lw=0.55,
                solid_capstyle="round",
                zorder=4,
            )
    for sgn in (-1, 1):
        ax.add_patch(
            Ellipse(
                (cx + sgn * 0.30 * s, cy + 0.27 * s),
                width=0.62 * s,
                height=0.24 * s,
                angle=sgn * 32,
                facecolor=WING,
                edgecolor="#9fb6c8",
                lw=0.4,
                alpha=0.85,
                zorder=2,
            )
        )
    ax.add_patch(Ellipse((cx, cy - 0.16 * s), 0.46 * s, 0.62 * s, facecolor=BLUE, edgecolor="none", zorder=3))
    for dy in (-0.06, -0.18, -0.30):
        ax.add_patch(
            Ellipse(
                (cx, cy + dy * s),
                0.40 * s,
                0.055 * s,
                facecolor=THORAX,
                edgecolor="none",
                alpha=0.55,
                zorder=4,
            )
        )
    ax.add_patch(Ellipse((cx, cy + 0.19 * s), 0.48 * s, 0.34 * s, facecolor=THORAX, edgecolor="none", zorder=3))
    ax.add_patch(Circle((cx, cy + 0.41 * s), 0.15 * s, facecolor=THORAX, edgecolor="none", zorder=3))
    for sgn in (-1, 1):
        ax.add_patch(
            Circle((cx + sgn * 0.085 * s, cy + 0.43 * s), 0.055 * s, facecolor="#8e1b1b", edgecolor="none", zorder=5)
        )


def signed(value):
    return f"{'+' if value >= 0 else '-'}{abs(value):.1f}"


def render(rows, meta, path):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, FIGSIZE[0])
    ax.set_ylim(0, FIGSIZE[1])
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.70,
        9.78,
        "HYPERGAMY  ·  EXPERIMENT 02",
        ha="left",
        va="center",
        fontsize=7.0,
        color=GREY,
        family="sans-serif",
    )
    ax.text(
        FIGSIZE[0] / 2,
        9.50,
        "THE FRUIT-FLY TEST",
        ha="center",
        va="center",
        fontsize=31,
        fontweight="black",
        color=INK,
        family="sans-serif",
    )
    ax.text(
        FIGSIZE[0] / 2,
        9.16,
        "clamp one hidden neuron  ·  watch the network change its mind",
        ha="center",
        va="center",
        fontsize=9.0,
        color=GREY,
        family="sans-serif",
    )
    ax.plot([0.70, 6.80], [8.94, 8.94], color=RULE, lw=0.9, zorder=1)

    max_abs = max(max(abs(r["d_on"]), abs(r["d_off"])) for r in rows) or 1.0
    bar_max = 1.08

    for i, r in enumerate(rows):
        cx = COLS[i // 5]
        cy = ROWS_Y[i % 5]
        winner = r["label"] == meta["winner"]

        dish_cx, dish_cy, radius = cx - 1.12, cy + 0.10, 0.42
        ax.add_patch(
            Circle(
                (dish_cx, dish_cy),
                radius,
                facecolor=DISH,
                edgecolor=POS if winner else RULE,
                lw=2.0 if winner else 1.0,
                zorder=1,
            )
        )
        fly(ax, dish_cx, dish_cy - 0.02, 0.70)

        tx = cx - 0.56
        ax.text(
            tx,
            cy + 0.50,
            f"NEURON #{r['label']}",
            ha="left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=POS if winner else INK,
            family="sans-serif",
        )
        ax.text(
            tx,
            cy + 0.32,
            f"fires on {r['fires']:.0%} of dates",
            ha="left",
            va="center",
            fontsize=6.4,
            color=GREY,
            family="sans-serif",
        )
        arrow = "\u2191" if r["d_cohen"] > 0 else "\u2193"
        ax.text(
            tx,
            cy + 0.14,
            f"triggered by {SHORT.get(r['trigger'], r['trigger'])} {arrow} (d={abs(r['d_cohen']):.1f})",
            ha="left",
            va="center",
            fontsize=6.8,
            color=INK,
            family="sans-serif",
        )

        for k, (key, y) in enumerate((("d_on", cy - 0.28), ("d_off", cy - 0.48))):
            value = r[key]
            length = abs(value) / max_abs * bar_max
            color = POS if value >= 0 else NEG
            ax.plot([tx, tx], [y - 0.075, y + 0.075], color="#bdbdbd", lw=0.9, zorder=2)
            ax.add_patch(
                plt.Rectangle(
                    (tx, y - 0.045),
                    max(length, 0.012),
                    0.09,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.95 if k == 0 else 0.62,
                    zorder=3,
                )
            )
            ax.text(
                tx + length + 0.07,
                y,
                f"{signed(value)} pp",
                ha="left",
                va="center",
                fontsize=6.4,
                color=color,
                family="sans-serif",
            )

    ax.text(
        FIGSIZE[0] / 2,
        1.90,
        "A weight tells you what a neuron listens to.\nClamping it tells you what it does.",
        ha="center",
        va="center",
        fontsize=10.4,
        fontweight="bold",
        color=INK,
        family="sans-serif",
        linespacing=1.3,
    )
    ax.text(
        FIGSIZE[0] / 2,
        1.50,
        f"Each dish is one hidden neuron on all {meta['n_rows']:,} dates  ·  "
        "top bar = driven to full firing (tanh +1), bottom bar = silenced (tanh -1).\n"
        "Bars are the measured shift in P(second date); the crimson dish is the neuron that reverses the most yes/no decisions.",
        ha="center",
        va="center",
        fontsize=7.0,
        color=GREY,
        family="sans-serif",
        linespacing=1.45,
    )
    ax.text(
        FIGSIZE[0] / 2,
        1.14,
        f"baseline P(yes) = {meta['baseline']:.1%}   ·   command neuron #{meta['winner']}: "
        f"{signed(rows[meta['winner_idx']]['d_on'])} pp on / {signed(rows[meta['winner_idx']]['d_off'])} pp off"
        f"   ·   {meta['flips']:,} decisions reversed",
        ha="center",
        va="center",
        fontsize=7.0,
        color=GREY,
        family="sans-serif",
    )
    ax.text(
        FIGSIZE[0] / 2,
        0.74,
        "after FlyWire's Drosophila connectome (2024): a wiring diagram is a hypothesis, stimulation is the experiment",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#9a9a9a",
        style="italic",
        family="sans-serif",
    )

    fig.savefig(path, dpi=DPI, facecolor="white")
    return path


def main():
    m, W1, b1, W2, b2, mu, sigma = load_model()
    X, y = load_dataset()
    # train.py uses all 8,378 dates with median-filled gaps; inference must too
    X = np.where(np.isnan(X), np.asarray(m["impute"], dtype=float), X)
    Xs = (X - mu) / sigma
    A, p = forward(Xs, W1, b1, W2, b2)

    order = np.argsort(-np.abs(W1).sum(axis=0))
    rows = []
    for pos, j in enumerate(order):
        activation = A[:, j]
        p_on = clamp(A, W2, b2, j, 1.0)
        p_off = clamp(A, W2, b2, j, -1.0)
        trigger, d_cohen = quartile_trigger(activation, X)
        rows.append(
            {
                "label": pos + 1,
                "col": int(j),
                "fires": float((activation > 0).mean()),
                "trigger": FEATURES[trigger],
                "trigger_idx": int(trigger),
                "d_cohen": d_cohen,
                "d_on": float(100.0 * (p_on.mean() - p.mean())),
                "d_off": float(100.0 * (p_off.mean() - p.mean())),
                "no_to_yes": int(((p < 0.5) & (p_on >= 0.5)).sum()),
                "yes_to_no": int(((p >= 0.5) & (p_off < 0.5)).sum()),
                "mean_activation": float(activation.mean()),
            }
        )

    # "Command neuron" = the one whose clamping reverses the most decisions. That is
    # the behavioural readout; a large mean shift that never crosses 0.5 is not.
    winner_idx = max(range(len(rows)), key=lambda i: rows[i]["no_to_yes"] + rows[i]["yes_to_no"])
    winner = rows[winner_idx]
    biggest = max(rows, key=lambda r: max(abs(r["d_on"]), abs(r["d_off"])))

    print(f"\nTHE FRUIT-FLY TEST -- {len(X):,} dates, {A.shape[1]} hidden neurons")
    print(f"baseline P(second date) = {p.mean():.1%}   actual yes rate = {y.mean():.1%}")
    print(
        f"\n{'#':>3}  {'fires':>5}  {'trigger':>19}  {'d':>4}  {'on':>8}  {'off':>8}  "
        f"{'no->yes':>7}  {'yes->no':>7}"
    )
    for r in rows:
        name = f"{SHORT.get(r['trigger'], r['trigger'])} {'up' if r['d_cohen'] > 0 else 'down'}"
        print(
            f"{r['label']:>3}  {r['fires']:>5.0%}  {name:>19}  {abs(r['d_cohen']):>4.1f}  "
            f"{r['d_on']:>+6.1f}p  {r['d_off']:>+6.1f}p  {r['no_to_yes']:>7,}  {r['yes_to_no']:>7,}"
        )
    print(
        f"\ncommand neuron (most decisions reversed): #{winner['label']}  "
        f"stimulate {winner['d_on']:+.1f} pp / silence {winner['d_off']:+.1f} pp  "
        f"({winner['no_to_yes'] + winner['yes_to_no']:,} flips)"
    )
    print(
        f"largest mean shift instead: #{biggest['label']}  "
        f"stimulate {biggest['d_on']:+.1f} pp / silence {biggest['d_off']:+.1f} pp  "
        f"({biggest['no_to_yes'] + biggest['yes_to_no']:,} flips)"
    )
    print(
        f"single-neuron sufficiency: clamping #{winner['label']} on turns "
        f"{winner['no_to_yes']:,} of {(p < 0.5).sum():,} 'no' dates into 'yes'"
    )
    print(f"triggered by {SHORT.get(winner['trigger'], winner['trigger'])} "
          f"({'high' if winner['d_cohen'] > 0 else 'low'} values, d={abs(winner['d_cohen']):.2f})")

    meta = {
        "baseline": float(p.mean()),
        "winner": winner["label"],
        "winner_idx": winner_idx,
        "flips": winner["no_to_yes"] + winner["yes_to_no"],
        "n_rows": int(len(X)),
    }
    render(rows, meta, OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
