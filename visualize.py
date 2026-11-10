"""Draw the trained network as the hypergamy meme.

Left column  = the 10 inputs (pink people)
Right column = the 10 hidden neurons (blue people)
Arrows       = the learned weights of layer 1 (10x10), arrowhead points at the neuron

Neurons are ordered by how much total weight flows into them, so the popular ones
sit at the top and the arrows pile up there. That is not cheating: hidden units
are interchangeable (permutation symmetry), so the order is free to choose.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights.json")
OUT = os.path.join(HERE, "hypergamy_network.png")

PINK = "#c2185b"
BLUE = "#1b9ad6"
POS = "#a3144a"
NEG = "#12557f"
INK = "#111111"
GREY = "#6b6b6b"

FIGSIZE = (7.5, 10.0)
DPI = 200
TOP_CONNECTIONS = 24  # only draw the strongest weights, 100 arrows would be soup

SHORT = {
    "age": "age",
    "income": "income",
    "goal": "dating goal",
    "attr": "attractive",
    "sinc": "sincere",
    "intel": "intelligent",
    "fun": "fun",
    "amb": "ambitious",
    "shar": "shared taste",
    "like": "overall like",
}


def person(ax, cx, cy, h, color):
    """The classic restroom-sign person, drawn from polygons and a circle."""
    def poly(pts):
        ax.add_patch(
            Polygon(
                [(cx + x * h, cy + y * h) for x, y in pts],
                closed=True,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
        )

    ax.add_patch(Circle((cx, cy + 0.375 * h), 0.118 * h, facecolor=color, edgecolor="none", zorder=3))

    poly([(-0.145, 0.235), (0.145, 0.235), (0.115, -0.06), (-0.115, -0.06)])  # torso
    poly([(-0.140, 0.225), (-0.238, 0.150), (-0.196, -0.118), (-0.104, -0.050)])  # left arm
    poly([(0.140, 0.225), (0.238, 0.150), (0.196, -0.118), (0.104, -0.050)])  # right arm
    poly([(-0.107, -0.045), (-0.012, -0.045), (-0.012, -0.500), (-0.098, -0.500)])  # left leg
    poly([(0.107, -0.045), (0.012, -0.045), (0.012, -0.500), (0.098, -0.500)])  # right leg


def main():
    with open(WEIGHTS, encoding="utf-8") as f:
        m = json.load(f)

    W1 = np.asarray(m["W1"], dtype=float)          # (inputs, hidden)
    features = m["features"]
    n_in, n_hidden = W1.shape
    rng = np.random.default_rng(11)

    # Order hidden units by incoming weight mass: the "most wanted" neurons on top.
    incoming = np.abs(W1).sum(axis=0)
    order = np.argsort(-incoming)
    W = W1[:, order]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, FIGSIZE[0])
    ax.set_ylim(0, FIGSIZE[1])
    ax.set_aspect("equal")
    ax.axis("off")

    left_x, right_x = 1.45, 6.05
    icon_h = 0.62
    y_top, y_bottom = 8.20, 1.66
    ys = np.linspace(y_top, y_bottom, n_in)

    for i in range(n_in):
        person(ax, left_x, ys[i], icon_h, PINK)
        person(ax, right_x, ys[i], icon_h, BLUE)

    # ---- labels -------------------------------------------------------------
    for i, name in enumerate(features):
        ax.text(
            left_x - 0.30,
            ys[i],
            SHORT.get(name, name),
            ha="right",
            va="center",
            fontsize=9.5,
            color=GREY,
            style="italic",
        )
    for j in range(n_hidden):
        ax.text(right_x + 0.30, ys[j], f"#{j + 1}", ha="left", va="center", fontsize=8.5, color=GREY)

    # ---- arrows -------------------------------------------------------------
    flat = np.dstack(np.unravel_index(np.argsort(-np.abs(W), axis=None), W.shape))[0]
    chosen = flat[:TOP_CONNECTIONS]
    wmax = float(np.abs(W).max())

    x0 = left_x + 0.245 * icon_h + 0.03
    x1 = right_x - 0.245 * icon_h - 0.03

    for i, j in chosen:
        w = float(W[i, j])
        strength = abs(w) / wmax
        # a little jitter so overlapping arrowheads bunch like a crowd, not one dot
        jy = rng.uniform(-0.10, 0.10) * (1.0 + 3.0 * strength)
        jx = rng.uniform(0.0, 0.10)
        color = POS if w >= 0 else NEG
        ax.add_patch(
            FancyArrowPatch(
                (x0, ys[i]),
                (x1 + jx, ys[j] + jy),
                arrowstyle="-|>",
                mutation_scale=7 + 7 * strength,
                linewidth=0.7 + 2.0 * strength,
                linestyle=(0, (3.2, 2.2)),
                color=color,
                alpha=0.55 + 0.45 * strength,
                shrinkA=0,
                shrinkB=0,
                joinstyle="miter",
                zorder=2,
            )
        )

    # ---- type ---------------------------------------------------------------
    ax.text(
        FIGSIZE[0] / 2,
        9.86,
        "HYPERGAMY",
        ha="center",
        va="top",
        fontsize=44,
        fontweight="black",
        color=INK,
        family="sans-serif",
    )
    for x, title, sub in (
        (left_x + 0.28, "WHOM THEY ARE", "10 inputs"),
        (right_x - 0.28, "WHOM IT WANTS", "10 hidden neurons"),
    ):
        ax.text(
            x,
            8.94,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
            family="sans-serif",
        )
        ax.text(
            x,
            8.70,
            sub,
            ha="center",
            va="center",
            fontsize=8,
            color=GREY,
            family="sans-serif",
        )

    ax.text(
        FIGSIZE[0] / 2,
        0.95,
        "Anything above the expectation arrow would be considered a catch,\n"
        "anything below would be settling.",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        family="sans-serif",
        linespacing=1.35,
    )
    ax.text(
        FIGSIZE[0] / 2,
        0.44,
        f"{m['n_rows']:,} speed dates  ·  val accuracy {m['accuracy_val']:.1%}"
        f"  ·  layer 1 = {n_in * n_hidden} weights, top {TOP_CONNECTIONS} drawn",
        ha="center",
        va="center",
        fontsize=7.2,
        color=GREY,
        family="sans-serif",
    )
    ax.text(
        FIGSIZE[0] / 2,
        0.26,
        "thickness = |weight|  ·  crimson pushes yes  ·  blue pushes no",
        ha="center",
        va="center",
        fontsize=7.2,
        color=GREY,
        family="sans-serif",
    )

    fig.savefig(OUT, dpi=DPI, facecolor="white")
    print(f"wrote {OUT}  ({W.shape[0]}x{W.shape[1]} layer, {TOP_CONNECTIONS} arrows)")


if __name__ == "__main__":
    main()
