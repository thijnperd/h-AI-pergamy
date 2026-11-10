"""Draw the meme the way the meme is actually drawn: two panels, one per side.

Left: what men said yes to. Right: what women said yes to. Same 10x10 layer each,
same styling as the single-panel figure, arrows from the trained weights.

  python visualize_gender.py   # reads weights_men.json + weights_women.json
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from visualize import GREY, INK, NEG, POS, SHORT, person

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hypergamy_men_women.png")

FIGSIZE = (11.0, 10.0)
DPI = 200
PANEL_W = FIGSIZE[0] / 2
TOP_CONNECTIONS = 24


def panel(ax, m, ox, title):
    W1 = np.asarray(m["W1"], dtype=float)
    order = np.argsort(-np.abs(W1).sum(axis=0))
    Wd = W1[:, order]
    n_in, n_hidden = Wd.shape

    left_x, right_x = ox + 1.42, ox + 4.30
    icon_h = 0.52
    ys = np.linspace(7.92, 1.92, n_in)

    for i in range(n_in):
        person(ax, left_x, ys[i], icon_h, "#c2185b")
        person(ax, right_x, ys[i], icon_h, "#1b9ad6")

    for i, name in enumerate(SHORT.values()):
        ax.text(left_x - 0.28, ys[i], name, ha="right", va="center", fontsize=7.4,
                color=GREY, style="italic")
    for j in range(n_hidden):
        ax.text(right_x + 0.27, ys[j], f"#{j + 1}", ha="left", va="center", fontsize=7.0, color=GREY)

    flat = np.dstack(np.unravel_index(np.argsort(-np.abs(Wd), axis=None), Wd.shape))[0]
    wmax = float(np.abs(Wd).max()) or 1.0
    rng = np.random.default_rng(23)
    x0 = left_x + 0.245 * icon_h + 0.02
    x1 = right_x - 0.245 * icon_h - 0.02

    for i, j in flat[:TOP_CONNECTIONS]:
        w = float(Wd[i, j])
        strength = abs(w) / wmax
        jy = rng.uniform(-0.09, 0.09) * (1.0 + 3.0 * strength)
        jx = rng.uniform(0.0, 0.08)
        ax.add_patch(
            FancyArrowPatch(
                (x0, ys[i]),
                (x1 + jx, ys[j] + jy),
                arrowstyle="-|>",
                mutation_scale=5.5 + 5.5 * strength,
                linewidth=0.55 + 1.7 * strength,
                linestyle=(0, (3.2, 2.2)),
                color=POS if w >= 0 else NEG,
                alpha=0.55 + 0.45 * strength,
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )

    cx = ox + PANEL_W / 2
    ax.text(cx, 8.72, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK)
    ax.text(
        cx, 8.44,
        f"{m['n_rows']:,} dates  ·  {m['accuracy_val']:.1%} val accuracy",
        ha="center", va="center", fontsize=7.4, color=GREY,
    )


def main():
    with open(os.path.join(HERE, "weights_men.json"), encoding="utf-8") as f:
        men = json.load(f)
    with open(os.path.join(HERE, "weights_women.json"), encoding="utf-8") as f:
        women = json.load(f)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, FIGSIZE[0])
    ax.set_ylim(0, FIGSIZE[1])
    ax.set_aspect("equal")
    ax.axis("off")

    panel(ax, men, 0.0, "WHOM MEN SAY YES TO")
    panel(ax, women, PANEL_W, "WHOM WOMEN SAY YES TO")

    ax.plot([PANEL_W, PANEL_W], [1.75, 9.0], color="#e2e2e2", lw=1.0, zorder=1)

    ax.text(
        FIGSIZE[0] / 2, 9.62, "HYPERGAMY", ha="center", va="center",
        fontsize=42, fontweight="black", color=INK,
    )
    ax.text(
        FIGSIZE[0] / 2, 9.22,
        "both panels are the trained network's weights  ·  pink is the date, blue is what the network wants",
        ha="center", va="center", fontsize=8.6, color=GREY,
    )

    ax.text(
        FIGSIZE[0] / 2, 1.42,
        "Anything above the expectation arrow is a catch. Anything below is settling.",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK,
    )
    ax.text(
        FIGSIZE[0] / 2, 1.06,
        "Speed-dating data is from 2002-2004. Gen Z inherited the meme, not the market: over half spend $0 a month on dates (BofA, 2026).\n"
        "In a 2025 Ipsos/GB poll, young men said they think women prioritise looks (50%) and money (39%); young women put humour and kindness first.",
        ha="center", va="center", fontsize=7.0, color=GREY, linespacing=1.5,
    )
    ax.text(
        FIGSIZE[0] / 2, 0.64,
        "A 2025 PLOS ONE study of English marriages 1837-2021 found no significant female hypergamy: the diagram is a hypothesis, the weights are the test.",
        ha="center", va="center", fontsize=6.9, color="#8f8f8f", style="italic",
    )
    ax.text(
        FIGSIZE[0] / 2, 0.32,
        "thickness = |weight|  ·  crimson pushes yes  ·  blue pushes no  ·  top 24 of 100 weights drawn",
        ha="center", va="center", fontsize=6.7, color="#9a9a9a",
    )

    fig.savefig(OUT, dpi=DPI, facecolor="white")
    print(f"wrote {OUT}  (2 panels, {men['n_rows']:,} + {women['n_rows']:,} dates)")


if __name__ == "__main__":
    main()
