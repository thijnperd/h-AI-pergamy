"""Animate training: one GIF of the arrows rearranging themselves.

`train.py` saves a snapshot of every weight every 10 epochs to training_frames.npz.
Frames are rendered here at 600x800 with a 64-colour palette so the whole thing
stays a couple of megabytes -- this folder is synced by OneDrive.

  python animate.py      # -> training.gif
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from visualize import GREY, INK, NEG, POS, SHORT, person

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "training_frames.npz")
OUT = os.path.join(HERE, "training.gif")

W, H = 6.0, 8.0
DPI = 100
TOP_CONNECTIONS = 24


def render(W1, order, epoch, acc, loss):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    left_x, right_x = 1.18, 4.72
    icon_h = 0.44
    ys = np.linspace(6.42, 1.58, len(order))
    Wd = W1[:, order]

    for i in range(len(order)):
        person(ax, left_x, ys[i], icon_h, "#c2185b")
        person(ax, right_x, ys[i], icon_h, "#1b9ad6")

    for i, name in enumerate(SHORT.values()):
        ax.text(left_x - 0.26, ys[i], name, ha="right", va="center", fontsize=6.2,
                color=GREY, style="italic")

    flat = np.dstack(np.unravel_index(np.argsort(-np.abs(Wd), axis=None), Wd.shape))[0]
    wmax = float(np.abs(Wd).max()) or 1.0
    rng = np.random.default_rng(11)  # same jitter every frame -> smooth motion
    x0 = left_x + 0.245 * icon_h + 0.02
    x1 = right_x - 0.245 * icon_h - 0.02

    for i, j in flat[:TOP_CONNECTIONS]:
        w = float(Wd[i, j])
        strength = abs(w) / wmax
        jy = rng.uniform(-0.10, 0.10) * (1.0 + 3.0 * strength)
        jx = rng.uniform(0.0, 0.10)
        ax.add_patch(
            FancyArrowPatch(
                (x0, ys[i]),
                (x1 + jx, ys[j] + jy),
                arrowstyle="-|>",
                mutation_scale=5 + 5 * strength,
                linewidth=0.5 + 1.6 * strength,
                linestyle=(0, (3.2, 2.2)),
                color=POS if w >= 0 else NEG,
                alpha=0.55 + 0.45 * strength,
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )

    ax.text(W / 2, 7.56, "HYPERGAMY", ha="center", va="center", fontsize=19,
            fontweight="bold", color=INK)
    ax.text(W / 2, 7.20, "watch the weights decide who is a catch", ha="center",
            va="center", fontsize=7.2, color=GREY)
    ax.text(W / 2, 0.86, f"epoch {epoch:3d}   ·   val accuracy {acc:.1%}   ·   val loss {loss:.3f}",
            ha="center", va="center", fontsize=7.0, color=GREY)

    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image.convert("P", palette=Image.ADAPTIVE, colors=64)


def main():
    data = np.load(FRAMES)
    W1s = data["W1"]
    # Fixed column order from the final weights, so neurons never jump around.
    order = np.argsort(-np.abs(W1s[-1]).sum(axis=0))

    frames = [
        render(W1s[i], order, int(data["epochs"][i]), float(data["val_acc"][i]), float(data["val_loss"][i]))
        for i in range(len(W1s))
    ]
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=240,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size = os.path.getsize(OUT) / 1024 / 1024
    print(f"wrote {OUT}  ({len(frames)} frames, {size:.2f} MB)")


if __name__ == "__main__":
    main()
