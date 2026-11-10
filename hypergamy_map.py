"""The hypergamy map: ten men, ten women, and the arrows the models actually draw.

This replaces the old scale chart. Instead of one abstract score, it answers three
questions on one picture:

  * **the heatmap** -- a 10x10 grid of real mutual probabilities. Row i is a man,
    column j is a woman, and the number is P(he says yes) x P(she says yes), both
    from the trained networks. Brighter means more likely.
  * **the arrows** -- each man's arrow points at the woman he is most likely to
    match with, each woman's arrow at the man she is most likely to match with.
    That is "what average people do", computed, not drawn by hand.
  * **your arrow** -- your profile scored against the same ten women (or men),
    drawn in black, with the cell you land in outlined.

The ten levels are *packages*, not ten real people: looks, age and income all move
together, which is what makes the grid interesting instead of one bright column.

    python hypergamy_map.py            # table + hypergamy_map.svg + .png
    python hypergamy_map.py --svg      # only the picture
    python hypergamy_map.py --demo     # the example profile, with its bands
"""

import argparse
import json
import os
import sys
from functools import lru_cache

import numpy as np

import route
from profile import DEFAULTS, forward, load, market_row, taste_row

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = 10
BASE_AGE = 24.0
BASE_INCOME = 30000.0
BASE_GOAL = 4.0
IMPORTANCE = [30.0, 10.0, 15.0, 15.0, 20.0, 10.0]
SINGULAR = {0: "woman", 1: "man"}
PLURAL = {0: "women", 1: "men"}


def rating(level):
    """Level 1 is a 3/10, level 10 is a 10/10."""
    return 3.0 + 7.0 * (level - 1) / (LEVELS - 1)


def archetype(gender, level):
    r = rating(level)
    profile = dict(DEFAULTS)
    profile.update(
        mode="regular",
        age=BASE_AGE + 1.1 * (level - 1),
        income=BASE_INCOME * (0.55 + 0.15 * level),
        goal=BASE_GOAL,
        self=[r] * 5,
        want=[r] * 6,
        importance=list(IMPORTANCE),
    )
    return profile


def _paths_score(paths, row):
    values = []
    for path in paths:
        model, layers = load(path)
        values.append(forward(model, layers, row))
    return float(np.mean(values))


def p_they_say_yes(person, person_gender):
    """P(a date of the opposite gender says yes to `person`)."""
    return _paths_score(route.market_models(person["age"], "regular", person_gender),
                        market_row(person))


def p_you_say_yes(date, your_gender, importance=None):
    """P(you say yes to `date`): the date's ratings scored with your importance."""
    row = taste_row({
        "age": date["age"],
        "income": date["income"],
        "goal": date["goal"],
        "want": list(date["self"]) + [float(np.mean(date["self"]))],
        "importance": list(importance) if importance is not None
                      else profile_importance(date, your_gender),
    })
    return _paths_score(route.taste_models(date["age"], "regular", your_gender), row)


def profile_importance(date, your_gender):
    """The rater's importance lives on the rater; the archetypes carry their own."""
    return date.get("importance", IMPORTANCE)


@lru_cache(maxsize=1)
def matrix():
    """10x10 mutual probabilities, rows are men, columns are women."""
    men = [archetype(1, k) for k in range(1, LEVELS + 1)]
    women = [archetype(0, k) for k in range(1, LEVELS + 1)]
    grid = np.zeros((LEVELS, LEVELS))
    for i, man in enumerate(men):
        her = p_they_say_yes(man, 1)
        for j, woman in enumerate(women):
            his = p_you_say_yes(
                {"age": woman["age"], "income": woman["income"], "goal": woman["goal"],
                 "self": woman["self"], "importance": man["importance"]},
                1,
            )
            grid[i, j] = his * her
    return grid


def _bands(probs):
    top = max(max(probs), 1e-9)
    bands = []
    for k, p in enumerate(probs, start=1):
        share = p / top
        band = "likely" if share >= 0.60 else "stretch" if share >= 0.25 else "hard"
        bands.append({"level": k, "mutual": float(p), "band": band, "share": share})
    return bands


def partner_map(profile, gender, partner_gender=None):
    """Everything the picture and the GUI need for one profile.

    `gender` is you and `partner_gender` is who you are interested in. They are two
    separate inputs, so the app never assumes you are straight or gay. The trained
    models only ever saw opposite-gender dates, and `same_gender` says when the
    answer is therefore an extrapolation rather than a measured one.
    """
    grid = matrix()
    if partner_gender is None:
        partner_gender = 1 - gender
    targets = [archetype(partner_gender, k) for k in range(1, LEVELS + 1)]
    their_yes = p_they_say_yes(profile, gender)
    probs = np.array([
        p_you_say_yes(target, gender, profile["importance"]) * their_yes
        for target in targets
    ])
    best = int(np.argmax(probs))
    level = int(np.clip(round((float(np.mean(profile["self"])) - 3.0) * (LEVELS - 1) / 7.0) + 1,
                        1, LEVELS))
    bands = _bands(probs)

    rivals = [p_they_say_yes(archetype(gender, k), gender) for k in range(1, LEVELS + 1)]
    percentile = 100.0 * float(np.mean(np.asarray(rivals) < their_yes))

    amb, attr = float(profile["importance"][4]), float(profile["importance"][0])
    hypergamy = 10.0 * (amb - attr) / (amb + attr) if (amb + attr) > 0 else 0.0

    he_curve = [float(np.mean(grid[:, j])) for j in range(LEVELS)]
    her_curve = [float(np.mean(grid[i, :])) for i in range(LEVELS)]

    return {
        "matrix": grid,
        "your_probs": probs,
        "best_level": best + 1,
        "your_level": level,
        "bands": bands,
        "their_yes": float(their_yes),
        "your_yes": float(p_you_say_yes(profile, gender, profile["importance"])),
        "chance": float(probs[best]),
        "chance_your_level": float(probs[level - 1]),
        "chance_typical": float(probs[4]),
        "percentile": percentile,
        "hypergamy": hypergamy,
        "he_curve": he_curve,
        "her_curve": her_curve,
        "gender": gender,
        "partner_gender": partner_gender,
        "same_gender": partner_gender == gender,
    }


def verdict(result):
    """One plain sentence, so nobody has to read a grid to know where they stand."""
    target = PLURAL[result.get("partner_gender", 1 - result["gender"])]
    chance = result["chance"]
    if chance >= 0.40:
        mood = "good" if chance >= 0.55 else "decent"
    else:
        mood = "low"
    tail = (f"With a level {result['your_level']} (your own level) it is "
            f"{result['chance_your_level']:.0%}")
    if result["your_level"] != 5:
        tail += f"; with a middle-of-the-road level 5 it is {result['chance_typical']:.0%}"
    return (f"A {mood} chance: about {chance:.0%} of level {result['best_level']} {target} "
            f"would say yes to you and you would say yes back. {tail}.")


def describe(result):
    target = SINGULAR[result.get("partner_gender", 1 - result["gender"])]
    lines = [verdict(result), ""]
    lines.append(f"best level        {result['best_level']} ({target}) with {result['chance']:.1%} mutual")
    lines.append(f"your own level    {result['your_level']} with {result['chance_your_level']:.1%} mutual")
    lines.append(f"middle level     5 with {result['chance_typical']:.1%} mutual")
    lines.append(f"hypergamy         {result['hypergamy']:+.1f} of 10  "
                 f"(negative = you value looks more than status, positive = the reverse)")
    lines.append(f"attractiveness    {result['their_yes']:.0%} would say yes  "
                 f"(the share of dates in the study pool the market model expects to like you)")
    lines.append(f"rank vs levels    {result['percentile']:.0f} of 100  "
                 f"(share of the ten packages the model puts below you)")
    if result.get("same_gender"):
        lines.append("")
        lines.append("  same-gender: the study only recorded opposite-gender dates, so this")
        lines.append("  is an extrapolation, not a measured number.")
    lines.append("")
    lines.append("  level  mutual   band      what it means")
    for band in result["bands"]:
        note = {"likely": "in range", "stretch": "needs something to change",
                "hard": "out of reach on these stats"}[band["band"]]
        lines.append(f"  {band['level']:>5}  {band['mutual']:>6.1%}   {band['band']:<8}  {note}")
    lines.append("")
    lines.append("  bands compare each level to your best cell: 60% or more is likely,")
    lines.append("  25% to 60% is stretch, below 25% is hard.")
    return "\n".join(lines)


def arrows_svg(result):
    """The arrow panel on its own, for embedding in the GUI."""
    grid = result["matrix"]
    top = float(grid.max()) or 1.0
    you = result["gender"]
    width, height = 680, 420
    left, right, first, step = 96, 560, 54, 34
    best_row = [int(np.argmax(grid[i])) for i in range(LEVELS)]
    best_col = [int(np.argmax(grid[:, j])) for j in range(LEVELS)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="420" fill="#ffffff"/>',
        '<text x="16" y="24" font-size="13" font-weight="bold" fill="#111">'
        'what average people do, and your arrow</text>',
    ]
    for i in range(LEVELS):
        y = first + i * step
        parts.append(f'<circle cx="{left}" cy="{y}" r="5" fill="#c2185b"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4}" font-size="10" text-anchor="end" '
                     f'fill="#6b6b6b">M{i + 1}</text>')
        j = best_row[i]
        ty = first + j * step
        sw = 0.5 + 2.2 * float(grid[i, j]) / top
        parts.append(f'<line x1="{left + 7}" y1="{y}" x2="{right - 7}" y2="{ty}" '
                     f'stroke="#1b9ad6" stroke-opacity="0.6" stroke-width="{sw:.2f}"/>')
    for j in range(LEVELS):
        y = first + j * step
        parts.append(f'<circle cx="{right}" cy="{y}" r="5" fill="#1b9ad6"/>')
        parts.append(f'<text x="{right + 12}" y="{y + 4}" font-size="10" '
                     f'fill="#6b6b6b">W{j + 1}</text>')
        i = best_col[j]
        ty = first + i * step
        sw = 0.5 + 2.2 * float(grid[i, j]) / top
        parts.append(f'<line x1="{left + 7}" y1="{ty}" x2="{right - 7}" y2="{y}" '
                     f'stroke="#c2185b" stroke-opacity="0.6" stroke-width="{sw:.2f}"/>')
    start = first + (result["your_level"] - 1) * step
    end = first + (result["best_level"] - 1) * step
    colour = "#111"
    if you == 1:
        parts.append(f'<line x1="{left + 7}" y1="{start}" x2="{right - 7}" y2="{end}" '
                     f'stroke="{colour}" stroke-width="3"/>')
        parts.append(f'<text x="{left + 16}" y="{start - 8}" font-size="11" '
                     f'font-weight="bold" fill="{colour}">your arrow</text>')
    else:
        parts.append(f'<line x1="{left + 7}" y1="{end}" x2="{right - 7}" y2="{start}" '
                     f'stroke="{colour}" stroke-width="3"/>')
        parts.append(f'<text x="{right - 84}" y="{start - 8}" font-size="11" '
                     f'font-weight="bold" fill="{colour}">your arrow</text>')
    parts.append(f'<text x="16" y="382" font-size="10" fill="#8a8a8a">'
                 f'M1-M10 and W1-W10 are ten made-up packages, not ten real people.</text>')
    parts.append(f'<text x="16" y="402" font-size="10" fill="#8a8a8a">'
                 f'blue: male level to the female level it matches with most often  ·  '
                 f'pink: the reverse  ·  black: you ({result["chance"]:.0%} mutual)</text>')
    parts.append('</svg>')
    return "".join(parts)


PALETTE = ["#ffffff", "#fce9f1", "#f8d2e3", "#f2b3ce", "#e98cb4",
           "#dd6199", "#cd3a7e", "#c2185b"]


def _colour(p, top):
    share = min(1.0, p / max(top, 1e-9))
    return PALETTE[min(len(PALETTE) - 1, int(round(share * (len(PALETTE) - 1))))]


def map_svg(result):
    grid, probs = result["matrix"], result["your_probs"]
    top = float(grid.max())
    you = result["gender"]
    x0, y0, cw, ch = 120, 132, 46, 34
    rows = [f"M{k}" for k in range(1, LEVELS + 1)]
    cols = [f"W{k}" for k in range(1, LEVELS + 1)]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="700" '
        'viewBox="0 0 1080 700" font-family="Helvetica,Arial,sans-serif">',
        '<rect width="1080" height="700" fill="#ffffff"/>',
        '<text x="40" y="46" font-size="26" font-weight="bold" fill="#111">HYPERGAMY MAP</text>',
        '<text x="40" y="70" font-size="12" fill="#6b6b6b">mutual probability for ten men '
        'x ten women, from the trained networks. rows: men. columns: women.</text>',
    ]

    best_row = [int(np.argmax(grid[i])) for i in range(LEVELS)]
    best_col = [int(np.argmax(grid[:, j])) for j in range(LEVELS)]

    for j, name in enumerate(cols):
        parts.append(f'<text x="{x0 + j * cw + cw / 2}" y="{y0 - 10}" font-size="10" '
                     f'text-anchor="middle" fill="#6b6b6b">{name}</text>')
    for i, name in enumerate(rows):
        parts.append(f'<text x="{x0 - 10}" y="{y0 + i * ch + ch / 2 + 4}" font-size="10" '
                     f'text-anchor="end" fill="#6b6b6b">{name}</text>')

    for i in range(LEVELS):
        for j in range(LEVELS):
            p = float(grid[i, j])
            x, y = x0 + j * cw, y0 + i * ch
            parts.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" '
                         f'fill="{_colour(p, top)}"/>')
            if j == best_row[i]:
                parts.append(f'<circle cx="{x + cw / 2}" cy="{y + 7}" r="2" fill="#111"/>')
            if i == best_col[j]:
                parts.append(f'<circle cx="{x + 6}" cy="{y + ch / 2}" r="2" fill="#1b9ad6"/>')

    best = result["best_level"] - 1
    level = result["your_level"] - 1
    if you == 1:
        cell_x, cell_y = x0 + best * cw, y0 + level * ch
    else:
        cell_x, cell_y = x0 + level * cw, y0 + best * ch
    parts.append(f'<rect x="{cell_x - 1}" y="{cell_y - 1}" width="{cw}" height="{ch}" '
                 f'fill="none" stroke="#111" stroke-width="2.5"/>')
    parts.append(f'<text x="{cell_x + cw + 4}" y="{cell_y + ch / 2 + 4}" font-size="11" '
                 f'font-weight="bold" fill="#111">you {result["chance"] * 100:.0f}%</text>')

    for k, colour in enumerate(PALETTE):
        lx = x0 + k * 34
        parts.append(f'<rect x="{lx}" y="{y0 + LEVELS * ch + 14}" width="30" height="10" '
                     f'fill="{colour}" stroke="#eeeeee"/>')
    parts.append(f'<text x="{x0}" y="{y0 + LEVELS * ch + 40}" font-size="9" fill="#8a8a8a">'
                 f'low</text>')
    parts.append(f'<text x="{x0 + len(PALETTE) * 34 - 18}" y="{y0 + LEVELS * ch + 40}" '
                 f'font-size="9" fill="#8a8a8a">high</text>')
    parts.append(f'<text x="{x0 + len(PALETTE) * 34 + 40}" y="{y0 + LEVELS * ch + 23}" '
                 f'font-size="9" fill="#8a8a8a">P(he says yes) x P(she says yes)  ·  '
                 f'black dot = his best column  ·  blue dot = her best row</text>')

    # the arrow panel: 10 men on the left, 10 women on the right
    ax_m, ax_w = 690, 980
    y_top, y_step = 130, 34
    parts.append(f'<text x="{(ax_m + ax_w) / 2}" y="{y_top - 22}" font-size="11" '
                 f'text-anchor="middle" fill="#6b6b6b">what average people do</text>')
    for i in range(LEVELS):
        y = y_top + i * y_step
        parts.append(f'<circle cx="{ax_m}" cy="{y}" r="5" fill="#c2185b"/>')
        parts.append(f'<text x="{ax_m - 12}" y="{y + 4}" font-size="10" text-anchor="end" '
                     f'fill="#6b6b6b">M{i + 1}</text>')
        j = best_row[i]
        ty = y_top + j * y_step
        sw = 0.5 + 2.4 * float(grid[i, j]) / top
        parts.append(f'<line x1="{ax_m + 7}" y1="{y}" x2="{ax_w - 7}" y2="{ty}" '
                     f'stroke="#1b9ad6" stroke-opacity="0.65" stroke-width="{sw:.2f}"/>')
    for j in range(LEVELS):
        y = y_top + j * y_step
        parts.append(f'<circle cx="{ax_w}" cy="{y}" r="5" fill="#1b9ad6"/>')
        parts.append(f'<text x="{ax_w + 12}" y="{y + 4}" font-size="10" fill="#6b6b6b">W{j + 1}</text>')
        i = best_col[j]
        ty = y_top + i * y_step
        sw = 0.5 + 2.4 * float(grid[i, j]) / top
        parts.append(f'<line x1="{ax_m + 7}" y1="{ty}" x2="{ax_w - 7}" y2="{y}" '
                     f'stroke="#c2185b" stroke-opacity="0.65" stroke-width="{sw:.2f}"/>')

    # your arrow
    if you == 1:
        start_y = y_top + (result["your_level"] - 1) * y_step
        end_y = y_top + best * y_step
        parts.append(f'<line x1="{ax_m + 7}" y1="{start_y}" x2="{ax_w - 7}" y2="{end_y}" '
                     f'stroke="#111" stroke-width="3"/>')
        parts.append(f'<text x="{ax_m + 26}" y="{start_y - 8}" font-size="11" font-weight="bold" '
                     f'fill="#111">your arrow</text>')
    else:
        start_y = y_top + (result["your_level"] - 1) * y_step
        end_y = y_top + best * y_step
        parts.append(f'<line x1="{ax_w - 7}" y1="{start_y}" x2="{ax_m + 7}" y2="{end_y}" '
                     f'stroke="#111" stroke-width="3"/>')
        parts.append(f'<text x="{ax_w - 96}" y="{start_y - 8}" font-size="11" font-weight="bold" '
                     f'fill="#111">your arrow</text>')

    parts.append(f'<text x="40" y="612" font-size="16" font-weight="bold" fill="#111">'
                 f'your hypergamy {result["hypergamy"]:+.1f} / 10</text>')
    parts.append(f'<text x="40" y="634" font-size="12" fill="#6b6b6b">'
                 f'attractiveness {result["their_yes"]:.0%} of dates would say yes  ·  '
                 f'chance {result["chance"]:.1%} with level {result["best_level"]}</text>')
    bands = "  ".join(f"{b['level']}:{b['band'][0]}" for b in result["bands"])
    parts.append(f'<text x="40" y="656" font-size="11" fill="#8a8a8a">'
                 f'reach by level (l = likely, s = stretch, h = hard)  {bands}</text>')
    parts.append(f'<text x="40" y="676" font-size="10" fill="#a0a0a0">'
                 f'levels are packages: looks, age and income move together  ·  '
                 f'both panels are the trained networks, nothing is hand-drawn</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def average_svg(result):
    width, height = 760, 430
    left, right, top, bottom = 80, 40, 60, 70
    inner_w, inner_h = width - left - right, height - top - bottom
    values = result["he_curve"] + result["her_curve"]
    ymax = max(0.05, max(values) * 1.15)

    def px(k):
        return left + (k - 1) * inner_w / (LEVELS - 1)

    def py(v):
        return top + inner_h - inner_h * v / ymax

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="30" y="32" font-size="19" font-weight="bold" fill="#111">'
             f'AVERAGE ATTRACTION</text>',
             f'<text x="30" y="50" font-size="11" fill="#6b6b6b">'
             f'mean mutual probability by level, men and women averaged separately</text>']

    for g in range(5):
        v = ymax * g / 4
        y = py(v)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" y2="{y:.1f}" '
                     f'stroke="#eeeeee"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 3:.1f}" font-size="9" text-anchor="end" '
                     f'fill="#8a8a8a">{v * 100:.0f}%</text>')
    for k in range(1, LEVELS + 1):
        parts.append(f'<text x="{px(k):.1f}" y="{top + inner_h + 18}" font-size="9" '
                     f'text-anchor="middle" fill="#8a8a8a">{k}</text>')
    parts.append(f'<text x="{left + inner_w / 2}" y="{top + inner_h + 38}" font-size="10" '
                 f'text-anchor="middle" fill="#6b6b6b">level (looks, age and income move together)</text>')

    for curve, colour, label in ((result["he_curve"], "#1b9ad6", "he says yes"),
                                 (result["her_curve"], "#c2185b", "she says yes")):
        pts = " ".join(f"{px(k):.1f},{py(v):.1f}" for k, v in enumerate(curve, start=1))
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.5" points="{pts}"/>')
        for k, v in enumerate(curve, start=1):
            parts.append(f'<circle cx="{px(k):.1f}" cy="{py(v):.1f}" r="3" fill="{colour}"/>')

    parts.append(f'<text x="{left + 8}" y="{top - 16}" font-size="11" fill="#1b9ad6">'
                 f'&#9679; he says yes</text>')
    parts.append(f'<text x="{left + 118}" y="{top - 16}" font-size="11" fill="#c2185b">'
                 f'&#9679; she says yes</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def render(profile, gender, prefix="hypergamy_map"):
    result = partner_map(profile, gender)
    svg_path = os.path.join(HERE, prefix + ".svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(map_svg(result))
    avg_path = os.path.join(HERE, "hypergamy_average.svg")
    with open(avg_path, "w", encoding="utf-8") as f:
        f.write(average_svg(result))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=170)
        ax.imshow(result["matrix"], cmap="RdPu", vmin=0, vmax=float(result["matrix"].max()))
        ax.set_xticks(range(LEVELS), [f"W{k}" for k in range(1, LEVELS + 1)], fontsize=8)
        ax.set_yticks(range(LEVELS), [f"M{k}" for k in range(1, LEVELS + 1)], fontsize=8)
        for i in range(LEVELS):
            ax.add_patch(Circle((int(np.argmax(result["matrix"][i])), i), 0.18,
                                fill=False, ec="#111", lw=1.0))
        ax.set_title("HYPERGAMY MAP  ·  mutual probability (men x women)", fontsize=11)
        ax.set_xlabel(f"your arrow: level {result['best_level']}  ·  chance "
                      f"{result['chance']:.1%}  ·  hypergamy {result['hypergamy']:+.1f}/10",
                      fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, prefix + ".png"), facecolor="white")
        plt.close(fig)
    except Exception as exc:                      # matplotlib is optional for the svg
        print(f"(no png: {exc})")
    return result, svg_path, avg_path


def main():
    parser = argparse.ArgumentParser(description="the 10x10 hypergamy map")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--svg", action="store_true")
    parser.add_argument("--gender", type=int, choices=[0, 1], default=1)
    parser.add_argument("--age", type=float, default=24.0)
    parser.add_argument("--income", type=float, default=32000.0)
    parser.add_argument("--self", dest="self_", default="7,7,7,7,5")
    parser.add_argument("--want", default="8,6,7,8,9,7")
    parser.add_argument("--importance", default="30,10,15,15,20,10")
    args = parser.parse_args()

    profile = dict(DEFAULTS)
    profile.update(mode="regular", age=args.age, income=args.income,
                   self=[float(v) for v in args.self_.split(",")],
                   want=[float(v) for v in args.want.split(",")],
                   importance=[float(v) for v in args.importance.split(",")])
    result, svg_path, avg_path = render(profile, args.gender)
    print(describe(result))
    print(f"\nwrote {svg_path}\nwrote {avg_path}")
    print(f"wrote {os.path.join(HERE, 'hypergamy_map.png')}")


if __name__ == "__main__":
    main()
