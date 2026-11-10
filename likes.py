"""What kind of woman likes what kind of man, and the other way round.

The taste model answers "will I say yes to this date" from the rater's stated
importance weights and the date's ratings. The two `likes` models add the
rater's own ratings as well (feature set `dyad`, target `dec`), so the answer
can depend on who is doing the choosing, not only on who is being chosen.

`tables()` fills two 10x10 grids: rows are the ten male packages, columns the
ten female packages. One grid is P(she says yes), one is P(he says yes), and
their product is the mutual chance. `who_likes_you()` answers the personal
version: which package of the gender you are interested in is most likely to
say yes to you.

    python likes.py --matrix        # both grids and the mutual grid
    python likes.py --demo          # a default profile: who likes you most
    python likes.py --payload '{"gender":"man",...}'

The packages are built from self-ratings, while the study rates the date from
the outside. Using a package's self-rating as the date's rating is the same
proxy the app already uses for "you say yes"; it is labelled here rather than
hidden.
"""

import argparse

import numpy as np

import hypergamy_map as hm
from profile import DEFAULTS, forward, load

MODELS = {0: "weights_likes_women.json", 1: "weights_likes_men.json"}
LEVELS = 10


def _row(target, rater):
    """The 20 inputs both likes models expect."""
    attrs = ([target["age"], target["income"], target["goal"]]
             + list(target["self"]) + [float(np.mean(target["self"]))])
    return attrs + list(rater["importance"]) + list(rater["self"])


def p_yes(rater_gender, target, rater):
    """P(the rater says yes to the target)."""
    model, layers = load(MODELS[rater_gender])
    return forward(model, layers, _row(target, rater))


def tables():
    """rows: male packages, columns: female packages."""
    men = [hm.archetype(1, k) for k in range(1, LEVELS + 1)]
    women = [hm.archetype(0, k) for k in range(1, LEVELS + 1)]
    her = np.array([[p_yes(0, man, woman) for woman in women] for man in men])
    his = np.array([[p_yes(1, woman, man) for woman in women] for man in men])
    return men, women, her, his, her * his


def who_likes_you(profile, gender, partner_gender):
    """The package most likely to say yes to you, and the mutual chance."""
    targets = [hm.archetype(partner_gender, k) for k in range(1, LEVELS + 1)]
    their = np.array([p_yes(partner_gender, profile, target) for target in targets])
    yours = np.array([p_yes(gender, target, profile) for target in targets])
    order = np.argsort(-their)
    return {
        "their_yes": their,
        "your_yes": yours,
        "mutual": their * yours,
        "order": order,
        "best": int(order[0]) + 1,
        "best_chance": float(their[order[0]]),
    }


def _grid(matrix, row_label, col_label):
    lines = ["      " + "".join(f"{col_label}{j:>2}  " for j in range(1, LEVELS + 1))]
    for i, row in enumerate(matrix, start=1):
        cells = "".join(f"{100 * v:5.1f} " for v in row)
        lines.append(f"  {row_label}{i:>2} {cells}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matrix", action="store_true",
                        help="print the 10x10 her/his/mutual grids")
    parser.add_argument("--demo", action="store_true",
                        help="score a default profile and list who likes them most")
    parser.add_argument("--payload", metavar="JSON",
                        help="your own profile, the same JSON the GUI posts")
    args = parser.parse_args()

    if args.matrix or not args.demo:
        _, _, her, his, mutual = tables()
        print("P(she says yes to him), rows: men, columns: women")
        print(_grid(her, "M", "W"))
        print("\nP(he says yes to her)")
        print(_grid(his, "M", "W"))
        print("\nmutual P(both say yes)")
        print(_grid(mutual, "M", "W"))
        print("\ncolumn spread of her yes:",
              " ".join(f"{100 * his[:, j].mean():.0f}%" for j in range(LEVELS)))

    if args.demo:
        profile = dict(DEFAULTS)
        profile.update(mode="regular", age=24, income=32000)
        for gender, label, partner in ((1, "man", 0), (0, "woman", 1)):
            out = who_likes_you(profile, gender, partner)
            print(f"\n{label} (default profile) -- packages of the other gender,")
            print("package  P(they say yes)  P(you say yes)  mutual")
            for k in out["order"][:5]:
                print(f"  {k + 1:>7}  {100 * out['their_yes'][k]:>14.1f}%"
                      f"  {100 * out['your_yes'][k]:>13.1f}%"
                      f"  {100 * out['mutual'][k]:>5.1f}%")
            print(f"  most likely to like you: package {out['best']}"
                  f" ({100 * out['best_chance']:.0f}%)")

    if args.payload:
        import json
        raw = json.loads(args.payload)
        profile = dict(DEFAULTS)
        for key in ("age", "income", "goal"):
            if raw.get(key) is not None:
                profile[key] = float(raw[key])
        for key in ("self", "want", "importance"):
            if isinstance(raw.get(key), list) and len(raw[key]) == len(profile[key]):
                profile[key] = [float(v) for v in raw[key]]
        gender = 1 if raw.get("gender") == "man" else 0
        partner = 0 if raw.get("partner_gender", "women") == "women" else 1
        out = who_likes_you(profile, gender, partner)
        print("\nyour profile -- packages of the other gender,")
        print("package  P(they say yes)  P(you say yes)  mutual")
        for k in out["order"][:5]:
            print(f"  {k + 1:>7}  {100 * out['their_yes'][k]:>14.1f}%"
                  f"  {100 * out['your_yes'][k]:>13.1f}%"
                  f"  {100 * out['mutual'][k]:>5.1f}%")
        print(f"  most likely to like you: package {out['best']}"
              f" ({100 * out['best_chance']:.0f}%)")


if __name__ == "__main__":
    main()
