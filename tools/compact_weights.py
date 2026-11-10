"""Shrink the trained weights so the repository can ship them.

The full-precision JSON is mostly digits nobody needs: a weight of -1.1483927401
and a weight of -1.14839 give the same prediction to within 1e-5. Rounding every
float to five decimals and dropping the indentation takes the six models from
about 100 KB to about 34 KB, which is small enough to commit and still large
enough to be exact for every practical purpose.

The script verifies itself: it reloads both versions and compares predictions on
random rows of the right width, and refuses to overwrite anything if the largest
difference is above 1e-5.

  python tools/compact_weights.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Without this, `import profile` finds the stdlib profiler, not this project's module.
sys.path.insert(0, HERE)

import profile  # noqa: E402  (path has to be set first)
FILES = [
    "weights.json",
    "weights_deep.json",
    "weights_deep_men.json",
    "weights_deep_women.json",
    "weights_market_men.json",
    "weights_market_women.json",
    "weights_young.json",
]
TOLERANCE = 1e-5
ROWS = 200


def round_floats(value):
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    return value


def main():
    rng = np.random.default_rng(0)
    total_before = total_after = 0
    for name in FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"skip {name} (not trained yet)")
            continue
        before = os.path.getsize(path)
        with open(path, encoding="utf-8") as f:
            model = json.load(f)

        compact = round_floats(model)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(compact, tmp, separators=(",", ":"))
            tmp_path = tmp.name

        width = len(model["mu"])
        probe = rng.uniform(0.0, 100.0, (ROWS, width))
        _, old_layers = profile.load(path)
        _, new_layers = profile.load(tmp_path)
        old = np.array([profile.forward(model, old_layers, row) for row in probe])
        new = np.array([profile.forward(compact, new_layers, row) for row in probe])
        worst = float(np.max(np.abs(old - new)))
        if worst > TOLERANCE:
            os.unlink(tmp_path)
            raise SystemExit(f"{name}: predictions moved by {worst:.2e}, refusing to overwrite")

        shutil.copy(path, os.path.join(tempfile.gettempdir(), name + ".full"))
        shutil.move(tmp_path, path)
        after = os.path.getsize(path)
        total_before += before
        total_after += after
        print(f"{name:<26} {before / 1024:7.1f} KB -> {after / 1024:6.1f} KB   "
              f"max |dp| {worst:.1e}")

    if total_before:
        print(f"\ntotal {total_before / 1024:.0f} KB -> {total_after / 1024:.0f} KB "
              f"({100 * (1 - total_after / total_before):.0f}% smaller)")


if __name__ == "__main__":
    main()
