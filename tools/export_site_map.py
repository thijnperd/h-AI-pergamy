"""Write the promo site's 10x10 map from the committed dyad weights.

The site must not ship a hand-copied heatmap. This exporter runs the same
``hypergamy_map.matrix()`` the CLI and GUI use, then records the package
definitions next to it so the page has one source of truth.

    python tools/export_site_map.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import hypergamy_map as hm  # noqa: E402


def main():
    grid = hm.matrix()
    packages = []
    for level in range(1, hm.LEVELS + 1):
        profile = hm.archetype(1, level)
        packages.append({
            "level": level,
            "looks": round(float(profile["self"][0]), 1),
            "age": round(float(profile["age"]), 1),
            "income": round(float(profile["income"]) / 1000.0, 1),
        })

    payload = {
        "generated_by": "tools/export_site_map.py from hypergamy_map.matrix()",
        "note": "100 forward passes through the committed dyad weights; rows men, columns women.",
        "grid": [[round(float(v), 4) for v in row] for row in grid],
        "max": round(float(grid.max()), 4),
        "packages": packages,
    }

    out = os.path.join(HERE, "site", "src", "data", "map.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out} (max {payload['max']})")


if __name__ == "__main__":
    main()
