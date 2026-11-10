"""Build `dist/hypergamy-runtime.zip`: the files you need to run the app, nothing else.

The repository is also the tutorial: training scripts, tests and the datasets live
next to the app. This script picks the runtime subset (the GUI, the map, the command
line, the seven weight files and the two charts) so a release can ship one small zip.

    python tools/make_zip.py            # writes dist/hypergamy-runtime.zip
    python tools/make_zip.py --list     # print the manifest and stop
"""

import argparse
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "dist", "hypergamy-runtime.zip")

MANIFEST = [
    "README.md",
    "requirements.txt",
    # the app
    "gui.py",
    "hypergamy_map.py",
    "profile.py",
    "route.py",
    "adult.py",
    "likes.py",
    "generate.py",
    "predict.py",
    "train.py",
    "data/population_priors.json",
    # the trained weights
    "weights.json",
    "weights_deep_men.json",
    "weights_deep_women.json",
    "weights_market_men.json",
    "weights_market_women.json",
    "weights_likes_men.json",
    "weights_likes_women.json",
    "weights_person_men.json",
    "weights_person_women.json",
    "weights_young.json",
    # the charts the README embeds
    "hypergamy_map.svg",
    "hypergamy_average.svg",
]


def main():
    parser = argparse.ArgumentParser(description="build the runtime zip")
    parser.add_argument("--list", action="store_true", help="print the manifest and stop")
    args = parser.parse_args()

    missing = [p for p in MANIFEST if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit("missing files: " + ", ".join(missing))

    if args.list:
        total = sum(os.path.getsize(os.path.join(ROOT, p)) for p in MANIFEST)
        for path in MANIFEST:
            print(f"  {os.path.getsize(os.path.join(ROOT, path)):>8,}  {path}")
        print(f"  {total:>8,}  bytes in {len(MANIFEST)} files")
        return

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in MANIFEST:
            z.write(os.path.join(ROOT, path), path)
    print(f"wrote {DEST}  ({os.path.getsize(DEST):,} bytes, {len(MANIFEST)} files)")


if __name__ == "__main__":
    sys.exit(main())
