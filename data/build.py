"""Build the datasets the models train on.

The public copies of the Columbia speed-dating study are split in two:

  * the original 195-variable file (preferences, self-ratings, both sides of every
    date) -- but its `income` column is empty;
  * a 15-column convenience copy (income present, no preferences).

They are the same 8,378 rows in the same order -- verified below on every shared
column -- so joining them row-for-row recovers one complete table.

Outputs:
  data/speed_dating_full.csv    8,378 dates, 34 columns
  data/speed_dating_young.csv   3,625 dates where the participant was 18-25

The raw downloads are kept out of git (see .gitignore); this script fetches them
if they are missing.

  python data/build.py
"""

import csv
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_speed_dating_195.csv")
TRIM = os.path.join(HERE, "speed_dating.csv")
OUT_FULL = os.path.join(HERE, "speed_dating_full.csv")
OUT_YOUNG = os.path.join(HERE, "speed_dating_young.csv")

RAW_URL = (
    "https://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/"
    "Speed%20Dating%20Data.csv"
)
TRIM_URL = (
    "https://raw.githubusercontent.com/keithmcnulty/speed_dating/master/"
    "speed_data_data.csv"
)

# Shared columns used to prove the two files are row-aligned before anything is
# written. If they ever disagree the join is refused.
ALIGN = ["dec", "attr", "sinc", "intel", "fun", "amb", "shar", "like", "age", "gender", "goal"]

BASE = ["gender", "age", "income", "goal", "career", "dec",
        "attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob", "met"]
PARTNER = ["dec_o", "match", "age_o", "attr_o", "sinc_o", "intel_o", "fun_o",
           "amb_o", "shar_o", "like_o", "prob_o", "met_o"]
PREF = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
SELF = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"]
IDS = ["iid", "pid", "wave"]
# Pre-event context that the original 34-column export dropped.  These are kept so
# the race/same-race analysis can be reproduced; the networks do not automatically
# consume them.
CONTEXT = ["race", "race_o", "samerace", "imprace", "imprelig", "int_corr"]

COLUMNS = IDS + BASE + PARTNER + PREF + SELF + CONTEXT


def download(url, path):
    if os.path.exists(path):
        return path
    print(f"downloading {os.path.basename(path)} ...")
    # Columbia rejects urllib's default user-agent, so identify ourselves.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 hypergamy/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        blob = response.read()
    with open(path, "wb") as f:
        f.write(blob)
    return path


def read_rows(path, encoding):
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        # The Columbia export starts with a UTF-8 BOM even though the body is latin-1.
        if reader.fieldnames:
            reader.fieldnames = [name.lstrip("\ufeff").strip() for name in reader.fieldnames]
        return list(reader)


def as_float(value):
    text = (value or "").strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def same(a, b):
    """NaN-aware equality, so blanks compare equal."""
    if a != a and b != b:
        return True
    return abs(a - b) < 1e-9


def main():
    download(RAW_URL, RAW)
    download(TRIM_URL, TRIM)

    full = read_rows(RAW, "latin-1")
    trim = read_rows(TRIM, "utf-8-sig")

    if len(full) != len(trim):
        raise SystemExit(f"row counts differ: {len(full)} vs {len(trim)}")

    # The join is positional, so prove the positions agree before trusting them.
    for key in ALIGN:
        bad = sum(1 for f, t in zip(full, trim) if not same(as_float(f.get(key)), as_float(t.get(key))))
        if bad:
            raise SystemExit(f"'{key}' disagrees on {bad:,} rows -- refusing to join")
    print(f"row alignment verified on {len(ALIGN)} columns x {len(full):,} rows")

    header = set(full[0].keys()) if full else set()
    missing = [c for c in COLUMNS if c not in header and c != "income"]
    if missing:
        raise SystemExit(f"raw file is missing columns: {missing}")

    out = []
    for f, t in zip(full, trim):
        row = {key: (f.get(key) or "").strip() for key in COLUMNS if key != "income"}
        row["income"] = (t.get("income") or "").strip()
        out.append(row)

    with open(OUT_FULL, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)

    young = [r for r in out if 18 <= as_float(r["age"]) <= 25]
    with open(OUT_YOUNG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(young)

    def pct(rows, key):
        vals = [as_float(r[key]) for r in rows]
        vals = [v for v in vals if v == v]
        return (sum(vals) / len(vals)) if vals else float("nan")

    from os.path import getsize
    print(f"wrote {OUT_FULL}  {len(out):,} rows, {len(COLUMNS)} columns, "
          f"{getsize(OUT_FULL) / 1e6:.2f} MB")
    print(f"      yes rate {pct(out, 'dec'):.1%}  match rate {pct(out, 'match'):.1%}  "
          f"income present {sum(1 for r in out if r['income']):,}")
    print(f"wrote {OUT_YOUNG}  {len(young):,} rows (age 18-25), "
          f"yes rate {pct(young, 'dec'):.1%}, {getsize(OUT_YOUNG) / 1e3:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
