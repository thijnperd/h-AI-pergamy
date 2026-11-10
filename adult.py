"""Advanced and adult layers, calibrated against published research.

No public dataset pairs height, body type, kinks or body measurements with a decision
to see someone again. So nothing here is *trained*. Instead every input falls into one
of three buckets, and the report prints which one:

  MEASURED    estimated from the 8,378 real speed dates in this project
  POPULATION  a percentile from 59,946 real OKCupid profiles -- descriptive only
  RESEARCH    an effect read off the published literature, cited below

What the literature actually says, and how it is used here:

* **Penis size.** Veale et al. 2015 (BJU International, 15,521 men): erect length mean
  13.12 cm / 5.17 in (SD 1.66 cm), erect girth mean 11.66 cm / 4.59 in. Lever et al.
  2006 (52,031 respondents): 85% of women were satisfied with their partner's size,
  only 15% said erect size was important, and girth mattered more than length. That is
  why the size term here is small and peaks in the middle-upper range: below the
  population median costs a little, the 75th-95th percentile helps a little, and the
  extreme top end stops helping (the nomogram puts 95th percentile near 6.0 in).
* **Breast size.** Swami & Tovee 2013 (n=361): medium breasts were rated most
  attractive (32.7%), then large (24.4%), then very large (19.1%). The US census study
  (2022) found men prefer larger than women do, with B/C the population favourite.
  Breast Size Satisfaction Survey 2020 (18,541 women): 47.5% want larger, 23.2%
  smaller, 29.3% are satisfied -- i.e. preference is broad, not a single peak.
* **Height.** Stulp et al. 2013 (5,782 speed-daters, 128,104 choices): women were most
  likely to choose a man 25 cm taller; men a woman 7 cm shorter; actual matches peaked
  at a 19 cm gap. The male-taller norm is real but modest in realised couples
  (PLOS ONE 2013: assortative r = .18, "not too tall" penalty past ~25 cm).
* **Kink.** Interest is common and practice is not rare: 68.8% of a nationally
  representative Belgian sample reported at least one BDSM fantasy or practice, 46.8%
  had actually done one, 26% called themselves interested and 7.6% identified as
  practitioners (Holvoet 2017, n=1,027). A Norwegian population study found over one
  in three endorsing a BDSM interest or behaviour (n=4,148), a Finnish sample 38%
  interested (n=8,137), and Joyal & Carpentier 45.6% with at least one paraphilic
  interest. The same Norwegian study found BDSM *behaviour* associated with higher
  sexual satisfaction while mere unfulfilled interest was associated with lower
  closeness -- so the effect here is compatibility, not participation, and it is
  deliberately small.
* **Desire discrepancy.** Common and one of the more reliable predictors of lower
  sexual and relationship satisfaction (Willoughby 2013, n=1,054 couples).

Effects are capped at 8 points in total so this layer cannot dominate the two trained
networks. It nudges, it does not decide.

  python adult.py     # the table, the priors and the sources
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PRIORS_PATH = os.path.join(HERE, "data", "population_priors.json")

MEASURED_AGE_GAP = {
    1: {"0-2": 0.5108, "3-5": 0.4684, "6-10": 0.4038, "11-99": 0.4870},
    0: {"0-2": 0.3741, "3-5": 0.3631, "6-10": 0.3610, "11-99": 0.3217},
}
MEASURED_SAME_RACE_PP = {1: 0.07, 0: 4.61}

BODY_TYPES = ["thin", "athletic", "average", "curvy", "full figured", "a little extra"]
BODY_EFFECT = {"thin": 0.5, "athletic": 1.5, "average": 0.0,
               "curvy": 0.8, "full figured": -1.2, "a little extra": -1.2}

# Self-reported size bucket -> percentile of the Veale 2015 nomogram, then effect.
SIZE_EFFECT = {1: -1.5, 2: -0.6, 3: 0.0, 4: 1.0, 5: 0.4}   # 5 = too large stops helping
CUP_EFFECT = {1: -0.4, 2: 0.4, 3: 1.2, 4: 0.8, 5: 0.0}    # C is the modal favourite

GENDERS = {0: "woman", 1: "man"}
MODES = ("regular", "advanced", "adult")
CAP_PP = 8.0

# Typical partner height, from the OKCupid medians (70 in men, 65 in women).
TYPICAL_HEIGHT = {0: 70.0, 1: 65.0}

SOURCES = {
    "penis_size": "Veale et al. 2015, BJU Int 115:978-986 (doi:10.1111/bju.13010); "
                  "Lever et al. 2006, J Sex Med",
    "breast_size": "Swami & Tovee 2013, Arch Sex Behav 42:1199; US census breast study 2022; "
                   "Breast Size Satisfaction Survey 2020 (n=18,541)",
    "height": "Stulp et al. 2013, Anim Behav (5,782 speed-daters); Stulp et al. 2013, PLOS ONE",
    "bdsm": "Holvoet et al. 2017 (J Sex Med 14:1152, n=1,027); Joyal & Carpentier 2017 (n=1,040); "
            "Baranowski et al. 2021 (Norway, n=4,148); Finnish prevalence 2023 (n=8,137); "
            "scoping review 2019 (J Sex Res)",
    "desire": "Willoughby et al. 2013 (n=1,054 married couples)",
}

PREVALENCE = {
    "says erect size is important": "15% of women (Lever 2006)",
    "satisfied with partner size": "85% of women (Lever 2006)",
    "wants larger breasts": "47.5% of women; 29.3% satisfied (BSSS 2020)",
    "men preferring medium breasts": "32.7%; large 24.4%; very large 19.1% (Swami 2013)",
    "BDSM fantasy or practice, any": "68.8% (Holvoet 2017); 38% interested in Finland (2023)",
    "BDSM practice, ever": "46.8% (Holvoet 2017); over 1 in 3 in Norway (2021)",
    "interested in BDSM": "26% (Holvoet 2017)",
    "identifies as kink practitioner": "7.6% (Holvoet 2017)",
    "BDSM and satisfaction": "behaviour associated with higher sexual satisfaction; unfulfilled interest with lower closeness (Norway 2021)",
    "desire discrepancy in couples": "common; predicts lower satisfaction (Willoughby 2013)",
}


def priors():
    try:
        with open(PRIORS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def height_percentile(height_in, gender):
    p = priors().get("height_in", {}).get("men" if gender else "women", {})
    if not p or height_in is None:
        return None
    points = sorted((float(k), float(v)) for k, v in p.items() if k != "n")
    if height_in <= points[0][1]:
        return 1.0
    if height_in >= points[-1][1]:
        return 99.0
    for (p0, h0), (p1, h1) in zip(points, points[1:]):
        if h0 <= height_in <= h1:
            span = (height_in - h0) / max(h1 - h0, 1e-9)
            return p0 + span * (p1 - p0)
    return 50.0


def _gap_bucket(years):
    years = abs(float(years))
    for lo, hi in ((0, 2), (3, 5), (6, 10)):
        if lo <= years <= hi:
            return f"{lo}-{hi}"
    return "11-99"


def _height_effect(profile, gender):
    """Stulp's shape: women choose men ~25 cm taller, men choose women ~7 cm shorter,
    and realised matches peak at a 19 cm gap. Penalise distance from the preferred
    gap, and penalise the "too tall" side a little harder."""
    height = profile.get("height_in")
    if not height:
        return 0.0, "height not given"
    partner = float(profile.get("partner_height_in", TYPICAL_HEIGHT[gender]))
    gap_cm = (partner - float(height)) * 2.54
    wanted = 7.5 if gender == 0 else -2.8          # inches, from Stulp 2013
    off = abs((partner - float(height)) - wanted)
    effect = -0.9 * off
    if gender == 0 and gap_cm > 25:                # the "too tall" penalty
        effect -= 0.6 * (gap_cm - 25) / 5.0
    return max(-3.0, min(3.0, effect)), f"height gap {gap_cm:+.0f} cm"


def adjustments(profile, gender):
    """Return (delta_they_say_yes, delta_you_say_yes, rows) with rows=(label, pp, basis)."""
    rows = []
    d_them = 0.0
    d_you = 0.0

    bucket = _gap_bucket(profile.get("age_gap", 3.0))
    table = MEASURED_AGE_GAP[gender]
    gap_delta = 100.0 * (table[bucket] - table["0-2"])
    d_you += 0.5 * gap_delta
    rows.append((f"age gap {bucket} years", 0.5 * gap_delta, "measured"))

    same_race = bool(profile.get("same_race", True))
    rows.append(("same race" if same_race else "different race",
                 MEASURED_SAME_RACE_PP[gender] if same_race else 0.0, "measured"))

    h = profile.get("height_in")
    if h:
        pct = height_percentile(h, gender)
        if pct is not None:
            rows.append((f"height {h:.0f} in ({pct:.0f}th percentile of real profiles)", 0.0, "population"))

    if profile.get("mode") in ("advanced", "adult") and h:
        effect, note = _height_effect(profile, gender)
        d_them += effect
        rows.append((f"{note}, preferred {7.5 if gender == 0 else -2.8:+.1f} in", effect, "research"))

    body = str(profile.get("body_type", "average")).lower()
    body_delta = BODY_EFFECT.get(body, 0.0)
    d_them += body_delta
    rows.append((f"body type: {body}", body_delta, "research"))

    if profile.get("mode") == "adult":
        d_them += 2.0 if bool(profile.get("orientation_match", True)) else -6.0
        rows.append(("orientation in their pool",
                     2.0 if profile.get("orientation_match", True) else -6.0, "research"))

        kink = float(profile.get("kink_match", 3.0))
        # Compatibility, not presence: the literature says interest is common and
        # practice is linked to satisfaction, so mismatch costs less than fit gains.
        kink_delta = -0.6 + 0.72 * kink
        d_them += kink_delta
        d_you += 0.5 * kink_delta
        rows.append((f"kink compatibility {kink:.0f}/5", kink_delta, "research"))

        libido = float(profile.get("libido_match", 3.0))
        libido_delta = -0.8 + 0.7 * libido
        d_you += libido_delta
        rows.append((f"libido compatibility {libido:.0f}/5", libido_delta, "research"))

        size = int(round(float(profile.get("genital", 3.0))))
        d_them += SIZE_EFFECT.get(size, 0.0)
        rows.append((f"size bucket {size}/5 ({'below median' if size < 3 else 'median' if size == 3 else 'above median'})",
                     SIZE_EFFECT.get(size, 0.0), "research"))

        cup = int(round(float(profile.get("cup", 3.0))))
        d_them += CUP_EFFECT.get(cup, 0.0)
        rows.append((f"breast size bucket {cup}/5 (C = favourite)", CUP_EFFECT.get(cup, 0.0), "research"))

        exp = float(profile.get("experience", 3.0))
        exp_delta = max(-1.0, min(1.5, (exp - 3.0) * 0.5))
        d_you += exp_delta
        rows.append(("relationship experience", exp_delta, "research"))

    return (max(-CAP_PP, min(CAP_PP, d_them)), max(-CAP_PP, min(CAP_PP, d_you)), rows)


def main():
    print("Where every input comes from:\n")
    print(f"  {'input':<58} {'effect':>7}  basis")
    print("  " + "-" * 78)
    for label, delta, basis in adjustments({"mode": "adult", "age_gap": 4.0, "same_race": True,
                                            "height_in": 70, "body_type": "athletic",
                                            "kink_match": 4, "libido_match": 4,
                                            "genital": 4, "cup": 3, "experience": 4}, 1)[2]:
        print(f"  {label:<58} {delta:>+6.1f}  {basis}")

    print("\nWhat the population looks like:")
    for key, value in PREVALENCE.items():
        print(f"  {key:<40} {value}")
    print("\nSources:")
    for key, value in SOURCES.items():
        print(f"  {key:<12} {value}")
    p = priors()
    if p:
        print(f"\nPopulation priors: {p.get('source', '')} ({p.get('n', 0):,} profiles)")
    print("\nOnly the age-gap and same-race rows are measured on real outcomes. "
          "The rest are research priors, not trained weights.")


if __name__ == "__main__":
    main()
