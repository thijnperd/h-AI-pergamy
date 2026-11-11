# AGENTS.md

Project: **HYPERGAMY** — NumPy neural networks trained on real speed dates, plus a local
native GUI and a profile tool that estimate the chance of a mutual yes, a hypergamy score,
the competitor rank, and the range of partners you can plausibly reach.

Rules, in order:

1. **Every number is either trained, measured, or labelled as a prior.** No invented
   statistics, no decorative arrows, no "dataset" that was generated.
2. **Only one chart family.** The hypergamy map (`hypergamy_map.py`) is the picture:
   a 10x10 heatmap of mutual probabilities with arrows and your own arrow. There is no
   fly experiment, no training GIF, and no second hypergamy chart.
3. **The adult layer never pretends.** There is no public outcome data for kinks or body
   measurements; those effects are `research` priors, capped, and printed as such.

## Quick start

```bash
python -m pip install -r requirements.txt
python data/build.py                 # downloads + joins; writes data/speed_dating_full.csv
python data/build_couples.py --states ca,tx,fl,ny,il  # ACS couple priors
python data/build_height.py         # NHANES height priors
python train.py --snapshots          # drawn model + training frames
python train.py --features taste --layers 24,12 --epochs 400 --lr 0.05 --gender 1 \
                --out weights_deep_men.json
python train.py --features taste --layers 24,12 --epochs 400 --lr 0.05 --gender 0 \
                --out weights_deep_women.json
python train.py --features market --target dec_o --layers 16,8 --gender 1 \
                --out weights_market_men.json
python train.py --features market --target dec_o --layers 16,8 --gender 0 \
                --out weights_market_women.json
python train.py --features dyad --layers 16,8 --gender 1 --out weights_likes_men.json
python train.py --features dyad --layers 16,8 --gender 0 --out weights_likes_women.json
python train.py --features taste --layers 24,12 --min-age 18 --max-age 25 --out weights_young.json
python generate.py --train           # person-generator weights
python gui.py                        # native desktop GUI; --serve keeps the legacy browser UI
python hypergamy_map.py              # the map: svg + png + the reach bands
python profile.py --demo
python partner.py --demo
python adult.py
python -m unittest discover -s tests -v   # 26-test v1.6.0 regression suite
```

## Layout

```
data/build.py          downloads the 195-variable file + the income copy, proves they are
                       row-aligned, writes speed_dating_full.csv and speed_dating_young.csv
data/build_couples.py  ACS 2023 couple-market priors -> data/couple_priors.json
data/build_height.py   NHANES 2021-23 measured heights -> data/height_priors.json
train.py               generic MLP (any depth), feature sets, targets, age filters, CV,
                       gradient check, temperature calibration
route.py               which model runs for which age/mode/gender
profile.py             the product: modes, adjustments, competitor rank, --range
adult.py               measured / population / research effects + sources + prevalence
hypergamy_map.py       the 10x10 map, your arrow, the reach bands, svg (and png)
visualize.py           drawn network -> hypergamy_network.png (legacy figure)
visualize_gender.py    two-panel meme (legacy figure)
visualize_chance.py    looks vs ambition vs income partial effects
gui.py                 launcher: native desktop GUI by default, legacy browser UI with --serve
local_gui.py           stdlib Tk desktop GUI: one window, every model, the map and the bands
couples.py             measured ACS couple context + NHANES height percentile
research.py            published evidence used as capped priors, with sources
likes.py               dyad models: P(a specific package says yes to you)
partner.py             measured who you end up with: ACS couples, no model score
predict.py             single-date logit decomposition
tests/test_hypergamy.py reference profile, average profiles, edge cases, random JSON,
                       map invariants, CV metadata, anatomy direction, HTML hooks
```

Generated (never hand-edited): `weights*.json`, `*.png`, `*.svg`, `training_frames.npz`
(gitignored). `data/raw_*.csv` and `*.zip` are gitignored; `data/population_priors.json`
is committed because it is small and `profile.py` uses it.

## Data

**Speed Dating Experiment** (Columbia, 2002–04, CC0). 8,378 dates, 195 variables.
Joined from two public copies because neither is complete: the 195-variable file has
preferences and self-ratings but an empty `income`; the 15-column copy has income.
`data/build.py` refuses to join unless 11 shared columns agree row-for-row (they do,
1.0000). The export also keeps `race`, `race_o`, `samerace`, `imprace`, `imprelig` and
`int_corr` (47 columns); they are context for analysis, not automatic model inputs.

* target `dec` = did the participant want a second date; `dec_o` = the partner's decision;
  `match` = both said yes (16.5%).
* `gender` 0 = women, 1 = men.
* `goal` is categorical (4 = serious relationship), not ordinal. Never label it "wants kids".
* missingness: income blank on ~half; medians from the **training split** only, stored
  as `impute`.

**OKCupid profiles** (JSE / rudeboybert, 59,946 rows, 22 variables). Descriptive only —
no outcome column. Used for height percentiles, body-type mix and ethnicity, never for
training. Do not “fix” this by inventing a yes/match label; HCMST has relationship
outcomes rather than date decisions, and synthetic match datasets are excluded.

**ACS 2023 1-year PUMS** (Census, CA/TX/FL/NY/IL): 250,172 opposite-sex married or
cohabiting couples, 500,344 partner rows. Used only for measured couple context (partner
age/income, age gap, same ethnicity/education), never as a date-acceptance label.

**NHANES August 2021-August 2023** (CDC, DEMO_L + BMX_L): 6,262 adults with measured
height. Used for height percentiles; the height-income correlation uses the poverty
ratio, not actual income.

## Models (reference numbers — do not regress silently)

| model | arch | CV | holdout | baseline |
|---|---|---|---|---|
| drawn | 10→10→1 | 77.4% ± 1.2 | 75.8% | 57.0% |
| taste all | 16→24→12→1 | 80.0% ± 0.9 | 76.5% | 57.0% |
| taste men | 16→24→12→1 | 81.1% ± 2.1 | 81.3% | 50.1% |
| taste women | 16→24→12→1 | 80.3% ± 1.8 | 76.2% | 65.8% |
| market men | 8→16→8→1 | 66.1% ± 2.5 | 66.4% | 61.5% |
| market women | 8→16→8→1 | 63.9% ± 2.0 | 61.6% | 55.8% |
| 18–25 | 16→24→12→1 | 79.8% ± 0.8 | 82.3% | 55.6% |
| dyad men | 20→16→8→1 | 80.9% ± 1.1 | 81.3% | 50.1% |
| dyad women | 20→16→8→1 | 79.5% ± 0.5 | 78.3% | 65.8% |

All committed weight files run without training. Retraining with the same seed
reproduced every CV above to four decimals; adding race/age-gap inputs made the taste
model worse (78.5% vs 81.1% for men), so they were not forced in. Gradient
checks 2.5e-09 to 1.1e-07. Temperature scaling (`temperature` in each weights file)
calibrates probabilities on held-out data; it never changes a decision.

Feature sets: `base` (10), `taste` (base + six stated preference weights), `market`
(age, income, goal + five self-ratings), `dyad` (20: target age/income/goal + target
self-ratings + mean + rater importance + rater self-ratings). Targets: `dec`, `dec_o`,
`match`.

## Routing

`route.py` decides which weights run: age 18–25 (or mode `18-25`) averages
`weights_deep_{gender}.json` with `weights_young.json`; every other age uses the gender
model alone. The market side always uses `weights_market_{gender}.json` — there is no
age-specific market model and the code says so rather than pretending.

## Gotchas

* A single hidden layer means **two** weight matrices. `net.depth == 2`, not 1 — this
  bug silently dropped `W1` from `weights.json` once.
* `predict()` returns shape `(n,)`; compare to `y.ravel()`, not `y`.
* Impute NaN from `m["impute"]` at inference, and use the right width: the market model
  has 8 inputs, the taste model 16.
* Per-element gradient error is meaningless (saturated tanh). Use the scale-normalised
  metric.
* `import profile` from `tools/` resolves to the stdlib profiler — insert the project
  directory on `sys.path` first.
* The SVG charts are generated text: never hand-edit them, regenerate with
  `python hypergamy_map.py`.

## Honesty rules for the profile tool

* `measured` = estimated from the 8,378 dates, ACS couples or NHANES. Only the age gap,
  same-race flag and the context tables qualify. ACS is *realised couples*, not a date
  decision, and is never fed to the date network.
* `population` = percentile from OKCupid or NHANES. Descriptive, no outcome.
* OKCupid has no date outcome; it may supply a descriptive prior, never a trained
  acceptance label.
* `research` = published effect, cited in `adult.py` and README.
* The whole adult layer is capped at ±8 points so it cannot outvote the networks.
* "Who you end up with" is measured ACS couples plus dyad scores on that measured grid.
  It is not a lifetime model; HCMST is the right data for that and is not shipped.
* The reach bands are relative to your own best cell (≥60% likely, 25–60% stretch,
  below 25% hard). They are a model output, not a promise.
* The four anatomy fields are not interchangeable: `size_cm` (man's own size) and
  `own_cup` (woman's own chest) move `d_them`; `cup` (date's chest) and
  `partner_size_cm` (preferred penis size) move `d_you`. The form hides the unused
  pair; tests assert ignored fields do not move the result.
* Never describe a `research` effect as trained. Never add a statistic without a source.

## Determinism

`SEED = 7`; CV uses `SEED + 1`; bootstrap `SEED + 100`; figure jitter `default_rng(11)`
and `(23)`. Same seed → byte-identical weights and charts.
