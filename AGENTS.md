# AGENTS.md

Project: **HYPERGAMY** — a NumPy neural network trained on real speed dates, drawn as
the hypergamy meme, plus a profile tool (regular / advanced / adult) that estimates the
chance of a mutual yes, a hypergamy score, and a competitor rank.

Three rules, in order:

1. **Every number is either trained, measured, or labelled as a prior.** No invented
   statistics, no decorative arrows, no "dataset" that was generated.
2. **The drawing stays 10×10.** It is the joke and the shape of the picture. Accuracy
   comes from depth and features, not from redrawing the meme.
3. **The adult layer never pretends.** There is no public outcome data for kinks or
   body measurements; those effects are `research` priors, capped, and printed as such.

## Quick start

```bash
python -m pip install numpy matplotlib
python data/build.py                     # downloads + joins; writes data/speed_dating_full.csv
python train.py --snapshots              # drawn model + training_frames.npz
python train.py --features taste --layers 24,12 --epochs 400 --lr 0.05 --out weights_deep.json
python train.py --features market --target dec_o --layers 16,8 --out weights_market.json
python visualize.py && python visualize_gender.py
python visualize_chance.py && python visualize_scale.py
python animate.py && python neurons.py && python predict.py --demo
python profile.py --demo
python adult.py
```

## Layout

```
data/build.py          downloads the 195-variable file + the income copy, proves they are
                       row-aligned, writes speed_dating_full.csv and speed_dating_young.csv
train.py               generic MLP (any depth), feature sets, targets, age filters, CV,
                       gradient check, temperature calibration
profile.py             the product: modes, adjustments, competitor rank, chart
adult.py               measured / population / research effects + sources + prevalence
visualize.py           drawn network -> hypergamy_network.png
visualize_gender.py    two-panel meme
visualize_chance.py    looks vs ambition vs income partial effects
visualize_scale.py     hypergamy scale + competitor histogram
animate.py             training frames -> training.gif
neurons.py             stimulation experiment -> fruitfly_neurons.png
predict.py             single-date logit decomposition
```

Generated (never hand-edited): `weights*.json`, `*.png`, `training.gif`,
`training_frames.npz` (gitignored). `data/raw_speed_dating_195.csv`, `data/_*.csv` and
`*.zip` are gitignored; `data/population_priors.json` is committed because it is small
and `profile.py` uses it.

## Data

**Speed Dating Experiment** (Columbia, 2002–04, CC0). 8,378 dates, 195 variables.
Joined from two public copies because neither is complete: the 195-variable file has
preferences and self-ratings but an empty `income`; the 15-column copy has income.
`data/build.py` refuses to join unless 11 shared columns agree row-for-row (they do,
1.0000).

* target `dec` = did the participant want a second date; `dec_o` = the partner's decision;
  `match` = both said yes (16.5%).
* `gender` 0 = women, 1 = men.
* `goal` is categorical (4 = serious relationship), not ordinal. Never label it "wants kids".
* missingness: income blank on ~half; medians from the **training split** only, stored
  as `impute`.

**OKCupid profiles** (JSE / rudeboybert, 59,946 rows, 22 variables). Descriptive only —
no outcome column. Used for height percentiles, body-type mix and ethnicity, never for
training.

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

Gradient checks 2.5e-09 to 1.1e-07. Temperature scaling (`temperature` in each weights
file) calibrates probabilities on held-out data; it never changes a decision.

Feature sets: `base` (10), `taste` (base + six stated preference weights), `market`
(age, income, goal + five self-ratings). Targets: `dec`, `dec_o`, `match`.

## Gotchas

* A single hidden layer means **two** weight matrices. `net.depth == 2`, not 1 — this
  bug silently dropped `W1` from `weights.json` once.
* `predict()` returns shape `(n,)`; compare to `y.ravel()`, not `y`.
* Impute NaN from `m["impute"]` at inference, and use the right width: the market model
  has 8 inputs, the taste model 16. Passing taste rows to the market model throws a
  broadcast error.
* Per-element gradient error is meaningless (saturated tanh). Use the scale-normalised
  metric.
* `fontweight="black"` falls back to bold; harmless warning.

## Honesty rules for the profile tool

* `measured` = estimated from the 8,378 dates. Only age gap and same-race qualify.
* `population` = percentile from OKCupid. Descriptive, no outcome.
* `research` = published effect (Veale 2015, Lever 2006, Swami & Tovée 2013, Stulp 2013,
  Holvoet 2017, Willoughby 2013). Cited in `adult.py` and README.
* The whole adult layer is capped at ±8 points so it cannot outvote the networks.
* Never describe a `research` effect as trained. Never add a statistic without a source.

## Determinism

`SEED = 7`; CV uses `SEED + 1`; bootstrap `SEED + 100`; figure jitter `default_rng(11)`
and `(23)`. Same seed → byte-identical weights and PNGs.
