# AGENTS.md — read this before touching anything

Project: **HYPERGAMY** — a tiny neural network trained on a small Kaggle dataset,
rendered as the "hypergamy" meme (two columns of people icons with arrows crossing
between them), plus a stimulation experiment that shows what each hidden neuron
actually does.

The joke is the point. The model is real and its weights are real, but nobody here
needs a good classifier: they need a picture that looks like the meme. Do not
sacrifice the picture for benchmark numbers, and do not fake the picture by drawing
random arrows — **every arrow must come from the trained weight matrix.**

---

## 1. Quick start

```bash
python -m pip install numpy matplotlib   # only deps, no torch, no sklearn
python train.py                          # -> weights.json  (also verifies gradients)
python visualize.py                      # -> hypergamy_network.png
python neurons.py                        # -> fruitfly_neurons.png (experiment 02)
python predict.py                        # score one date, neuron by neuron
```

Python 3.14 was used, numpy 2.5.4, matplotlib 3.11.2. Runtime is a few seconds;
900 epochs of full-batch Adam on 3,581 rows is trivial.

## 2. Layout

```
train.py                  dataset loading, NumPy MLP, Adam, gradient check, weights.json
visualize.py              reads weights.json, draws hypergamy_network.png
neurons.py                clamp-one-neuron experiment, draws fruitfly_neurons.png
predict.py                single-date inference with a logit decomposition
weights.json              GENERATED artifact (do not hand-edit)
hypergamy_network.png     GENERATED artifact (7.5x10 in @ 200 dpi = 1500x2000)
fruitfly_neurons.png      GENERATED artifact (7.5x10 in @ 200 dpi)
data/speed_dating.csv     dataset, committed on purpose (428 KB, offline-friendly)
README.md                 human-facing readme
done-and-needs-to-be-done.md   status + backlog
AGENTS.md                 this file
```

All scripts use `os.path.dirname(os.path.abspath(__file__))`, so they work from
any cwd. Keep it that way. `neurons.py` imports `load_dataset`/`FEATURES` from
`train.py` and `SHORT` from `visualize.py`; keep both import-safe (no work at
module level beyond constants).

## 3. The dataset

Kaggle: `annavictoria/speed-dating-experiment` (CC0). Columbia University speed
dating experiment, 2002–2004, Fisman & Iyengar.

The committed CSV is the trimmed public copy from
`https://raw.githubusercontent.com/keithmcnulty/speed_dating/master/speed_data_data.csv`
— 428 KB, 8,378 rows, 15 columns:

`gender, age, income, goal, career, dec, attr, sinc, intel, fun, amb, shar, like, prob, met`

* `dec` — 0/1, did this person want a second date. **This is the target.**
* `attr, sinc, intel, fun, amb, shar` — 1–10 ratings of the date.
* `like` — 1–10 overall, `prob` — 1–10 probability the feeling is mutual, `met` — 1–3.
* `goal`, `career`, `gender` — categorical-ish (`career` is free text, unusable as-is).

**Gotcha:** `train.py` skips any row with a blank field, so only **3,581 of 8,378**
rows survive. That is deliberate and stable. Do not "fix" it by imputing unless a
task actually needs more data — the row count is printed and baked into the image
caption, so changing it changes the artifact.

10 inputs used: `age, income, goal, attr, sinc, intel, fun, amb, shar, like`.

## 4. The model

```
10 inputs -> 10 hidden (tanh) -> 1 output (sigmoid)
W1: 10x10, b1: 10, W2: 10x1, b2: 1
loss: binary cross-entropy | optimiser: hand-rolled Adam (b1 .9, b2 .999, eps 1e-8)
lr 0.08, 900 full-batch epochs, val split 20%, SEED = 7
inputs standardised with train-set mean/std (sigma == 0 -> 1)
```

Reference numbers (do not regress silently):
`train accuracy 82.5%`, `val accuracy 74.6%` (95% bootstrap CI 71.4–77.7%),
majority-class baseline 60.0% → **lift +14.6 pp**,
a date rated 10/10 on everything → `P(second date) = 61.3%`,
**gradient check 1.1e-08** (finite differences, scale-normalised, limit 1e-5).

Val loss starts climbing after ~epoch 200 (overfitting); that is known and accepted,
because 10 hidden units on 2,800 rows is a meme, not a paper. If someone asks for a
better model, add regularisation or early stopping — do not quietly change the
architecture, the drawing depends on it being 10x10.

`weights.json` now also stores `mu` and `sigma` (the training-set standardisation),
`accuracy_val_ci`, `majority_val`, `grad_check_max_rel_err` and `p_10_out_of_10`.
`predict.py` and `neurons.py` refuse to run without `mu`/`sigma`; if you see that
error, re-run `train.py`.

## 5. How the drawing works (the part that matters)

`visualize.py` builds the figure in **inches**: `fig 7.5x10`, one axes filling
`[0,0,1,1]`, `xlim(0,7.5)`, `ylim(0,10)`, `aspect('equal')`. So one data unit is one
inch and the person icons cannot be stretched by the layout. Keep this system; if
you switch to a normalised 0–1 space, the icons distort.

* People icons are drawn from `Circle` + `Polygon` in `person(ax, cx, cy, h, color)`,
  local coords ±0.5 in units of `h`. Head `r=0.118h` at `y=+0.375h`, torso/arms/legs
  as quads. This is why they look like restroom signs without needing a font.
* Left column `x=1.45` pink `#c2185b`, right column `x=6.05` blue `#1b9ad6`,
  `icon_h=0.62`, rows `linspace(8.20, 1.66, 10)`.
* Arrows are `FancyArrowPatch`, `arrowstyle="-|>"`, dashed `(0,(3.2,2.2))`,
  `lw = 0.7 + 2.0*strength`, `alpha = 0.55 + 0.45*strength`,
  `mutation_scale = 7 + 7*strength`, where `strength = |w| / max|W|`.
* Colour by sign: `POS "#a3144a"` (pushes toward yes), `NEG "#12557f"`.
* Only the **top 24 of 100** connections by `|w|` are drawn (`TOP_CONNECTIONS`).
  All 100 is unreadable soup. If you raise it, check the render before committing.
* Hidden units are **sorted by total incoming `|w|`** so popular neurons sit on top
  and arrowheads pile up there — that funnel is what sells the meme. This is legal:
  hidden units are permutation-symmetric, so any column order is the same model.
  `order = np.argsort(-np.abs(W1).sum(axis=0))`, and labels are renumbered `#1..#10`
  **after** reordering. **That numbering is the project-wide convention**: the fly
  figure and `predict.py` use it too, so "#9" is the same neuron everywhere.
* Endpoint jitter (`rng = default_rng(11)`) scatters arrowheads landing on the same
  neuron so a pile looks like a crowd, not one dot. Seeded → deterministic output.
  If you change the number of arrows or their order, the picture changes; that is
  expected, not a bug.

Copy in the image (meme pastiche, keep it):
title `HYPERGAMY`, column headers `WHOM THEY ARE` / `WHOM IT WANTS`,
caption *"Anything above the expectation arrow would be considered a catch, anything
below would be settling."*, plus a small grey stat line.

## 6. The fruit-fly test (`neurons.py`)

Framing: FlyWire published the complete adult *Drosophila* connectome in 2024; a
wiring diagram is a hypothesis, and stimulation is the experiment. This script runs
that protocol on our 10 hidden neurons.

Per neuron, over all 3,581 dates:
`stimulate` clamp `a_j = +1`, `silence` clamp `a_j = -1`, recompute every prediction;
`d_on`/`d_off` are the mean shift in P(yes) in percentage points; `no->yes` and
`yes->no` are decision flips. The `trigger` is empirical, not from the weights:
quartile contrast (top 25% vs bottom 25%) of that input against the neuron's
activation, reported as Cohen's d, taking the input with the largest |d|.

Findings (reference, do not regress):
* command neuron **#9**: +21.0 pp on / −8.4 pp off, 1,285 flips, 873 of 2,113 "no"
  dates turned into "yes"; triggered by high `shar` (d=1.37); fires on 23% of dates.
* largest mean shift **#10**: −23.3 pp on / +19.5 pp off, **0 flips**; triggered by
  low `like` (d=2.9). A veto that never changes a decision.
* Five of ten neurons have zero flips. That is a finding, not a bug — do not
  re-rank or re-tune to make every neuron look decisive.

Drawing conventions: same 7.5x10 inch canvas, 2 columns x 5 rows of cells, one
`fly()` per neuron inside a dish (Circle + Ellipses, blue body to match the meme's
hidden-neuron colour). Bars are always signed by the *measured* effect: crimson =
raises P(yes), blue = lowers it. The crimson dish marks the command neuron. Keep
labels at least 0.1 in from cell edges — the bottom row sits close to the caption.

## 7. Known quirks

* `findfont: Failed to find font weight black for DejaVu Sans, now using 700` — the
  title asks for `fontweight="black"`, matplotlib only has bold. Harmless; output is
  bold. Only silence it by lowering the weight, never by adding a font dependency
  unless a task asks for a specific typeface.
* **Per-element relative gradient error is meaningless here.** After training, most
  tanh units sit at `|a| > 0.999` (40 of 60 on a 6-row batch), so their gradients are
  ~1e-11 and finite differences drown in float noise. `gradient_check` therefore
  normalises by the largest gradient in the network. Do not "fix" it back to a
  per-element ratio; it will report 1e-3 on a perfectly correct backward pass.
* `weights.json` and both PNGs are build artifacts. Regenerate them, never patch them.
* This folder is a OneDrive-synced git repo. Avoid giant outputs (GIFs > ~10 MB,
  video) without asking; OneDrive chokes and syncs per-file.
* Dataset is CC0, code has no license file yet.

## 8. Verified state

* `python train.py` → exit 0, prints 3,581 dates / 40.5% said yes, final
  train 82.5% / val 74.6% / CI 71.4–77.7%, majority baseline 60.0%,
  grad check 1.1e-08 PASS, writes `weights.json`.
* Determinism re-checked: retraining reproduces `W1`, `W2` and the metrics exactly
  (JSON equality), and `visualize.py` reproduces a byte-identical PNG (md5 stable).
* `python visualize.py` → exit 0, writes the PNG, `(10x10 layer, 24 arrows)`.
* `python neurons.py` → exit 0, prints the 10-row table, writes `fruitfly_neurons.png`.
* `python predict.py --demo` → exit 0, three scores (61.3% / 28.7% / 85.7%),
  contributions sum to the logit, labels match the drawing.
* Both images visually inspected: no overlapping text, footer inside the canvas.

## 9. Backlog

See `done-and-needs-to-be-done.md`. Highest-value next things: two-panel gender
split (the meme's actual punchline), a training animation, and drawing layer 2.
