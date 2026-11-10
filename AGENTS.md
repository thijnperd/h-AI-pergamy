# AGENTS.md — read this before touching anything

Project: **HYPERGAMY** — a tiny neural network trained on a small Kaggle dataset,
rendered as the "hypergamy" meme (two columns of people icons with arrows crossing
between them), plus a stimulation experiment that shows what each hidden neuron
actually does and a two-panel gender split that reproduces the meme's real format.

The joke is the point. The model is real and its weights are real, but nobody here
needs a good classifier: they need a picture that looks like the meme. Do not
sacrifice the picture for benchmark numbers, and do not fake the picture by drawing
random arrows — **every arrow must come from the trained weight matrix.**

---

## 1. Quick start

```bash
python -m pip install numpy matplotlib   # only deps, no torch, no sklearn
python train.py                          # -> weights.json + training_frames.npz
python train.py --gender 1 --out weights_men.json
python train.py --gender 0 --out weights_women.json
python visualize.py                      # -> hypergamy_network.png
python visualize_gender.py               # -> hypergamy_men_women.png
python animate.py                        # -> training.gif
python neurons.py                        # -> fruitfly_neurons.png
python predict.py                        # score one date, neuron by neuron
```

Python 3.14 / numpy 2.5.4 / matplotlib 3.11.2 / Pillow 12.3.0. A full `train.py`
run (5-fold CV + final fit) takes ~15 s; each gender run similar.

## 2. Layout

```
train.py                  data, MLP, Adam+L2, CV, gradient check, gender flag, snapshots
visualize.py              reads weights.json -> hypergamy_network.png
visualize_gender.py       reads weights_men/women.json -> hypergamy_men_women.png
animate.py                reads training_frames.npz -> training.gif (Pillow, no imageio)
neurons.py                clamp-one-neuron experiment -> fruitfly_neurons.png
predict.py                single-date inference with a logit decomposition
weights.json              GENERATED (do not hand-edit)
weights_men.json          GENERATED
weights_women.json        GENERATED
training_frames.npz       GENERATED, gitignored (30 W1 snapshots for the GIF)
hypergamy_network.png     GENERATED, 7.5x10 in @ 200 dpi
hypergamy_men_women.png   GENERATED, 11x10 in @ 200 dpi
fruitfly_neurons.png      GENERATED, 7.5x10 in @ 200 dpi
training.gif              GENERATED, 600x800, 30 frames, ~1.04 MB
data/speed_dating.csv     dataset, committed on purpose (428 KB, offline-friendly)
```

All scripts resolve paths from `__file__`, so they work from any cwd. `neurons.py`
imports from `train.py` and `visualize.py`; `visualize_gender.py` and `animate.py`
import `person`/colours/`SHORT` from `visualize.py`. Keep those modules import-safe:
nothing beyond constants may run at import time.

## 3. The dataset

Kaggle `annavictoria/speed-dating-experiment` (CC0), 2002–2004, Fisman & Iyengar.
Committed copy: trimmed public CSV, 8,378 rows, 15 columns:

`gender, age, income, goal, career, dec, attr, sinc, intel, fun, amb, shar, like, prob, met`

* `dec` — 0/1, did this person want a second date. **Target.**
* `attr, sinc, intel, fun, amb, shar` — 1–10 ratings of the date.
* `like` — 1–10 overall; `prob` — guessed probability the other person said yes; `met` — 1–3.
* `goal` — 1–6 reason for attending (1 = fun night out, 4 = serious relationship).
  It is **categorical**, not ordinal, and is used as a plain number. The old
  "wants kids?" label in `SHORT` was simply wrong; do not restore it.
* `gender` — **0 = women, 1 = men** (Kaggle codebook; the career distribution
  confirms it).

**Missingness is the main data fact.** Only 3,581 rows are complete on the 10
inputs; `income` is blank on 4,099 of 8,378. `load_dataset` returns NaN for blanks
and `train()` fills with **training-split medians**, so all 8,378 dates are used.
Medians are saved as `impute` in `weights.json`. Any script that standardises raw
inputs must fill NaN from `m["impute"]` first — `neurons.py` does; forgetting this
turns whole figures into `nan` (this actually happened once).

10 inputs: `age, income, goal, attr, sinc, intel, fun, amb, shar, like`.

## 4. The model

```
10 inputs -> 10 hidden (tanh) -> 1 output (sigmoid)
W1 10x10, b1 10, W2 10x1, b2 1
loss BCE | Adam (0.9/0.999/1e-8) | lr 0.08 | L2 1e-3 | 300 epochs | val 20% | SEED 7
median imputation from the train split, then standardisation (sigma==0 -> 1)
```

Reference numbers (do not regress silently):

| metric | value |
|---|---|
| 5-fold CV | **77.4% ± 1.2%** |
| holdout train / val | 78.9% / **75.8%** (95% CI 73.8–77.9) |
| majority baseline | 57.0% → **+18.8 pp** |
| gradient check | **2.5e-09** (limit 1e-5) |
| 10/10 attribute date (age 27, goal 4) | **98.7%** |
| men: CV / val / baseline / lift | 78.4% ± 0.9% / 77.0% / 50.1% / +26.9 pp |
| women: CV / val / baseline / lift | 76.8% ± 0.5% / 77.5% / 65.8% / +11.7 pp |

History: the first version used complete rows only (3,581), 900 epochs, no L2, and
scored 76.5% CV / 74.6% holdout. Adding rows + L2 bought ~1 pp; adding `prob` and
`met` would buy ~0.7 pp more but needs 12 inputs, which breaks the 10×10 drawing,
so it is deliberately out. Do not change the architecture for accuracy.

**The 10/10 demo means a plausible date**: age 27, income 69487, goal 4, everything
else 10. Feeding age=10/goal=10 (as the first version did) is out-of-distribution
and produces a meaningless number; do not go back to it.

`weights.json` also stores `mu`, `sigma`, `impute`, `accuracy_cv`, `accuracy_cv_std`,
`accuracy_val_ci`, `majority_val`, `grad_check_max_rel_err`, `p_10_out_of_10`,
`n_complete`, `l2`.

## 5. How the drawings work (the part that matters)

Both figure scripts build in **inches**: one axes filling `[0,0,1,1]` with
`xlim/ylim` equal to the figure size and `aspect('equal')`. One data unit = one
inch, so icons cannot be stretched. Keep this system.

`person(ax, cx, cy, h, color)` draws the restroom-sign figure from a `Circle` and
`Polygon` quads in local coordinates ±0.5·h. Reuse it; do not duplicate the icon.

Single panel (`hypergamy_network.png`, 7.5×10):
* left `x=1.45` pink `#c2185b`, right `x=6.05` blue `#1b9ad6`, `icon_h=0.62`,
  rows `linspace(8.20, 1.66, 10)`.
* 24 of 100 arrows by `|w|`; `lw = 0.7 + 2.0*strength`,
  `alpha = 0.55 + 0.45*strength`, `mutation_scale = 7 + 7*strength`,
  `strength = |w| / max|W|`.
* `POS "#a3144a"` = pushes yes, `NEG "#12557f"` = pushes no.
* Seeded jitter (`default_rng(11)`) so arrowheads pile up like a crowd.

Two panels (`hypergamy_men_women.png`, 11×10): panel width 5.5, `ox = 0 / 5.5`;
`left_x = ox+1.42`, `right_x = ox+4.30`, `icon_h=0.52`, rows `linspace(7.92,1.92,10)`,
divider at `x=5.5`. Left = men's decisions, right = women's. Jitter seed 23.

`#1..#10` is defined by sorting columns on `|W1|.sum(axis=0)` **per network**, and
that numbering must match across `visualize.py`, `visualize_gender.py` (per panel)
and `predict.py`. `neurons.py` uses the base model's order.

`training.gif`: 600×800, 64-colour palette, 240 ms/frame, `disposal=2`. Frames use
the **final** column order so neurons never jump; only the weights move. Budget is
~10 MB because this folder syncs through OneDrive — the current file is 1.04 MB.
`animate.py` uses Pillow directly (imageio is not installed).

## 6. The fruit-fly test (`neurons.py`)

Framing: FlyWire published the complete adult *Drosophila* connectome in 2024; a
wiring diagram is a hypothesis, and stimulation is the experiment. Per hidden
neuron, over all 8,378 dates: clamp `a_j = +1` / `-1`, recompute, and count decision
flips. `d_on`/`d_off` are mean shifts in P(yes) in percentage points. The `trigger`
is empirical — quartile contrast of each input against the neuron's activation,
Cohen's d — and never reads the weights.

Reference findings (current model):
* **#6 command neuron**: +17.9 pp on / −16.3 pp off, 4,230 flips, 2,201 of 4,996
  "no" dates flipped to "yes", triggered by high `attr` (d=3.2).
* **#9 largest mean shift**: −26.9 pp when silenced.
* **#5, #8, #10 flip nothing.** That is a finding, not a bug; do not re-rank to make
  every neuron look decisive.

Drawing: 2×5 cells, `fly()` drawn from ellipses; crimson dish = command neuron; bars
signed by the measured effect. Keep the bottom row clear of the caption (grid ends
at y≈2.33, caption at 1.90).

## 7. Research copy (Gen Z)

The two-panel footer and the README's "Gen Z field notes" carry sourced claims.
**Never invent a statistic.** Current, checked sources:
* Ipsos/JOE UK, May 2025 (n=2,475, GB, 16–24 base 221 men / 154 women): young men
  think women prioritise looks (50%) and financial status (39%); young women rank
  humour and kindness first.
* BofA *Better Money Habits* 2026 (Ipsos, Feb 2026): over half of Gen Z spend $0 a
  month on dates.
* Talker Research for Chime, Aug 2025 (n=2,000): 47% of men vs 34% of women think
  the man should pay for the whole date.
* PLOS ONE, Jan 2025, English marriages 1837–2021: no significant female hypergamy.
Keep the tone: the diagram is a hypothesis, the weights are the test. No moralising,
no red-pill endorsement — the joke is the artefact.

## 8. Known quirks

* `findfont: Failed to find font weight black ...` — `fontweight="black"` falls back
  to bold. Harmless.
* **Broadcasting**: `predict()` returns shape `(n,)`; compare it to `y.ravel()`, never
  to `y` of shape `(n,1)`, or NumPy silently builds an `n×n` comparison and the
  training accuracy comes out ~50%. This bug existed once; don't reintroduce it.
* **NaN imputation at inference** (see §3). `predict.py` takes all 10 inputs, so it
  does not need `impute`; `neurons.py` does.
* Per-element relative gradient error is meaningless here (saturated tanh). Use the
  scale-normalised metric in `gradient_check`.
* Generated artifacts (`weights*.json`, `*.png`, `training.gif`,
  `training_frames.npz`) are never hand-edited. `training_frames.npz` is gitignored;
  the rest are committed on purpose.
* Determinism: `SEED = 7` in `train.py`, `default_rng(11)` / `(23)` for jitter in the
  figures. CV uses `SEED+1`, bootstrap `SEED+100`, so changing those shifts reported
  numbers but not the artifact weights.

## 9. Verified state

* `python train.py` → exit 0; 8,378 dates (3,581 complete); CV 77.4% ± 1.2%;
  holdout 78.9/75.8%; majority 57.0%; grad check 2.5e-09 PASS; 10/10 → 98.7%;
  writes `weights.json` + 30 snapshots.
* `train.py --gender 1/0` → men 78.4% CV / 77.0% val, women 76.8% CV / 77.5% val.
* `visualize.py`, `visualize_gender.py`, `animate.py`, `neurons.py` → exit 0.
  GIF: 30 frames, 600×800, 1.04 MB.
* `predict.py --demo` → exit 0; contributions sum to the logit; labels match.
* All three PNGs and one GIF frame visually inspected: no overlapping text, footers
  inside the canvas, two panels balanced, command-neuron ring on #6.
* `py_compile` clean on all scripts.

## 10. Backlog

See `done-and-needs-to-be-done.md`. Still open: draw layer 2 (hidden→output), a
`requirements.txt`, a license, social crops.
