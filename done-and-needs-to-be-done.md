# Done / needs to be done

Status as of 2026-11-10.

---

## DONE

### 1. Dataset chosen and committed
Kaggle **Speed Dating Experiment** (`annavictoria/speed-dating-experiment`, CC0),
8,378 rows, 15 columns. `data/speed_dating.csv` is the trimmed public copy.
42.0% said yes; 3,581 rows complete on the 10 inputs.

### 2. Neural network from scratch
10 → 10 (tanh) → 1 (sigmoid), BCE, hand-rolled Adam, pure NumPy, `SEED = 7`.
5-fold CV **77.4% ± 1.2%**, holdout **75.8%** vs 57.0% majority (+18.8 pp),
gradient check **2.5e-09**. A realistic 10/10 date → 98.7%.

### 3. Meme visualization
`visualize.py` → `hypergamy_network.png` (7.5×10 @ 200 dpi). 10 pink inputs → 10 blue
neurons, dashed arrows, thickness ∝ |weight|, top 24 of 100 drawn, hidden units
sorted by incoming mass so the funnel lands at the top. Seeded jitter, deterministic
output. The `goal` label was corrected from "wants kids?" to "dating goal".

### 4. Docs + relocation
README / AGENTS / this file. Project moved to its current folder and verified.

### 5. Verification instead of assertion
Finite-difference gradient check every run (scale-normalised because saturated tanh
units make per-element error meaningless); majority baseline; bootstrap CI;
determinism check (identical weights and PNG hash); `mu`/`sigma`/`impute` stored so
inference matches training. Fixed a label-mapping bug in `predict.py` and a
broadcasting bug that made training accuracy read ~50%.

### 6. The fruit-fly test
`neurons.py` clamps each hidden neuron on/off across all 8,378 dates, measures mean
shift **and** decision flips, and derives an empirical trigger by quartile contrast.
Current: **#6 is the command neuron** (+17.9/−16.3 pp, 4,230 flips, 2,201 "no" dates
flipped, triggered by attractiveness d=3.2); **#9 has the largest mean shift**
(−26.9 pp silenced); three neurons flip nothing. Figure: 10 flies in dishes, 2×5.

### 7. Accuracy improved (same joke)
Selected by 5-fold CV, keeping the 10×10 layer:
* all 8,378 rows instead of 3,581 complete ones (train-split median imputation);
* L2 1e-3, 300 epochs instead of 900 unregularised.
Result: **77.4% CV vs 76.5%** for the old setup, +18.8 pp over baseline.
`prob`/`met` would reach 78.1% but needs 12 inputs, so it was rejected — the shape of
the joke outranks 0.7 pp.

### 8. Training animation
`animate.py` → `training.gif`, 30 frames (a weight snapshot every 10 epochs),
600×800, 64 colours, **1.04 MB**. Pillow directly; imageio is not installed. Final
column order fixed so neurons don't jump between frames.

### 9. Two-panel gender split (the meme's real format)
`train.py --gender 0/1` → `weights_women.json` / `weights_men.json`;
`visualize_gender.py` → `hypergamy_men_women.png`, 11×10, divider, per-panel
accuracy. Men: 78.4% CV / 77.0% val (+26.9 pp over 50.1%). Women: 76.8% CV / 77.5%
val (+11.7 pp over 65.8%).
**Honest finding: the panels look far more similar than the meme claims** — same
funnel, holdouts within 0.5 pp. Reported as-is in README and on the figure.

### 10. Gen Z research + meme copy
Sourced field notes on the two-panel footer and in README: Ipsos/JOE UK 2025 (young
men think women prioritise looks 50% / money 39%; young women rank humour and
kindness first), BofA 2026 (over half of Gen Z spend $0/month on dates), Talker/Chime
2025 (47% of men vs 34% of women think the man should pay), PLOS ONE 2025 (no
significant female hypergamy in English marriages 1837–2021).

### 11. Published to GitHub
`thijnperd/h-AI-pergamy` (private, as created by the owner).

---

## NEEDS TO BE DONE

### 1. Draw layer 2
Only W1 is drawn; hidden→output (10×1) is invisible, so the picture shows a network
that never emits an answer. Add a third column (heart or ring) with its own arrows.
Note it squeezes the 7.5 in width; the two-panel figure has even less room.

### 2. Per-gender fruit-fly test
The fly experiment currently runs on the base model only. Running it on the men's and
women's nets would show whether the command neuron differs by side — probably the
most interesting remaining result.

### 3. Small hygiene
* `requirements.txt` (`numpy`, `matplotlib`, `Pillow`).
* No license file. Dataset is CC0; the code license is the owner's call.
* `training_frames.npz` is gitignored; if the GIF is ever rebuilt elsewhere, rerun
  `train.py` first.

### 4. Optional polish
* Social crops (square/portrait), transparent-background variant.
* A typeface for the title (Anton/Impact-like) if the bold fallback looks thin.

---

## Notes / constraints to respect

* Every arrow must trace to a real trained weight. No decorative arrows.
* Every number in `fruitfly_neurons.png` comes from a recompute, not a weight sign.
* Keep the layer at 10×10 — the drawing is built around exactly 10 people per column.
* Generated files are regenerated, never hand-edited.
* Neuron labels `#1..#10` are the per-network sorted order and must stay consistent
  across all figures and `predict.py`.
* Any new statistic in copy must be sourced. The research section in AGENTS.md lists
  the approved numbers.
* Reference metrics: CV 77.4% / val 75.8% / majority 57.0% / grad 2.5e-09 /
  men 78.4% / women 76.8% / #6 = 4,230 flips.
