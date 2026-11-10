# Done / needs to be done

Status as of 2026-11-10.

---

## DONE

### 1. Dataset chosen and committed
* Kaggle **Speed Dating Experiment** (`annavictoria/speed-dating-experiment`, CC0).
* `data/speed_dating.csv` — 8,378 rows, 15 columns, 428 KB, public mirror, no
  credentials needed. Thematically perfect for the hypergamy meme (1–10 ratings of a
  date + yes/no second-date decision).
* 3,581 rows are usable after dropping blanks; 40.5% of them said yes.

### 2. Neural network trained from scratch
* `train.py`, pure NumPy: 10 → 10 (tanh) → 1 (sigmoid), BCE, hand-rolled Adam,
  lr 0.08, 900 full-batch epochs, `SEED = 7`.
* Results: **train 82.5%, val 74.6%**; a date rated 10/10 on everything →
  **P(second date) = 61.3%**.
* Writes `weights.json` (weights, biases, metrics, loss history, feature list).

### 3. Meme visualization
* `visualize.py` → `hypergamy_network.png`, 7.5×10 in @ 200 dpi.
* 10 pink people (inputs) → 10 blue people (hidden neurons), dashed arrows with
  arrowheads, thickness/opacity ∝ |learned weight|, crimson = positive,
  blue = negative, top 24 of 100 connections drawn.
* Hidden units sorted by incoming weight mass so the arrows funnel to the top — the
  iconic hypergamy look. Legal because hidden units are permutation-symmetric.
* Seeded endpoint jitter so piled-up arrowheads read as a crowd.

### 4. Docs + relocation
* `README.md` (human-facing), `AGENTS.md` (agent-facing), this file.
* Project copied to `C:\Users\Thijn\OneDrive - SG Were Di\Documents\GitHub\h-AI-pergamy`;
  the old Freebuff workspace was emptied of project files.
* Verified after the move: `train.py` exit 0, `visualize.py` exit 0, PNG regenerated,
  image visually inspected (no collisions, footer inside canvas, funnel intact).

### 5. "Make it really working": verification instead of assertion
* **Finite-difference gradient check** on every run: max |analytic − numeric| divided
  by the largest gradient = **1.1e-08** (PASS at 1e-5). Per-element relative error
  was tried first and is useless here (saturated tanh units have ~1e-11 gradients);
  the scale-normalised metric is the honest one and is documented in AGENTS.md.
* **Majority-class baseline** printed next to accuracy: 60.0% vs 74.6% val, i.e.
  **+14.6 pp lift**, plus a **95% bootstrap CI 71.4–77.7%**.
* **Determinism proven**: retraining reproduces `W1`, `W2` and all metrics exactly;
  `visualize.py` reproduces a byte-identical PNG.
* `weights.json` now stores `mu`/`sigma`, so inference standardises exactly as
  training did. `predict.py` prints the exact logit split `b2 + Σ a_j·W2_j`.
* Two internal bugs found and fixed while verifying: `predict.py` mapped its summary
  labels through the ordering array twice (reported "#7" for the top yes-pusher that
  was really #9), and the "command neuron" was ranked by mean probability shift
  rather than decisions reversed.

### 6. The fruit-fly test (`neurons.py` + `fruitfly_neurons.png`)
* Protocol mirrors FlyWire/optogenetics: clamp each hidden neuron on (`+1`) and off
  (`-1`) across all 3,581 dates, measure the mean shift in P(yes) in percentage
  points **and** the number of yes/no decisions reversed.
* Empirical trigger per neuron: quartile contrast of each input against activation,
  Cohen's d, best input reported. Never reads the weights for this.
* Findings: **#9 is the command neuron** (+21.0 pp on, −8.4 pp off, 1,285 flips,
  873 of 2,113 "no" dates flipped to "yes", triggered by high shared taste);
  **#10 has the largest mean shift** (−23.3 pp on) yet **0 flips** — a veto that
  never changes a decision. Five of ten neurons flip nothing.
* Figure: 10 labelled flies in dishes, 2×5 grid, signed bars, command neuron ringed.
  Visually inspected: caption/bars no longer collide after moving the grid up.

---

## NEEDS TO BE DONE

Ordered by how much it adds to the joke per unit of work.

### 1. Two-panel gender split  ← biggest win, this is the meme's actual punchline
The reference image has *"Whom men would happily partner with"* next to *"Whom women
would happily partner with"*. Today there is one panel, so the punchline is missing.
* Train two nets on `gender == 0` and `gender == 1` (confirm the coding in the data).
* Draw them side by side with a vertical divider, each panel 10×10, shared styling.
* Expect the male panel to look "wide range" and the female panel "narrow" if the
  meme holds — **report what actually happens, do not tune the picture to force it.**
* If this lands, the fruit-fly test should be run per gender too; that comparison is
  probably the most interesting result in the repo.

### 2. Animate training
Export a GIF/MP4 of the arrows while the network trains (every N epochs → one frame,
then `imageio`/`PillowWriter`). Keep it under ~10 MB — OneDrive-synced.

### 3. Draw layer 2
Only W1 is drawn; the hidden→output layer (10×1) is invisible, so the picture shows a
network that never produces an answer. Add a third column (single output node, maybe
a heart or ring) with its own arrows. Note: it will squeeze the 7.5 in width.

### 4. Small hygiene
* `requirements.txt` (`numpy`, `matplotlib` — pin if it ever breaks).
* Nothing is committed to git yet — the folder is a fresh repo with no commits.
  Ask the user before the first commit/push.
* No license file. Dataset is CC0; the code license is the user's call.

### 5. Optional polish
* Variant exports: square/portrait crops for socials, transparent background.
* A specific typeface for the title (Anton/Impact-like) if the bold fallback looks
  too thin — requires shipping a font file.
* Merge `predict.py` and `neurons.py` into one interactive "lab bench" with sliders
  (would need a GUI dependency; not obviously worth it).

---

## Notes / constraints to respect

* Every arrow must trace to a real trained weight. No decorative or random arrows.
* Every number in `fruitfly_neurons.png` must come from a recompute, not from a
  weight sign: clamping means running the network again.
* Keep the layer at 10×10 — the drawing is built around exactly 10 people per column.
* `weights.json` and both PNGs are generated; regenerate, never hand-edit.
* Determinism: `SEED = 7` in `train.py`, `default_rng(11)` in `visualize.py`.
* Neuron labels `#1..#10` are the sorted order from `visualize.py` and must stay
  consistent across all three images/scripts.
* Reference metrics: train 82.5% / val 74.6% / 10-out-of-10 date = 61.3% /
  grad check 1.1e-08 / #9 = 1,285 flips.
