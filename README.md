# HYPERGAMY: a neural network, drawn like the meme

A tiny 10 → 10 → 1 neural network, written from scratch in NumPy, trained on a
small Kaggle dataset, rendered as the hypergamy diagram — and then interrogated
one neuron at a time.

![the diagram](hypergamy_network.png)

![the fruit-fly test](fruitfly_neurons.png)

## Dataset

**Speed Dating Experiment** — Kaggle: [`annavictoria/speed-dating-experiment`](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment)
(CC0, 2002–2004, Columbia University; Fisman & Iyengar, *Gender Differences in Mate
Selection*).

The copy in `data/speed_dating.csv` is a trimmed version of it (428 KB, 8,378 dates,
15 columns) hosted publicly here:
<https://raw.githubusercontent.com/keithmcnulty/speed_dating/master/speed_data_data.csv>

Each row is one four-minute date. The participant rated their partner 1–10 on
attractiveness, sincerity, intelligence, fun, ambitiousness and shared interests,
and then said whether they wanted a second date (`dec`).

Columns: `gender, age, income, goal, career, dec, attr, sinc, intel, fun, amb, shar, like, prob, met`

3,581 rows survive the blank-field filter; 40.5% of them said yes.

## The network

```
10 inputs            hidden layer        output
age       ┐    ┌── h1 ──┐
income    ├────┼── h2 ──┤
goal      ├────┼── h3 ──┼──── P(second date)
attr      ├────┼  ...   ┤
sinc      ├────┼── h9 ──┤
intel     │    └── h10 ─┘
fun       │
amb       │    W1: 10x10   ← this is what gets drawn
shar      │    W2: 10x1
like      ┘
```

* hidden units: `tanh`, 10 of them
* output: `sigmoid` (binary cross-entropy)
* optimiser: Adam, lr 0.08, 900 full-batch epochs
* **74.6% validation accuracy** against a **60.0% majority-class baseline**
  (95% bootstrap interval 71.4–77.7%), and a date rated 10/10 on everything gets
  **61.3%**

## Is it real?

`train.py` proves it on every run instead of asking you to trust it:

* **Backprop is checked against central finite differences.** The worst difference
  between the hand-written gradient and the numerically measured one is
  **1.1e-08** of the largest gradient in the network. Threshold for PASS: 1e-5.
* **The score is reported next to the trivial baseline.** 74.6% accuracy means
  nothing on a dataset where 60.0% of dates say no, so the +14.6 pp lift is printed
  next to it, with a bootstrap interval over the held-out split.
* **Training is deterministic.** Same seed, same weights: `W1`, `W2` and the metrics
  are bit-identical across reruns, and the PNG hashes identically too.
* **Inference uses the saved training statistics.** `weights.json` carries `mu` and
  `sigma`, so `predict.py` and `neurons.py` standardise inputs exactly the way
  training did — no silent train/serve skew.

## The fruit-fly test

In 2024 the FlyWire consortium published the complete connectome of an adult
*Drosophila*: every neuron, every synapse. A wiring diagram is a hypothesis about
what a neuron does; the experiment is to switch the neuron on and watch the animal
behave. `neurons.py` runs the same protocol on this network, which is small enough
to test exhaustively.

For each of the 10 hidden neurons, over all 3,581 dates:

1. **Stimulate** — force its activation to `+1`, recompute every prediction.
2. **Silence** — force it to `-1`, recompute.
3. **Trigger** — top-quartile vs bottom-quartile contrast on each input: which
   rating best separates the dates where this neuron is firing?
4. **Count flips** — how many yes/no decisions actually reverse.

What it finds:

* **Neuron #9 is the command neuron.** Stimulating it raises the average
  probability by **+21.0 pp** and silencing it drops it by 8.4 pp; it reverses
  **1,285** decisions, and turning it on alone converts **873 of 2,113 "no" dates
  into "yes"**. It fires on only 23% of dates, and it is triggered by *high shared
  taste* (d = 1.37). A rare neuron that decides.
* **Neuron #10 is the biggest single influence that never decides anything.**
  Clamping it on moves the average probability by −23.3 pp — the largest mean shift
  in the layer — but flips **zero** decisions. It tracks *low overall like* and
  pushes dates that were already below the line further down.
* Five of the ten neurons have zero flips in either direction. Most of the layer
  modulates the answer; a couple of neurons actually choose it.

The figure is `fruitfly_neurons.png`; the numbers are printed by `python neurons.py`.

## Try it on a date

```bash
python predict.py                  # prompts for the ten inputs, Enter = default
python predict.py attr=10 fun=9    # or pass overrides directly
python predict.py --demo           # three canned dates
```

It prints the probability and the exact logit decomposition — `b2 + Σ a_j·W2_j` —
so every hidden neuron's contribution is visible, with the same `#1..#10` labels as
the drawing.

## Running it

```bash
python -m pip install numpy matplotlib
python train.py        # trains, verifies gradients, writes weights.json
python visualize.py    # reads weights.json, writes hypergamy_network.png
python neurons.py      # clamp-and-measure experiment, writes fruitfly_neurons.png
python predict.py      # score one date, neuron by neuron
```

## How the drawing works

* **Left column** = the 10 inputs, one pink person each.
* **Right column** = the 10 hidden neurons, one blue person each.
* **Arrows** = the 10×10 weight matrix of layer 1. Arrowhead points at the neuron,
  thickness and opacity scale with `|w|`, crimson = positive weight, blue = negative.
* Only the **24 strongest** of the 100 connections are drawn. All 100 is soup.
* Hidden units are **sorted by total incoming weight**, so the popular neurons sit at
  the top and the arrows pile up there. That is allowed: hidden units are
  interchangeable (permutation symmetry), so the order is free to pick.
* That sorted order defines the labels `#1..#10`, and `neurons.py` and `predict.py`
  use the same convention, so "#9" means the same neuron everywhere.

The one thing that isn't a joke: it really is the trained model's weights, and it
really did learn that `like`, `attr` and `fun` matter a lot — the fruit-fly test is
where that stops being a claim and becomes a measurement.
