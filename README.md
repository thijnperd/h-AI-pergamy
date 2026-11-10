# HYPERGAMY

A neural network trained on **8,378 real speed dates**, drawn as a 10 x 10 hypergamy
map, and wrapped in a local tool that estimates **your chance of a mutual yes**, **how
hypergamous you are**, and **which partners are actually in reach**.

Every number below is computed from trained weights. No decorative arrows, no invented
statistics.

![the hypergamy map](hypergamy_map.svg)

## The map

One picture, three answers.

* **rows are men, columns are women.** Each cell is P(he says yes) x P(she says yes),
  both probabilities from the trained networks. Brighter means more likely.
* **the arrows** are what average people do: every man points at the woman he is most
  likely to match with, every woman points at the man she is most likely to match with.
  A black dot marks each man's best column, a blue dot each woman's best row.
* **your arrow** is the thick black line and the outlined cell, built from your own
  numbers. It also prints your **hypergamy score between -10 and +10**, your
  **attractiveness percentile against same-gender competitors**, and your **chance** at
  the level you reach best.
* **the ten levels are packages**, not ten real people: looks, age and income move
  together, which is why the grid has structure instead of one bright column.

The second chart is the average yes rate at each level for each side:

![average attraction](hypergamy_average.svg)

## Who you can reach

`python profile.py --range` and the GUI classify the ten levels relative to your own best
cell: **likely** at 60% or more of it, **stretch** between 25% and 60%, **hard** below
that. It is a model output with real error bars, not a promise.

## Quick start

```bash
python -m pip install -r requirements.txt   # numpy + matplotlib
python gui.py                               # http://127.0.0.1:8000
python hypergamy_map.py                     # the map, the bands, both svg charts
python profile.py --mode regular --gender 1 --age 24 --income 30000 \
    --self 7,7,7,7,5 --want 8,6,7,8,9,7 --importance 30,10,15,15,20,10 --range
```

The page walks itself: pick a mode, fill the form, press Compute, then read the grid
and the bands. `python profile.py --help`, `python gui.py --help` and
`python hypergamy_map.py --help` list every flag.

## What is inside

| file | what it does |
|---|---|
| `gui.py` | the local page, one compute per profile, nothing leaves the machine |
| `hypergamy_map.py` | the ten by ten map, the reach bands and both svg charts |
| `profile.py` | the same numbers on the command line, plus `--range` and `--chart` |
| `adult.py` | every adjustment with its basis and its source, printed by running it |
| `route.py` | which weight files run for a given age and mode |
| `train.py` | the training loop, the CV and the calibration |

The trained weights are committed, so a clone runs immediately; `train.py` reproduces
them. Seven files cover the taste, market and 18 to 25 models, about 60 KB in total.

## Limits

The market model is 64–66% accurate, so the chance it prints is an estimate with real
error bars. Partner income is not in the data, so status is ambition plus your own
income. The ten levels are archetypes, not people: they rank packages, and a real person
is never exactly level 7. Compatibility effects are priors, not learned weights, and are
capped so they cannot dominate the trained models.
