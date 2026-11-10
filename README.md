# HYPERGAMY

A neural network trained on **8,378 real speed dates**, drawn as a 10 x 10 hypergamy
map, and wrapped in a local tool that estimates **your chance of a mutual yes**, **how
far above your own ratings you aim (reaching)**, and **which partners are in reach**.

Every number below comes out of trained weights. No decorative arrows, no invented
statistics.

![the hypergamy map](hypergamy_map.svg)

## Does it actually predict anything?

The graph is the honest answer: five-fold cross-validation on dates the model never saw,
next to the score you get by always guessing the commonest answer.

![accuracy](accuracy.svg)

| model | inputs | 5-fold CV | always-guess baseline | lift |
|---|---|---|---|---|
| taste, men | 16 | **81.1% ± 2.1** | 50.1% | +31.0 |
| taste, women | 16 | **80.3% ± 1.8** | 65.8% | +14.5 |
| taste, everyone | 16 | **80.0% ± 0.9** | 57.0% | +23.0 |
| market, men | 8 | 66.1% ± 2.5 | 61.5% | +4.6 |
| market, women | 8 | 63.9% ± 2.0 | 55.8% | +8.1 |
| map / base | 10 | 77.4% ± 1.2 | 57.0% | +20.4 |
| 18-25 | 10 | 79.8% ± 0.8 | 55.6% | +24.2 |

The taste models (will *you* say yes) clear their baselines by 15-31 points. The market
models (will *they* say yes) are weaker, +5 to +8 points: whether a stranger says yes is
mostly noise. The app prints these CVs next to every model it used, so you can weigh the
number yourself. Re-checked against the committed weights: holdout accuracy 75.8-81.3%
for the taste models, 61.6-66.4% for market.

## The map

Rows are men, columns are women, each cell is P(he says yes) x P(she says yes).

* **the heatmap** — 100 real forward passes, not a drawing.
* **the arrows** — what average people do: each man points at the woman he is most
  likely to match with, each woman at the man she is most likely to match with, black
  dot for his best column, blue dot for her best row.
* **your arrow** — the thick line and the boxed cell, next to your **reaching score
  between -10 and +10**, your **attractiveness** (the share of dates who would say yes
  to you) and your **chance** at the package you reach best.
* **the ten rungs are packages**: looks, age and income move together, which is why the
  grid has structure instead of one bright column.

![average attraction](hypergamy_average.svg)

## What each package is

The rows and columns are **packages**, not ten real women or men. A package is one rung
of a made-up ladder: looks, age and income climb together, and each rung is 10% of the
ladder. Package 1 is the bottom rung, package 10 the top.

| package | looks | age | income |
|---|---|---|---|
| 1 | 3.0 / 10 | 24 | $21.0k |
| 2 | 3.8 / 10 | 25 | $25.5k |
| 3 | 4.6 / 10 | 26 | $30.0k |
| 4 | 5.3 / 10 | 27 | $34.5k |
| 5 | 6.1 / 10 | 28 | $39.0k |
| 6 | 6.9 / 10 | 30 | $43.5k |
| 7 | 7.7 / 10 | 31 | $48.0k |
| 8 | 8.4 / 10 | 32 | $52.5k |
| 9 | 9.2 / 10 | 33 | $57.0k |
| 10 | 10.0 / 10 | 34 | $61.5k |

So "package 10" is **a 34-year-old on about $61.5k rated 10/10 on looks** — the top rung
the model builds, not "the perfect woman", and not a score out of ten. The age and income
climb is what makes the grid a diagonal instead of one bright column.

## Who would like you most

`attractiveness` is one number: the share of dates in the study pool the market model
expects to say yes to someone with your stats (age, income, goal, self-ratings). It is
not split by package, because the model never learned "which rung of the ladder likes
this person"; it learned one yes-rate per person. The app therefore does not claim that
package 9 likes you more than package 2: across a row of the map their yes-rate is the
same and the cell changes because *your* yes changes. What it does answer is **which
package gives the highest mutual chance**, shown as your best-fitting package, and how
that compares with your own package and a middle one.

## Who you can reach

| share of your best cell | band | what it means |
|---|---|---|
| 60% or more | **likely** | in range on these stats |
| 25% – 60% | **stretch** | needs something to change |
| below 25% | **hard** | out of reach on these stats |

## Quick start

```bash
python -m pip install -r requirements.txt
python gui.py
```

Open `http://127.0.0.1:8000/`, fill the form, press **Compute**. The profile is created
in the GUI, not on the command line. Modes: **regular** (ten numbers), **advanced**
(+ height, build, age gap), **adult** (+ kink, libido, size), **18-25** (force the young
model). Pick who you are and who you are interested in; same-gender pairings are flagged
as an extrapolation because the study only recorded opposite-gender dates.

## What is trained (both genders)

Six weight files plus the base model and the 18-25 model are committed. Height, body
type, age gap and the compatibility inputs are **not trained** - no public dataset pairs
them with a second date - so they run on a documented prior, capped at 8 points, and
every row prints its basis: `measured`, `population` or `research`.

The networks, the routing, the map generation and every source:
**[HOWITWORKS.md](HOWITWORKS.md)**. The effect table: `python adult.py`.

## Releases

Each release ships **`hypergamy-runtime.zip`** - GUI, map, command line, weights and the
charts, nothing to train. Build it with `python tools/make_zip.py`;
`.github/workflows/release.yml` attaches it to every published release.
Download: [the latest release](../../releases/latest).

## Limits

The market model is 64-66% accurate, so its chance is an estimate with real error bars.
Partner income is not in the data. The packages are archetypes, the bands are relative
to your own best cell, and the per-package "who likes you" split is not modelled.
