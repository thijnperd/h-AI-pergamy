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

`python profile.py --range` (and the GUI) classifies the ten levels relative to your own
best cell: **likely** at 60% or more of it, **stretch** between 25% and 60%, **hard**
below that. It is a model output with real error bars, not a promise.

## How to use it

```mermaid
flowchart TD
    A["1. install Python 3.10+"] --> B["2. pip install -r requirements.txt"]
    B --> C["3. python gui.py"]
    C --> D["4. open http://127.0.0.1:8000"]
    D --> E{"5. pick a mode"}
    E -- regular --> F["10 numbers"]
    E -- advanced --> G["+ height, build, age gap"]
    E -- adult --> H["+ compatibility inputs"]
    E -- 18-25 --> I["force the young model"]
    F --> J["6. choose man or woman"]
    G --> J
    H --> J
    I --> J
    J --> K["7. age, income, goal"]
    K --> L["8. rate yourself, what you want, what matters"]
    L --> M["9. press Compute"]
    M --> N["10. read chance, adjustments, hypergamy"]
    N --> O["11. read the map and the reach bands"]
    O --> P["12. Ctrl+C to stop"]
```
