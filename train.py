"""Train the HYPERGAMY networks from scratch -- pure NumPy, no frameworks.

There are two jobs in this file, and they are deliberately separate:

1. **The drawn network.** 10 inputs -> 10 hidden -> 1. This is the one the meme
   pictures, and its shape is frozen at 10x10 because the drawing is built
   around ten people in each column. `visualize.py` reads its weights back.

2. **The profile network.** Deeper (16 -> 24 -> 12 -> 1 by default) and fed with
   the preferences each participant stated *before* the event, so it can answer
   "will I say yes to this person" as a function of what you want, not just of
   what the person is. `profile.py` uses it.

Both are trained on real rows from the joined 195-variable dataset (see
`data/build.py`). Nothing is synthesised.

Every run checks its own backprop against finite differences and reports 5-fold
CV next to the majority-class baseline, so "it learned" is measured.

  python train.py                                          # the drawn 10x10 model
  python train.py --features taste --layers 24,12 --out weights_deep.json
  python train.py --features taste --layers 24,12 --gender 1 --out weights_deep_men.json
  python train.py --features market --target dec_o --layers 16,8 \
                  --gender 0 --out weights_market_women.json
  python train.py --min-age 18 --max-age 25 --out weights_young.json
"""

import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FULL = os.path.join(HERE, "data", "speed_dating_full.csv")
DATA_TRIM = os.path.join(HERE, "data", "speed_dating.csv")
SNAP = os.path.join(HERE, "training_frames.npz")

# The ten inputs of the drawn network -- one person per input in the left column.
FEATURES = ["age", "income", "goal", "attr", "sinc", "intel", "fun", "amb", "shar", "like"]
TARGET = "dec"

# What the participant said they want, asked before the event (0-100 allocation).
PREF = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
# How the participant rated themselves (1-10); shar3_1 is absent from the source.
SELF = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"]

FEATURE_SETS = {
    "base": FEATURES,
    "taste": FEATURES + PREF,
    "market": ["age", "income", "goal"] + SELF,
    # who says yes to whom: the date without the post-decision "like", plus the
    # rater's stated wants and the rater's own ratings.
    "dyad": [f for f in FEATURES if f != "like"] + PREF + SELF,
}

DEFAULTS = {
    "epochs": 300,
    "lr": 0.08,
    "l2": 1e-3,
    "val_fraction": 0.2,
    "seed": 7,
    "cv_folds": 5,
    "snapshot_every": 10,
}

GENDER_LABEL = {None: "all participants", 0: "women", 1: "men"}


# --------------------------------------------------------------------------- data

def data_path(prefer_full=True):
    if prefer_full and os.path.exists(DATA_FULL):
        return DATA_FULL
    if os.path.exists(DATA_TRIM):
        return DATA_TRIM
    raise SystemExit("no dataset found -- run `python data/build.py` first")


def load_dataset(gender=None, features=None, target=TARGET, path=None,
                 min_age=None, max_age=None):
    """Rows with a target, NaN where an input is blank.

    `gender` 0 = women, 1 = men in the codebook; the drawings rely on that.
    """
    features = list(FEATURES if features is None else features)
    with open(path or data_path(), newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    X, y = [], []
    for row in rows:
        try:
            label = float(row[target])
        except (KeyError, TypeError, ValueError):
            continue
        if gender is not None and str(row.get("gender", "")).strip() != str(gender):
            continue
        age = row.get("age", "")
        if min_age is not None or max_age is not None:
            try:
                age_value = float(age)
            except (TypeError, ValueError):
                continue
            if min_age is not None and age_value < min_age:
                continue
            if max_age is not None and age_value > max_age:
                continue
        values = []
        for key in features:
            raw = (row.get(key) or "").strip()
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(np.nan)
        X.append(values)
        y.append(label)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1, 1)


# ------------------------------------------------------------------------- model

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    """Fully connected, tanh hidden layers, sigmoid output, BCE loss."""

    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    @property
    def depth(self):
        return len(self.W)

    def logits(self, X):
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = np.tanh(z) if i < self.depth - 1 else z
        return np.asarray(a).ravel()

    def activations(self, X):
        a = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a[-1] @ W + b
            a.append(sigmoid(z) if i == self.depth - 1 else np.tanh(z))
        return a

    def forward(self, X):
        a = self.activations(X)
        hidden = a[1] if self.depth == 1 else a[1:-1]
        return hidden, a[-1]

    def predict(self, X):
        return sigmoid(self.logits(X))

    def loss(self, X, y):
        p = np.clip(self.predict(X), 1e-9, 1 - 1e-9)
        return float(-np.mean(y.ravel() * np.log(p) + (1 - y.ravel()) * np.log(1 - p)))

    def gradients(self, X, y):
        a = self.activations(X)
        n = X.shape[0]
        delta = (a[-1] - y) / n                      # dBCE/dlogit
        gW = [None] * self.depth
        gb = [None] * self.depth
        for layer in range(self.depth - 1, -1, -1):
            gW[layer] = a[layer].T @ delta
            gb[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.W[layer].T) * (1.0 - a[layer] ** 2)
        return gW, gb


def gradient_check(net, X, y, rows=6, eps=1e-4):
    """Worst error between backprop and central finite differences, scaled by the
    largest gradient in the network.

    Per-element relative error is meaningless here: saturated tanh units have
    gradients near 1e-11, where finite differences drown in float noise. If the
    hand-written backprop were wrong, training would still limp along and the
    picture would still look fine -- this number is the proof that it isn't.
    """
    X, y = X[:rows], y[:rows]
    gW, gb = net.gradients(X, y)
    worst, largest = 0.0, 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for param, grad in zip(params, grads):
            for idx in np.ndindex(param.shape):
                keep = param[idx]
                param[idx] = keep + eps
                plus = net.loss(X, y)
                param[idx] = keep - eps
                minus = net.loss(X, y)
                param[idx] = keep
                worst = max(worst, abs((plus - minus) / (2.0 * eps) - grad[idx]))
                largest = max(largest, abs(grad[idx]))
    return worst / max(largest, 1e-12)


def bootstrap_ci(correct, iters=2000, seed=107):
    rng = np.random.default_rng(seed)
    n = len(correct)
    draws = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(iters)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def predict(net, mu, sigma, X):
    return net.predict((X - mu) / sigma)


def fit(Xtr, ytr, rng, sizes, epochs=300, lr=0.08, l2=1e-3,
        Xva=None, yva=None, record=False, snapshot_every=10):
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (Xtr - mu) / sigma
    Xv = None if Xva is None else (Xva - mu) / sigma

    net = MLP(sizes, rng)
    m = [np.zeros_like(p) for p in net.W] + [np.zeros_like(p) for p in net.b]
    v = [np.zeros_like(p) for p in net.W] + [np.zeros_like(p) for p in net.b]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    frames = []

    for epoch in range(1, epochs + 1):
        gW, gb = net.gradients(Xs, ytr)
        grads = [g + l2 * w for g, w in zip(gW, net.W)] + list(gb)
        params = net.W + net.b
        for i, (param, grad) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * grad ** 2
            m_hat = m[i] / (1 - beta1 ** epoch)
            v_hat = v[i] / (1 - beta2 ** epoch)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if record and (epoch % snapshot_every == 0 or epoch == epochs):
            frames.append({
                "epoch": epoch,
                "W1": net.W[0].copy(),
                "b1": net.b[0].copy(),
                "W2": net.W[-1].ravel().copy() if net.depth == 2 else np.zeros(1),
                "b2": net.b[-1].copy(),
                "train_loss": net.loss(Xs, ytr),
                "val_loss": net.loss(Xv, yva),
                "val_acc": float(((net.predict(Xv) > 0.5) == yva.ravel()).mean()),
            })

    return net, mu, sigma, frames


def fit_temperature(net, mu, sigma, X, y):
    """Calibrate the confidence of the network on held-out data.

    A deep net trained to 80% accuracy still outputs 0.99 for plenty of wrong
    rows, which would make the profile tool absurd ("your chance is 100%").
    Temperature scaling fits one scalar T so that sigmoid(logit / T) minimises
    held-out log loss. It cannot change a single decision -- the ranking and the
    accuracy are identical -- it only makes the probabilities honest.
    """
    z = net.logits((X - mu) / sigma)
    best_t, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.5, 6.0, 111):
        p = np.clip(sigmoid(z / temperature), 1e-9, 1 - 1e-9)
        nll = float(-np.mean(y.ravel() * np.log(p) + (1 - y.ravel()) * np.log(1 - p)))
        if nll < best_nll:
            best_t, best_nll = float(temperature), nll
    return best_t


def train(X, y, sizes, epochs, lr, l2, val_fraction, seed, record):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - val_fraction))
    Xtr_raw, Xva_raw = X[idx[:cut]], X[idx[cut:]]
    ytr, yva = y[idx[:cut]], y[idx[cut:]]

    # Medians come from the training split only; income is blank on half the rows.
    median = np.nanmedian(Xtr_raw, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
    Xtr = np.where(np.isnan(Xtr_raw), median, Xtr_raw)
    Xva = np.where(np.isnan(Xva_raw), median, Xva_raw)

    net, mu, sigma, frames = fit(Xtr, ytr, rng, sizes, epochs=epochs, lr=lr, l2=l2,
                                 Xva=Xva, yva=yva, record=record)

    correct = ((predict(net, mu, sigma, Xva) > 0.5) == yva.ravel()).astype(float)
    ci_lo, ci_hi = bootstrap_ci(correct)

    stats = {
        "acc_train": float(((predict(net, mu, sigma, Xtr) > 0.5) == ytr.ravel()).mean()),
        "acc_val": float(correct.mean()),
        "temperature": fit_temperature(net, mu, sigma, Xva, yva),
        "n_train": len(Xtr),
        "n_val": len(Xva),
        "majority_val": float(max(yva.mean(), 1.0 - yva.mean())),
        "ci_val": (ci_lo, ci_hi),
        "grad_err": gradient_check(net, (Xtr - mu) / sigma, ytr),
        "median": median,
        "mu": mu,
        "sigma": sigma,
        "frames": frames,
    }
    return net, stats


def cross_validate(X, y, sizes, folds, epochs, lr, l2, seed):
    """Honest k-fold accuracy; medians and standardisation are refit inside each fold."""
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(X))
    accs = []
    for f in range(folds):
        test = idx[f::folds]
        tr = np.setdiff1d(idx, test)
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xte = np.where(np.isnan(X[test]), med, X[test])
        net, mu, sigma, _ = fit(Xtr, y[tr], np.random.default_rng(seed), sizes,
                                epochs=epochs, lr=lr, l2=l2)
        accs.append(float(((predict(net, mu, sigma, Xte) > 0.5) == y[test].ravel()).mean()))
    return float(np.mean(accs)), float(np.std(accs))


# --------------------------------------------------------------------------- cli

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--features", default="base", choices=sorted(FEATURE_SETS))
    p.add_argument("--target", default=TARGET, choices=["dec", "dec_o", "match"])
    p.add_argument("--gender", type=int, default=None, choices=[0, 1])
    p.add_argument("--min-age", type=float, default=None)
    p.add_argument("--max-age", type=float, default=None)
    p.add_argument("--layers", default="10", help="hidden layer sizes, e.g. 24,12")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--l2", type=float, default=DEFAULTS["l2"])
    p.add_argument("--out", default=os.path.join(HERE, "weights.json"))
    p.add_argument("--snapshots", action="store_true",
                   help="save training frames for animate.py (drawn model only)")
    return p.parse_args()


def main():
    args = parse_args()
    hidden = [int(x) for x in args.layers.replace(" ", "").split(",") if x]
    features = FEATURE_SETS[args.features]
    sizes = [len(features)] + hidden + [1]

    X, y = load_dataset(gender=args.gender, features=features, target=args.target,
                        min_age=args.min_age, max_age=args.max_age)
    if len(X) < 200:
        raise SystemExit(f"only {len(X)} rows -- not enough to train")
    complete = int((~np.isnan(X)).all(axis=1).sum())
    scope = GENDER_LABEL[args.gender]
    if args.min_age is not None or args.max_age is not None:
        scope += f", age {args.min_age:g}-{args.max_age:g}"
    print(f"loaded {len(X):,} rows for {scope}  "
          f"({y.mean():.1%} positive, {complete:,} complete, rest median-filled)")
    print(f"inputs ({len(features)}): {', '.join(features)}")
    print(f"architecture: {' -> '.join(map(str, sizes))}")

    net, stats = train(X, y, sizes, args.epochs, args.lr, args.l2,
                       DEFAULTS["val_fraction"], DEFAULTS["seed"], args.snapshots)

    for f in stats["frames"]:
        if f["epoch"] % 100 == 0 or f["epoch"] == args.epochs:
            print(f"epoch {f['epoch']:4d}  train_loss {f['train_loss']:.4f}  "
                  f"val_loss {f['val_loss']:.4f}  val_acc {f['val_acc']:.1%}")

    cv_mean, cv_std = cross_validate(X, y, sizes, DEFAULTS["cv_folds"],
                                     args.epochs, args.lr, args.l2, DEFAULTS["seed"])
    print(f"\n5-fold CV accuracy {cv_mean:.1%} +/- {cv_std:.1%}  "
          f"({DEFAULTS['cv_folds']} folds, same architecture)")
    print(f"holdout: train {stats['acc_train']:.1%}   val {stats['acc_val']:.1%}   "
          f"[95% CI {stats['ci_val'][0]:.1%}-{stats['ci_val'][1]:.1%}]")
    lift = stats["acc_val"] - stats["majority_val"]
    print(f"majority-class baseline {stats['majority_val']:.1%}   ->   lift {lift:+.1%}")
    verdict = "PASS" if stats["grad_err"] < 1e-5 else "FAIL"
    print(f"gradient check vs finite differences: {stats['grad_err']:.1e} "
          f"(limit 1e-5)  {verdict}")

    p_demo = None
    if features == FEATURES:
        dream = np.array([[10.0] * len(FEATURES)])
        dream[:, FEATURES.index("age")] = 27.0
        dream[:, FEATURES.index("income")] = 69487.0
        dream[:, FEATURES.index("goal")] = 4.0
        p_demo = float(predict(net, stats["mu"], stats["sigma"], dream)[0])
        print(f"a date rated 10/10 on every attribute (age 27) -> {p_demo:.1%}")

    history = [{"epoch": f["epoch"], "train_loss": round(f["train_loss"], 4),
                "val_loss": round(f["val_loss"], 4)}
               for f in stats["frames"]
               if f["epoch"] % 100 == 0 or f["epoch"] == args.epochs]

    payload = {
        "features": features,
        "target": args.target,
        "arch": sizes,
        "layers": [{"W": w.tolist(), "b": b.tolist()} for w, b in zip(net.W, net.b)],
        "mu": stats["mu"].tolist(),
        "sigma": stats["sigma"].tolist(),
        "impute": stats["median"].tolist(),
        "accuracy_train": round(stats["acc_train"], 4),
        "accuracy_val": round(stats["acc_val"], 4),
        "accuracy_val_ci": [round(stats["ci_val"][0], 4), round(stats["ci_val"][1], 4)],
        "accuracy_cv": round(cv_mean, 4),
        "accuracy_cv_std": round(cv_std, 4),
        "majority_val": round(stats["majority_val"], 4),
        "temperature": round(stats["temperature"], 4),
        "grad_check_max_rel_err": float(f"{stats['grad_err']:.3e}"),
        "n_train": stats["n_train"],
        "n_val": stats["n_val"],
        "n_rows": int(len(X)),
        "n_complete": complete,
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "history": history,
        "scope": {"gender": args.gender, "min_age": args.min_age, "max_age": args.max_age},
    }
    if p_demo is not None:
        payload["p_10_out_of_10"] = round(p_demo, 4)
    # The drawings read the flattened single-hidden-layer keys.
    if net.depth == 2:
        payload.update({
            "W1": net.W[0].tolist(), "b1": net.b[0].tolist(),
            "W2": net.W[-1].tolist(), "b2": net.b[-1].tolist(),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")

    if args.snapshots:
        frames = stats["frames"]
        np.savez_compressed(
            SNAP,
            epochs=np.array([f["epoch"] for f in frames]),
            W1=np.array([f["W1"] for f in frames]),
            b1=np.array([f["b1"] for f in frames]),
            W2=np.array([f["W2"] for f in frames]),
            b2=np.array([f["b2"] for f in frames]),
            train_loss=np.array([f["train_loss"] for f in frames]),
            val_loss=np.array([f["val_loss"] for f in frames]),
            val_acc=np.array([f["val_acc"] for f in frames]),
        )
        print(f"wrote {SNAP}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
