"""Train a tiny 10 -> 10 -> 1 neural network (pure NumPy, no frameworks).

Dataset: "Speed Dating Experiment" (Kaggle: annavictoria/speed-dating-experiment),
8,378 speed dates from 2002-2004. Each row is one date: how the participant rated
their partner on six attributes, plus who they were, and whether they said yes to a
second date (dec = 0/1).

All 8,378 dates are used: the 3,581 complete rows plus median-imputed columns for
the rest (income alone is blank on half the rows). 300 epochs with L2 1e-3 was
picked by 5-fold CV -- see `cross_validate`.

The network predicts dec from 10 inputs. Layer 1 is 10x10, which is exactly the
shape of the hypergamy meme -> see visualize.py.

Every run checks its own backprop against finite differences, and reports 5-fold CV
accuracy next to the majority-class baseline and a bootstrap interval, so the "it
learned something" claim is measured rather than asserted.

  python train.py                                       # everyone -> weights.json
  python train.py --gender 1 --out weights_men.json     # men's decisions
  python train.py --gender 0 --out weights_women.json   # women's decisions
"""

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "speed_dating.csv")
OUT = os.path.join(HERE, "weights.json")
SNAP = os.path.join(HERE, "training_frames.npz")

# 10 inputs -> the 10 people in the left column of the drawing.
FEATURES = ["age", "income", "goal", "attr", "sinc", "intel", "fun", "amb", "shar", "like"]
TARGET = "dec"
HIDDEN = 10

EPOCHS = 300
LR = 0.08
L2 = 1e-3
VAL_FRACTION = 0.2
SEED = 7
CV_FOLDS = 5
SNAPSHOT_EVERY = 10

GENDER_LABEL = {None: "all participants", 0: "women", 1: "men"}


def load_dataset(gender=None):
    """All rows with a target, NaN where an input is blank.

    `gender` 0 = women, 1 = men in the Kaggle codebook; the drawings rely on it.
    """
    with open(DATA, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    X, y = [], []
    for row in rows:
        try:
            target = float(row[TARGET])
        except (KeyError, TypeError, ValueError):
            continue
        if gender is not None and str(row.get("gender", "")).strip() != str(gender):
            continue
        values = []
        for key in FEATURES:
            raw = (row.get(key) or "").strip()
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(np.nan)
        X.append(values)
        y.append(target)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1, 1)


def standardize(train, other):
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (train - mu) / sigma, (other - mu) / sigma, mu, sigma


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Network:
    def __init__(self, n_in, n_hidden, rng):
        # Xavier-ish init: without it, 10 tanh units just die.
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / (n_in + n_hidden)), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / (n_hidden + 1)), (n_hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        p = sigmoid(h @ self.W2 + self.b2)
        return h, p

    def loss(self, X, y):
        p = np.clip(self.forward(X)[1], 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def gradients(self, X, y):
        h, p = self.forward(X)
        n = X.shape[0]

        d_out = (p - y) / n                      # dBCE/dlogit
        gW2 = h.T @ d_out
        gb2 = d_out.sum(axis=0)

        d_h = (d_out @ self.W2.T) * (1.0 - h ** 2)   # tanh'
        gW1 = X.T @ d_h
        gb1 = d_h.sum(axis=0)

        return gW1, gb1, gW2, gb2


def gradient_check(net, X, y, rows=6, eps=1e-4):
    """Worst error between backprop and central finite differences, normalised by
    the largest gradient in the network.

    This is the honesty test: if the hand-written gradients were wrong, training
    would still limp along and the picture would still look fine, but this number
    would not be ~1e-8. Per-element relative error is misleading here -- after
    training, most tanh units sit at |a| > 0.999, their gradients are ~1e-11 and
    finite differences drown in floating-point noise -- so we normalise by the
    biggest gradient instead.
    """
    X, y = X[:rows], y[:rows]
    analytic = net.gradients(X, y)
    worst_abs = 0.0
    largest = 0.0
    for param, grad in zip([net.W1, net.b1, net.W2, net.b2], analytic):
        for idx in np.ndindex(param.shape):
            keep = param[idx]
            param[idx] = keep + eps
            loss_plus = net.loss(X, y)
            param[idx] = keep - eps
            loss_minus = net.loss(X, y)
            param[idx] = keep
            numeric = (loss_plus - loss_minus) / (2.0 * eps)
            worst_abs = max(worst_abs, abs(numeric - grad[idx]))
            largest = max(largest, abs(grad[idx]))
    return worst_abs / max(largest, 1e-12)


def bootstrap_ci(correct, iters=2000, seed=SEED + 100):
    """95% percentile interval for accuracy on the held-out split."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    draws = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(iters)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def predict(net, mu, sigma, X):
    return net.forward((X - mu) / sigma)[1].ravel()


def fit(Xtr, ytr, rng, epochs=EPOCHS, Xva=None, yva=None, record=False):
    """Train on already-imputed raw inputs. Returns (net, mu, sigma, frames)."""
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (Xtr - mu) / sigma
    Xv = None if Xva is None else (Xva - mu) / sigma

    net = Network(Xtr.shape[1], HIDDEN, rng)
    params = [net.W1, net.b1, net.W2, net.b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    frames = []

    for epoch in range(1, epochs + 1):
        gW1, gb1, gW2, gb2 = net.gradients(Xs, ytr)
        grads = [gW1 + L2 * net.W1, gb1, gW2 + L2 * net.W2, gb2]
        for i, (p, gi) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * gi
            v[i] = beta2 * v[i] + (1 - beta2) * gi ** 2
            m_hat = m[i] / (1 - beta1 ** epoch)
            v_hat = v[i] / (1 - beta2 ** epoch)
            p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        if record and (epoch % SNAPSHOT_EVERY == 0 or epoch == epochs):
            frames.append(
                {
                    "epoch": epoch,
                    "W1": net.W1.copy(),
                    "b1": net.b1.copy(),
                    "W2": net.W2.ravel().copy(),
                    "b2": net.b2.copy(),
                    "train_loss": net.loss(Xs, ytr),
                    "val_loss": net.loss(Xv, yva),
                    "val_acc": float(((net.forward(Xv)[1] > 0.5) == yva).mean()),
                }
            )

    return net, mu, sigma, frames


def train(X, y):
    rng = np.random.default_rng(SEED)

    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - VAL_FRACTION))
    Xtr_raw, Xva_raw = X[idx[:cut]], X[idx[cut:]]
    ytr, yva = y[idx[:cut]], y[idx[cut:]]

    # Medians come from the training split only; income is blank on ~half the rows.
    median = np.nanmedian(Xtr_raw, axis=0)
    Xtr = np.where(np.isnan(Xtr_raw), median, Xtr_raw)
    Xva = np.where(np.isnan(Xva_raw), median, Xva_raw)

    net, mu, sigma, frames = fit(Xtr, ytr, rng, Xva=Xva, yva=yva, record=True)

    val_correct = ((predict(net, mu, sigma, Xva) > 0.5) == yva.ravel()).astype(float)
    ci_lo, ci_hi = bootstrap_ci(val_correct)

    stats = {
        "acc_train": float(((predict(net, mu, sigma, Xtr) > 0.5) == ytr.ravel()).mean()),
        "acc_val": float(val_correct.mean()),
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


def cross_validate(X, y, folds=CV_FOLDS):
    """Honest 5-fold accuracy; medians/standardisation refit inside each fold."""
    rng = np.random.default_rng(SEED + 1)
    idx = rng.permutation(len(X))
    accs = []
    for f in range(folds):
        test = idx[f::folds]
        tr = np.setdiff1d(idx, test)
        med = np.nanmedian(X[tr], axis=0)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xte = np.where(np.isnan(X[test]), med, X[test])
        net, mu, sigma, _ = fit(Xtr, y[tr], np.random.default_rng(SEED))
        pred = predict(net, mu, sigma, Xte)
        accs.append(float(((pred > 0.5) == y[test].ravel()).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    args = sys.argv[1:]
    gender = int(args[args.index("--gender") + 1]) if "--gender" in args else None
    out = args[args.index("--out") + 1] if "--out" in args else OUT

    X, y = load_dataset(gender)
    complete = int((~np.isnan(X)).all(axis=1).sum())
    print(
        f"loaded {len(X):,} dates for {GENDER_LABEL[gender]}  "
        f"({y.mean():.1%} said yes, {complete:,} complete, rest median-filled)"
    )
    print("inputs:", ", ".join(FEATURES))

    net, stats = train(X, y)

    for f in stats["frames"]:
        if f["epoch"] % 100 == 0 or f["epoch"] == EPOCHS:
            print(
                f"epoch {f['epoch']:4d}  train_loss {f['train_loss']:.4f}  "
                f"val_loss {f['val_loss']:.4f}  val_acc {f['val_acc']:.1%}"
            )

    cv_mean, cv_std = cross_validate(X, y)

    print(
        f"\n5-fold CV accuracy {cv_mean:.1%} +/- {cv_std:.1%}   "
        f"({CV_FOLDS} folds, same architecture, L2 {L2:g}, {EPOCHS} epochs)"
    )
    print(
        f"holdout: train {stats['acc_train']:.1%}   val {stats['acc_val']:.1%}   "
        f"[95% CI {stats['ci_val'][0]:.1%}-{stats['ci_val'][1]:.1%}]"
    )
    lift = stats["acc_val"] - stats["majority_val"]
    print(f"majority-class baseline {stats['majority_val']:.1%}   ->   lift {lift:+.1%}")
    verdict = "PASS" if stats["grad_err"] < 1e-5 else "FAIL"
    print(f"gradient check vs finite differences: error {stats['grad_err']:.1e} (limit 1e-5)  {verdict}")

    # Meme demo: a plausible date rated 10/10 on every attribute.
    dream = np.array([[10.0] * len(FEATURES)])
    dream[:, FEATURES.index("age")] = 27.0
    dream[:, FEATURES.index("income")] = 69487.0   # dollars, not a 1-10 score
    dream[:, FEATURES.index("goal")] = 4.0         # 4 = looking for a relationship
    p = float(predict(net, stats["mu"], stats["sigma"], dream)[0])
    print(f"a date rated 10/10 on every attribute (age 27) -> P(second date) = {p:.1%}")

    history = [
        {"epoch": f["epoch"], "train_loss": round(f["train_loss"], 4), "val_loss": round(f["val_loss"], 4)}
        for f in stats["frames"]
        if f["epoch"] % 100 == 0 or f["epoch"] == EPOCHS
    ]

    payload = {
        "features": FEATURES,
        "target": TARGET,
        "W1": net.W1.tolist(),
        "b1": net.b1.tolist(),
        "W2": net.W2.tolist(),
        "b2": net.b2.tolist(),
        "mu": stats["mu"].tolist(),
        "sigma": stats["sigma"].tolist(),
        "impute": stats["median"].tolist(),
        "accuracy_train": round(stats["acc_train"], 4),
        "accuracy_val": round(stats["acc_val"], 4),
        "accuracy_val_ci": [round(stats["ci_val"][0], 4), round(stats["ci_val"][1], 4)],
        "accuracy_cv": round(cv_mean, 4),
        "accuracy_cv_std": round(cv_std, 4),
        "majority_val": round(stats["majority_val"], 4),
        "grad_check_max_rel_err": float(f"{stats['grad_err']:.3e}"),
        "p_10_out_of_10": round(p, 4),
        "n_train": stats["n_train"],
        "n_val": stats["n_val"],
        "n_rows": int(len(X)),
        "n_complete": complete,
        "epochs": EPOCHS,
        "l2": L2,
        "history": history,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")

    if gender is None:
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
