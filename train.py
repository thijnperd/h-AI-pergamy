"""Train a tiny 10 -> 10 -> 1 neural network (pure NumPy, no frameworks).

Dataset: "Speed Dating Experiment" (Kaggle: annavictoria/speed-dating-experiment),
a 428 KB CSV of 8,378 speed dates from 2002-2004. Each row is one date: how the
participant rated their partner on six attributes, plus who they were, and
whether they said yes to a second date (dec = 0/1).

The network predicts dec from 10 inputs. Layer 1 is 10x10, which is exactly the
shape of the hypergamy meme -> see visualize.py.

Every run checks its own backprop against finite differences, and reports the
validation accuracy next to the majority-class baseline and a bootstrap interval,
so the "it learned something" claim is measured rather than asserted.
"""

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "speed_dating.csv")
OUT = os.path.join(HERE, "weights.json")

# 10 inputs -> the 10 people in the left column of the drawing.
FEATURES = ["age", "income", "goal", "attr", "sinc", "intel", "fun", "amb", "shar", "like"]
TARGET = "dec"
HIDDEN = 10

EPOCHS = 900
LR = 0.08
VAL_FRACTION = 0.2
SEED = 7


def load_dataset():
    with open(DATA, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    X, y = [], []
    for row in rows:
        try:
            X.append([float(row[k]) for k in FEATURES])
            y.append(float(row[TARGET]))
        except (KeyError, TypeError, ValueError):
            continue  # skip rows with blanks

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1, 1)


def standardize(train, val):
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (train - mu) / sigma, (val - mu) / sigma, mu, sigma


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


def train(X, y):
    rng = np.random.default_rng(SEED)

    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - VAL_FRACTION))
    Xtr, Xva = X[idx[:cut]], X[idx[cut:]]
    ytr, yva = y[idx[:cut]], y[idx[cut:]]

    Xtr, Xva, mu, sigma = standardize(Xtr, Xva)
    net = Network(Xtr.shape[1], HIDDEN, rng)

    params = [net.W1, net.b1, net.W2, net.b2]
    grads = [np.zeros_like(p) for p in params]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = []
    for epoch in range(1, EPOCHS + 1):
        g = net.gradients(Xtr, ytr)
        for i, (p, gi) in enumerate(zip(params, g)):
            grads[i] = gi
            m[i] = beta1 * m[i] + (1 - beta1) * gi
            v[i] = beta2 * v[i] + (1 - beta2) * gi ** 2
            m_hat = m[i] / (1 - beta1 ** epoch)
            v_hat = v[i] / (1 - beta2 ** epoch)
            p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        if epoch % 100 == 0 or epoch == 1:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(net.loss(Xtr, ytr), 4),
                    "val_loss": round(net.loss(Xva, yva), 4),
                }
            )
            print(f"epoch {epoch:4d}  train_loss {history[-1]['train_loss']:.4f}  val_loss {history[-1]['val_loss']:.4f}")

    def accuracy(Xs, ys):
        return float(((net.forward(Xs)[1] > 0.5) == ys).mean())

    val_correct = ((net.forward(Xva)[1] > 0.5) == yva).ravel().astype(float)
    ci_lo, ci_hi = bootstrap_ci(val_correct)

    stats = {
        "acc_train": accuracy(Xtr, ytr),
        "acc_val": accuracy(Xva, yva),
        "n_train": len(Xtr),
        "n_val": len(Xva),
        "majority_val": float(max(yva.mean(), 1.0 - yva.mean())),
        "ci_val": (ci_lo, ci_hi),
        "grad_err": gradient_check(net, Xtr, ytr),
    }
    return net, mu, sigma, stats, history


def main():
    X, y = load_dataset()
    print(f"loaded {len(X)} dates  ({y.mean():.1%} said yes)")
    print("inputs:", ", ".join(FEATURES))

    net, mu, sigma, stats, history = train(X, y)

    print(
        f"\ntrain accuracy {stats['acc_train']:.1%}   "
        f"val accuracy {stats['acc_val']:.1%}   "
        f"[95% CI {stats['ci_val'][0]:.1%}-{stats['ci_val'][1]:.1%}]"
    )
    lift = stats["acc_val"] - stats["majority_val"]
    print(f"majority-class baseline {stats['majority_val']:.1%}   ->   lift {lift:+.1%}")
    verdict = "PASS" if stats["grad_err"] < 1e-5 else "FAIL"
    print(f"gradient check vs finite differences: error {stats['grad_err']:.1e} (limit 1e-5)  {verdict}")

    # Meme demo: what does the net think of a date rated 10/10 on everything?
    dream = np.array([[10.0] * len(FEATURES)])
    dream[:, FEATURES.index("income")] = 69487.0  # income is dollars, not a 1-10 score
    p = float(net.forward((dream - mu) / sigma)[1][0, 0])
    print(f"a date rated 10/10 on everything -> P(second date) = {p:.1%}")

    payload = {
        "features": FEATURES,
        "target": TARGET,
        "W1": net.W1.tolist(),
        "b1": net.b1.tolist(),
        "W2": net.W2.tolist(),
        "b2": net.b2.tolist(),
        "accuracy_train": round(stats["acc_train"], 4),
        "accuracy_val": round(stats["acc_val"], 4),
        "accuracy_val_ci": [round(stats["ci_val"][0], 4), round(stats["ci_val"][1], 4)],
        "majority_val": round(stats["majority_val"], 4),
        "grad_check_max_rel_err": float(f"{stats['grad_err']:.3e}"),
        "p_10_out_of_10": round(p, 4),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "n_train": stats["n_train"],
        "n_val": stats["n_val"],
        "n_rows": int(len(X)),
        "epochs": EPOCHS,
        "history": history,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
