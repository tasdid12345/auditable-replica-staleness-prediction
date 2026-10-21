"""Can a class-balanced, CV-tuned LightGBM predict stale replica reads better than a replica-lag threshold rule?
Seeded simulation of a replicated store; time-ordered train/test split; tuning by CV on training data only."""
import json, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
MODEL_SEEDS = [42, 43, 44]     # both models are run with each seed
N_READS = 200_000              # simulated reads (generated directly, no subsampling needed)
HORIZON = 1000.0               # simulated seconds
WRITE_RATE = 20.0              # Poisson write rate (writes per second)
N_REP = 3                      # replicas in the store
TEST_FRAC = 0.25
FEATURES = ["sync_age", "pending", "replica_load", "write_rate", "replica_id"]
LGB_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 31) for mc in (20, 100)]
PROB_GRID = np.linspace(0.05, 0.95, 19)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def simulate(rng):
    """Return features X, stale labels y and read times t (already sorted by time)."""
    w = np.sort(rng.uniform(0, HORIZON, rng.poisson(WRITE_RATE * HORIZON)))  # Poisson writes
    base = rng.uniform(0.3, 0.7, N_REP)       # per-replica base load
    phase = rng.uniform(0, 2 * np.pi, N_REP)  # per-replica load phase

    def load(r, t):  # slowly varying replica load in [0, 1]
        return np.clip(base[r] + 0.25 * np.sin(2 * np.pi * t / 300 + phase[r]), 0, 1)

    # per-replica delivery (d) and apply (a) times; each replica processes writes in order
    rep_d, rep_a = [], []
    for r in range(N_REP):
        d = np.maximum.accumulate(w + rng.lognormal(np.log(0.01), 1.0, len(w)))
        a = np.maximum.accumulate(d + rng.lognormal(np.log(0.05) + 2.0 * load(r, w), 0.8))
        rep_d.append(d)
        rep_a.append(a)
    t = np.sort(rng.uniform(0, HORIZON, N_READS))   # read times
    rep = rng.integers(0, N_REP, N_READS)           # replica chosen for each read
    latest = np.searchsorted(w, t, "right")         # writes committed by time t
    X = np.zeros((N_READS, len(FEATURES)))
    applied = np.zeros(N_READS)
    for r in range(N_REP):
        m = rep == r
        tr = t[m]
        n_app = np.searchsorted(rep_a[r], tr, "right")  # writes applied by replica r
        n_rec = np.searchsorted(rep_d[r], tr, "right")  # writes received by replica r
        last = np.where(n_app > 0, rep_a[r][np.maximum(n_app - 1, 0)], 0.0)
        X[m, 0] = tr - last                # sync age: time since last applied update
        X[m, 1] = n_rec - n_app            # pending: received but not yet applied
        X[m, 2] = load(r, tr)              # replica load
        applied[m] = n_app
    X[:, 3] = np.searchsorted(w, t, "right") - np.searchsorted(w, t - 1.0, "right")  # writes in last 1 s
    X[:, 4] = rep                          # replica id
    y = (applied < latest).astype(int)     # stale: chosen replica lacks latest committed write
    return X, y, t


def f1_at(y, pred):
    return f1_score(y, pred, pos_label=1, zero_division=0)


def fit_baseline(X, y, seed):
    """Threshold rule: stale if sync_age > thr; thr chosen by stratified CV F1 on training data."""
    grid = np.unique(np.quantile(X[:, 0], np.linspace(0.05, 0.95, 37)))
    scores = np.zeros(len(grid))
    for _, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        for k, thr in enumerate(grid):
            scores[k] += f1_at(y[va], X[va, 0] > thr)
    return grid[np.argmax(scores)]


def lgb_model(params, seed):
    # class_weight="balanced" handles the stale/fresh imbalance
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                              random_state=seed, verbose=-1, **params)


def fit_improved(X, y, seed):
    """Class-balanced LightGBM; leaves/min-child-samples and decision threshold chosen by out-of-fold CV F1."""
    best = (-1.0, None, None)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    for params in LGB_GRID:
        oof = np.zeros(len(y))
        for tr, va in folds:
            oof[va] = lgb_model(params, seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        for thr in PROB_GRID:
            f = f1_at(y, oof > thr)
            if f > best[0]:
                best = (f, params, thr)
    _, params, thr = best
    return lgb_model(params, seed).fit(X, y), thr


def evaluate(y, pred, score):
    """Shared evaluation: stale-class F1 (main), ROC-AUC and false-positive rate (secondary)."""
    fp = np.sum((pred == 1) & (y == 0))
    tn = np.sum((pred == 0) & (y == 0))
    return dict(f1=f1_at(y, pred), roc_auc=roc_auc_score(y, score), fpr=fp / (fp + tn))


def main():
    log("simulating replicated store")
    X, y, _ = simulate(np.random.default_rng(SEED))
    n_train = int(N_READS * (1 - TEST_FRAC))
    # time-ordered split, fixed before any model is fit
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    log(f"reads={N_READS} train={len(ytr)} test={len(yte)} "
        f"stale_rate train={ytr.mean():.3f} test={yte.mean():.3f}")
    if ytr.sum() == 0 or yte.sum() == 0 or ytr.sum() == len(ytr):
        log("ERROR: degenerate labels, cannot produce meaningful metrics")
        sys.exit(1)

    rows = {"baseline": [], "improved": []}
    for s in MODEL_SEEDS:
        log(f"seed {s}: tuning baseline threshold by CV")
        thr = fit_baseline(Xtr, ytr, s)
        rows["baseline"].append(evaluate(yte, (Xte[:, 0] > thr).astype(int), Xte[:, 0]))
        log(f"seed {s}: tuning LightGBM by CV")
        model, pthr = fit_improved(Xtr, ytr, s)
        p = model.predict_proba(Xte)[:, 1]
        rows["improved"].append(evaluate(yte, (p > pthr).astype(int), p))
        log(f"seed {s}: test F1 baseline={rows['baseline'][-1]['f1']:.4f} "
            f"improved={rows['improved'][-1]['f1']:.4f}")

    metrics = {}
    for name, rs in rows.items():
        for k in ("f1", "roc_auc", "fpr"):
            v = np.array([r[k] for r in rs], dtype=float)
            metrics[f"{name}_{k}"] = float(v.mean())
            metrics[f"{name}_{k}_std"] = float(v.std())
    if not all(np.isfinite(list(metrics.values()))):
        log("ERROR: non-finite metrics")
        sys.exit(1)
    metrics["baseline_score"] = metrics["baseline_f1"]
    metrics["improved_score"] = metrics["improved_f1"]
    metrics["baseline_score_std"] = metrics["baseline_f1_std"]
    metrics["improved_score_std"] = metrics["improved_f1_std"]
    metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]

    results = {
        "question": "Can a class-balanced, CV-tuned LightGBM predict stale replica reads more accurately "
                    "than a standard replica-lag threshold rule, using the same seeded simulated replicated store?",
        "method": "Seeded simulation of a replicated store with Poisson writes, per-replica lognormal "
                  "propagation lags and random reads; reads labelled stale when the chosen replica lacks the "
                  "latest write. Time-ordered train/test split fixed before fitting; tuning only by CV on training data; "
                  "test set used once. Both models run with seeds 42, 43, 44.",
        "dataset": {
            "name": "Simulated replicated store (generated in analysis.py)",
            "source_url": "none: generated in-script",
            "n_rows": N_READS, "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "stale-class F1 on held-out test reads",
        "higher_is_better": True,
        "baseline": "Standard replica-lag threshold rule: flag a read as stale when the replica's time since "
                    "last sync exceeds a threshold chosen by stratified CV F1 on the training set.",
        "improved": "Class-balanced LightGBM on per-read features (sync age, pending-update count, replica load, "
                    "recent write rate, replica id); leaves and min child samples chosen by CV, and the decision "
                    "threshold chosen on out-of-fold probabilities by CV F1 on the training set.",
        "metrics": metrics,
        "notes": "Simulation is a stand-in, not a re-implementation of re:GRIDiT, and the paper reports no "
                 "measurable model. Labels and features come from the simulator; pending updates are a "
                 "replica-side queue count, so the task is not trivially solved. The split is time-ordered "
                 "as the method specifies rather than stratified. Std is over model/CV seeds only; the "
                 "simulated dataset is a single draw with a fixed seed.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("wrote results.json")

    # optional chart: baseline vs improved stale-class F1 on the test set
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline", "improved"], [metrics["baseline_f1"], metrics["improved_f1"]],
           yerr=[metrics["baseline_f1_std"], metrics["improved_f1_std"]], capsize=6,
           color=["grey", "steelblue"])
    ax.set_ylabel("stale-class F1 (test)")
    ax.set_title("Stale-read detection")
    fig.tight_layout()
    fig.savefig("figures/f1_comparison.png", dpi=120)
    log("done")


if __name__ == "__main__":
    main()
