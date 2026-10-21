# Stale replica reads: threshold rule vs class-balanced LightGBM

Motivation: "How replicated data management in the cloud can benefit from a data grid protocol: the re:GRIDiT approach" (2009) argues that replicated cloud stores need explicit, auditable decisions about which replica serves a read and how stale it may be. This experiment asks whether a learned classifier can make those decisions more accurately than a simple replica-lag rule, and whether such decisions can be audited against a known ground truth.

Question: Can a class-balanced, CV-tuned LightGBM predict stale replica reads more accurately than a standard replica-lag threshold rule, using the same seeded simulated replicated store?

Dataset: a seeded simulated replicated store generated inside analysis.py (Poisson writes, per-replica lognormal propagation lags, random reads). This is not the paper's own dataset; the paper reports no measurable model.

Baseline: a threshold rule that flags a read as stale when the chosen replica's time since last sync exceeds a threshold tuned by cross-validation on the training split.

Improved model: a class-balanced LightGBM on per-read features (sync age, pending-update count, replica load, recent write rate, replica id), with tree size and decision threshold chosen by cross-validation on the training split. The test split is used once, for both models.

Main metric: stale-class F1 (higher is better). Secondary: ROC-AUC and false-positive rate.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to results.json (and figures/f1_comparison.png).

## Results (latest run)

In my run, on a seeded simulated replicated store (50,000 held-out test reads), stale-class F1 was 0.971 for a standard replica-lag threshold rule versus 0.996 for a class-balanced, CV-tuned LightGBM, a small gain of about 2.5 points from a single simulated dataset draw.

| Metric | Value |
|---|---|
| baseline_f1 | 0.9711 |
| baseline_f1_std | 0 |
| baseline_roc_auc | 0.7979 |
| baseline_roc_auc_std | 1.11e-16 |
| baseline_fpr | 0.8196 |
| baseline_fpr_std | 0 |
| improved_f1 | 0.9959 |
| improved_f1_std | 0 |
| improved_roc_auc | 0.984 |
| improved_roc_auc_std | 0 |
| improved_fpr | 0.2415 |
| improved_fpr_std | 0 |
| baseline_score | 0.9711 |
| improved_score | 0.9959 |
| baseline_score_std | 0 |
| improved_score_std | 0 |
| improvement | 0.02485 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulation is a stand-in, not a re-implementation of re:GRIDiT, and the paper reports no measurable model. Labels and features come from the simulator; pending updates are a replica-side queue count, so the task is not trivially solved. The split is time-ordered as the method specifies rather than stratified. Std is over model/CV seeds only; the simulated dataset is a single draw with a fixed seed.

## Reference

This project is motivated by the following paper:

> LC Voicu, H Schuldt (2009). *How replicated data management in the cloud can benefit from a data grid protocol: the re: gridit approach*. Proceedings of the first international workshop on. https://dl.acm.org/doi/abs/10.1145/1651263.1651272

Paper author: Prof. Heiko Schuldt, University of Basel (heiko.schuldt@unibas.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
