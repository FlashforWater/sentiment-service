# Experiment Notes

## Setup
- Dataset: IMDB (2000 train, 500 eval, seed=42)
- Model: distilbert-base-uncased (66M params)
- Hardware: Mac M-series, MPS

## Run history

### run1 (baseline, 1 epoch)
- lr: 2e-5, batch: 16, epochs: 1
- eval_accuracy: 0.828
- eval_loss: 0.42
- training time: ~40s

### run-3epoch (overfitting demo)
- Same config as run1 but epochs=3
- eval_accuracy: 0.816 (worse than 1 epoch!)
- eval_loss by epoch: ~0.42 → ~0.44 → 0.47 (rebounding)
- train_loss: 0.55 → 0.27 → 0.18 (kept dropping)
- **Classic overfitting**: model memorizing the 2000 samples

## Lessons learned

1. With only 2000 samples and a 66M-param model, 1 epoch is the sweet spot.
2. Training loss dropping ≠ model improving. Must look at eval metrics.
3. eval_loss going up while train_loss going down is the textbook overfitting signal.

## Next experiments to try
- [ ] Increase train_size to 5000 — does overfitting still happen at 3 epochs?
- [ ] Try lr 5e-5 with 1 epoch — is lr the bottleneck?
- [ ] Add EarlyStoppingCallback as production safeguard

---

## Stage 4: Comparative Experiments

### Setup
Same baseline as stage 3 (IMDB, DistilBERT, 1 epoch, batch 16),
varying one factor at a time.

### Results table

| Run | Change | train_size | lr | eval_acc | eval_loss | time |
|---|---|---|---|---|---|---|
| run1 | baseline | 2000 | 2e-5 | 0.828 | 0.434 | ~40s |
| run-lr5e5 | lr x2.5 | 2000 | 5e-5 | 0.816 | 0.396 | ~40s |
| run-data5000 | data x2.5 | 5000 | 2e-5 | **0.838** | **0.357** | ~110s |

### Key findings

1. **More data > tuning lr.**
   With the same 2.5x scale-up, increasing data gained +1.0% acc,
   while increasing lr lost -1.2% acc.

2. **Higher lr changes training dynamics, not just speed.**
   lr=5e-5 caused a clear plateau around step 5-15 — the model
   bounced near the optimum instead of converging smoothly.
   Final train_loss was lowest among the three, but eval_acc was
   worst. Classic example of train/eval indicator mismatch.

3. **Training time scales linearly with data.**
   2.5x data → 2.75x time. Useful for capacity planning.

### Industry rule of thumb confirmed

For fine-tuning tasks, the priority is roughly:
  data quantity > data quality > model size > hyperparameters

I now have first-hand evidence of this from my own experiments,
not just from blog posts.

### Next experiments (deferred)
- [ ] train_size 10000 — does the gain continue?
- [ ] Full dataset (25000) — what's the realistic ceiling?
- [ ] data5000 + 2 epoch — does more data unlock more epochs?
