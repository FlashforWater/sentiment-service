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
