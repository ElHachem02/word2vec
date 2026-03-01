# Data

## Overview

This project uses the Brown Corpus (downloaded via `kagglehub`) to train word embeddings.
Class balance visualization: [dataset/plots/label_distribution.png](dataset/plots/label_distribution.png)

## Tokens

| Item | Value |
|---|---|
| Vocabulary size used in training | 3000 |
| Unknown token | `<UNK>` |

Token diagnostics:
- [dataset/plots/top_tokens.png](dataset/plots/top_tokens.png)
- [dataset/plots/unk_ratio.png](dataset/plots/unk_ratio.png)

# Model

Both models use an 80/10/10 train/val/test split.
Split plot: [dataset/plots/split_sizes.png](dataset/plots/split_sizes.png)

## Continuous Bag of Words (CBOW)

CBOW predicts the center word from surrounding context words.
Training uses cross-entropy over the softmax output.

### Command

```bash
cd src
uv run python main.py --train cbow
```

### Loss plot

[src/cbow_loss_history_plot.png](src/cbow_loss_history_plot.png)

### Validation/Test mini table

| Metric | Value |
|---|---|
| `val_loss` | from training log |
| `test_loss` | from training log |

## Skip-Gram with Negative Sampling (SGNS)

SGNS predicts context words from a center word using sampled negatives.
Training uses logistic loss for positive and negative pairs.

### Command

```bash
cd src
uv run python main.py --train sgns
```

### Loss plot

[src/sgns_loss_history_plot.png](src/sgns_loss_history_plot.png)

### Validation/Test mini table

| Metric | Value |
|---|---|
| `val_loss` | from training log |
| `test_loss` | from training log |
