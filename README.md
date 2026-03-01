# Data

## Overview

This project uses the Brown Corpus (downloaded via `kagglehub`) to train word embeddings.
Class balance visualization:

![Label distribution](dataset/plots/label_distribution.png)

## Tokens

| Item                             | Value   |
| -------------------------------- | ------- |
| Vocabulary size used in training | 3000    |
| Unknown token                    | `<UNK>` |

Token diagnostics:

![Top tokens](dataset/plots/top_tokens.png)
![UNK ratio](dataset/plots/unk_ratio.png)

# Model

Both models use an 80/10/10 train/val/test split.
Split plot:

![Train/Val/Test split sizes](dataset/plots/split_sizes.png)

## Continuous Bag of Words (CBOW)

CBOW predicts the center word from surrounding context words.
Training uses cross-entropy over the softmax output.

### Command

```bash
cd src
uv run python main.py --train cbow
```

### Loss plot

![CBOW loss history](src/cbow_loss_history_plot.png)

### Validation/Test mini table

| Metric      | Value             |
| ----------- | ----------------- |
| `val_loss`  | from training log |
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

![SGNS loss history](src/sgns_loss_history_plot.png)

### Validation/Test mini table

| Metric      | Value             |
| ----------- | ----------------- |
| `val_loss`  | from training log |
| `test_loss` | from training log |
