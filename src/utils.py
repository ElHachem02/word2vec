import numpy as np
import os
import re
import csv
from collections import Counter
from pathlib import Path

UNK_TOKEN = "<UNK>"


# Auxiliary functions for training the model
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))  # Numerical stability
    return e_x / e_x.sum(axis=0)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically-stable sigmoid."""
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


# Auxiliary functions for loading and preprocessing the data
def load_corpus_from_csv(
    data_path: str,
    max_rows: int | None = None,
) -> tuple[list[list[str]], list[str], dict[str, float]]:
    """
    Load corpus from brown.csv as list of documents (each document = list of words).

    Reads from the top-level CSV `brown.csv` (columns: filename, para_id, sent_id,
    raw_text, tokenized_text, tokenized_pos, label). Uses only `tokenized_text`:
    whitespace-split, lowercased, non-letters stripped.
    """
    if os.path.isdir(data_path):
        csv_path = os.path.join(data_path, "brown.csv")
    else:
        csv_path = data_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find brown.csv at {csv_path}")

    result: list[list[str]] = []
    labels: list[str] = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            tokenized_text = row.get("tokenized_text") or ""
            labels.append(row.get("label", ""))
            words: list[str] = []
            for tok in tokenized_text.split():
                tok = tok.lower()
                tok = re.sub(r"[^a-z]", "", tok)
                if tok:
                    words.append(tok)
            result.append(words)

    # Transform labels into a dict "label: ratio in total samples"
    label_counts = Counter(labels)
    if len(labels) == 0:
        label_dist: dict[str, float] = {}
    else:
        label_dist = {
            label: count_label / len(labels)
            for label, count_label in label_counts.items()
        }

    return result, labels, label_dist


def build_vocab(train_words: list[str], vocab_size: int) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build vocabulary from training words. Includes UNK for OOV.
    Vocab: [UNK, top (vocab_size-1) most common words]
    """
    counts = Counter(train_words)
    common_words = [UNK_TOKEN] + [w for w, _ in counts.most_common(vocab_size - 1)]
    word_to_id = {w: i for i, w in enumerate(common_words)}
    id_to_word = {i: w for i, w in enumerate(common_words)}
    return word_to_id, id_to_word


def tokenize(words: list[str], word_to_id: dict[str, int]) -> list[int]:
    """
    Convert words to IDs. OOV words map to UNK, preserving sequence and adjacency.
    """
    unk_id = word_to_id[UNK_TOKEN]
    return [word_to_id.get(w, unk_id) for w in words]


def split_train_val_test(
    documents: list[list[str]],
    labels: list[str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """
    Split documents (list of word lists) into train, val, test.
    Returns (train_docs, val_docs, test_docs).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    if labels is None:
        indices = rng.permutation(len(documents))
        n = len(documents)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files = [documents[i] for i in indices[:n_train]]
        val_files = [
            documents[i] for i in indices[n_train: n_train + n_val]
        ]
        test_files = [documents[i] for i in indices[n_train + n_val:]]
        return train_files, val_files, test_files

    if len(labels) != len(documents):
        raise ValueError("labels length must match documents length")

    label_to_indices: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label_indices in label_to_indices.values():
        label_indices_arr = np.array(label_indices, dtype=np.int64)
        rng.shuffle(label_indices_arr)

        n_label = len(label_indices_arr)
        n_train = int(n_label * train_ratio)
        n_val = int(n_label * val_ratio)

        train_indices.extend(label_indices_arr[:n_train].tolist())
        val_indices.extend(
            label_indices_arr[n_train: n_train + n_val].tolist()
        )
        test_indices.extend(label_indices_arr[n_train + n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_files = [documents[i] for i in train_indices]
    val_files = [documents[i] for i in val_indices]
    test_files = [documents[i] for i in test_indices]

    return train_files, val_files, test_files


def load_csv_train_val_test(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_rows: int | None = None,
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """
    Load brown.csv and split into train/val/test in one step.
    Returns (train_docs, val_docs, test_docs); no intermediate document list needed.
    """
    documents, labels, _ = load_corpus_from_csv(
        data_path,
        max_rows=max_rows,
    )

    return split_train_val_test(
        documents,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def load_csv_train_val_test_with_label_dist(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_rows: int | None = None,
) -> tuple[
    list[list[str]],
    list[list[str]],
    list[list[str]],
    dict[str, dict[str, float]],
]:
    """Load, stratify split, and return label distributions."""
    documents, labels, overall_label_dist = load_corpus_from_csv(
        data_path,
        max_rows=max_rows,
    )

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    if len(labels) != len(documents):
        raise ValueError("labels length must match documents length")

    rng = np.random.default_rng(seed)
    label_to_indices: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label_indices in label_to_indices.values():
        label_indices_arr = np.array(label_indices, dtype=np.int64)
        rng.shuffle(label_indices_arr)

        n_label = len(label_indices_arr)
        n_train = int(n_label * train_ratio)
        n_val = int(n_label * val_ratio)

        train_indices.extend(label_indices_arr[:n_train].tolist())
        val_indices.extend(
            label_indices_arr[n_train: n_train + n_val].tolist()
        )
        test_indices.extend(label_indices_arr[n_train + n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_files = [documents[i] for i in train_indices]
    val_files = [documents[i] for i in val_indices]
    test_files = [documents[i] for i in test_indices]

    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    train_label_dist = {
        label: count / len(train_labels)
        for label, count in train_counts.items()
    } if train_labels else {}
    val_label_dist = {
        label: count / len(val_labels)
        for label, count in val_counts.items()
    } if val_labels else {}
    test_label_dist = {
        label: count / len(test_labels)
        for label, count in test_counts.items()
    } if test_labels else {}

    label_distributions = {
        "overall": overall_label_dist,
        "train": train_label_dist,
        "val": val_label_dist,
        "test": test_label_dist,
    }

    return train_files, val_files, test_files, label_distributions


def load_and_preprocess(data_path: str, max_rows: int = 100) -> list[list[str]]:
    """Legacy: load from CSV and return flat list of words."""
    documents, _, _ = load_corpus_from_csv(data_path, max_rows=max_rows)
    return documents


def plot_loss_history(
    loss_history_path: str | Path,
    out_path: str | Path | None = None,
) -> str:
    """
    Plot tracked training loss history saved from main training loop.

    Expects an `.npz` file with keys:
    - `across_epochs`: 1D array of average loss per epoch
    - `within_epoch`: object array, each item is a per-update loss list
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    loss_history_path = Path(loss_history_path)
    if not loss_history_path.exists():
        raise FileNotFoundError(
            f"Could not find loss history file at {loss_history_path}"
        )

    loaded = np.load(loss_history_path, allow_pickle=True)
    across_epochs = np.asarray(
        loaded["across_epochs"],
        dtype=np.float64,
    )
    within_epoch_raw = loaded["within_epoch"]

    within_epoch: list[np.ndarray] = [
        np.asarray(epoch_losses, dtype=np.float64)
        for epoch_losses in within_epoch_raw
    ]

    if out_path is None:
        out_path = loss_history_path.with_name(
            f"{loss_history_path.stem}_plot.png"
        )
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if across_epochs.size > 0:
        axes[0].plot(
            np.arange(1, across_epochs.size + 1),
            across_epochs,
            marker="o",
        )
    axes[0].set_title("Average loss across epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    has_within_epoch = False
    for epoch_index, epoch_losses in enumerate(within_epoch, start=1):
        if epoch_losses.size == 0:
            continue
        has_within_epoch = True
        axes[1].plot(epoch_losses, alpha=0.5, label=f"epoch {epoch_index}")

    axes[1].set_title("Per-update loss within each epoch")
    axes[1].set_xlabel("Update step")
    axes[1].set_ylabel("Loss")
    if has_within_epoch and len(within_epoch) <= 10:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

    return str(out_path)


## Auxiliary function for grid search evaluation
def grid_search():
    return