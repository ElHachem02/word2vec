import numpy as np
import os
import re
import csv
from collections import Counter

UNK_TOKEN = "<UNK>"


## Auxiliary functions for training the model
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))  # Numerical stability
    return e_x / e_x.sum(axis=0)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically-stable sigmoid."""
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


## Auxiliary functions for loading and preprocessing the data
def load_corpus_from_csv(data_path: str, max_rows: int | None = None) -> list[list[str]]:
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

    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            tokenized_text = row.get("tokenized_text") or ""
            words: list[str] = []
            for tok in tokenized_text.split():
                tok = tok.lower()
                tok = re.sub(r"[^a-z]", "", tok)
                if tok:
                    words.append(tok)
            result.append(words)

    return result


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
    indices = rng.permutation(len(documents))
    n = len(documents)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = [documents[i] for i in indices[:n_train]]
    val_files = [documents[i] for i in indices[n_train: n_train + n_val]]
    test_files = [documents[i] for i in indices[n_train + n_val:]]

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
    documents = load_corpus_from_csv(data_path, max_rows=max_rows)
    return split_train_val_test(
        documents,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def load_and_preprocess(data_path: str, max_rows: int = 100) -> list[list[str]]:
    """Legacy: load from CSV and return flat list of words."""
    documents = load_corpus_from_csv(data_path, max_rows=max_rows)
    return documents
