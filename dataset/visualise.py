from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import kagglehub

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.utils import UNK_TOKEN
from src.utils import (
    load_csv_train_val_test_with_label_dist,
    build_vocab,
    tokenize,
)

matplotlib.use("Agg")  # headless-safe; we only save figures


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_top_k(counts: Counter, out_path: Path, k: int = 30):
    top = counts.most_common(k)
    words = [w for w, _ in top]
    freqs = [c for _, c in top]

    plt.figure(figsize=(12, 5))
    plt.bar(words, freqs)
    plt.title(f"Top {k} most frequent tokens in training data")
    plt.xlabel("Token")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    _savefig(out_path)


def plot_split_sizes(
    train_files: list[list[str]],
    val_files: list[list[str]],
    test_files: list[list[str]],
    out_path: Path,
):
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    total_tokens = {k: sum(len(x) for x in v) for k, v in splits.items()}
    num_files = {k: len(v) for k, v in splits.items()}
    distinct_tokens = {
        split_name: len(set(w for file_words in files for w in file_words))
        for split_name, files in splits.items()
    }

    labels = list(splits.keys())
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width,
        [num_files[label] for label in labels],
        width,
        label="#paragraphs",
    )
    plt.bar(
        x,
        [total_tokens[label] for label in labels],
        width,
        label="#tokens",
    )
    plt.bar(
        x + width,
        [distinct_tokens[label] for label in labels],
        width,
        label="#distinct tokens",
    )
    plt.xticks(x, labels)
    plt.title("Split sizes")
    plt.legend()
    _savefig(out_path)


def plot_unk_ratio(
    word_to_id: dict[str, int],
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    out_path: Path,
):
    unk_id = word_to_id[UNK_TOKEN]

    def ratio(ids: list[int]) -> float:
        if not ids:
            return 0.0
        return sum(1 for x in ids if x == unk_id) / len(ids)

    labels = ["train", "val", "test"]
    ratios = [ratio(train_ids), ratio(val_ids), ratio(test_ids)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, ratios)
    plt.title("UNK ratio per split (OOV after vocab cutoff)")
    plt.ylabel("UNK / total tokens")
    plt.ylim(0, max(ratios) * 1.2 + 1e-6)
    _savefig(out_path)


def plot_label_distribution(
    label_distributions: dict[str, dict[str, float]],
    out_path: Path,
):
    split_order = ["overall", "train", "val", "test"]
    labels = sorted(
        {
            label
            for split_name in split_order
            for label in label_distributions.get(split_name, {}).keys()
        }
    )

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(12, 5))
    for idx, split_name in enumerate(split_order):
        dist = label_distributions.get(split_name, {})
        y = [dist.get(label, 0.0) for label in labels]
        plt.bar(x + (idx - 1.5) * width, y, width, label=split_name)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Ratio")
    plt.title("Label distribution (overall vs splits)")
    plt.legend()
    _savefig(out_path)


def plot_overall_label_distribution_pie(
    label_distributions: dict[str, dict[str, float]],
    out_path: Path,
):
    overall = label_distributions.get("overall", {})
    if not overall:
        return

    raw_labels = list(overall.keys())
    labels = [
        "sf" if label == "science_fiction" else label
        for label in raw_labels
    ]
    sizes = [overall[label] for label in raw_labels]

    def _autopct(pct: float) -> str:
        if pct < 5.2:
            return ""
        return f"{pct:.1f}%"

    plt.figure(figsize=(7, 7))
    plt.pie(
        sizes,
        labels=labels,
        autopct=_autopct,
        startangle=90,
    )
    plt.title("Overall label distribution")
    plt.axis("equal")
    _savefig(out_path)


def visualise_dataset(
    data_path: str,
    vocab_size: int = 3000,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    out_dir: str | None = None,
) -> dict:
    """
    End-to-end dataset visualization:
    - Structure summary (folders/files/extensions/depth)
    - Tokens/file histogram
    - Split sizes plot
    - UNK ratio plot
    - Window coverage per file

    Saves plots next to this file (by default) and returns a summary dict.
    """
    if out_dir is None:
        out = Path(__file__).resolve().parent
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load brown.csv, stratified split, and label distributions.
    (
        train_data,
        val_data,
        test_data,
        label_distributions,
    ) = load_csv_train_val_test_with_label_dist(
        data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_words_flat = [w for words in train_data for w in words]
    counts = Counter(train_words_flat)

    word_to_id, _ = build_vocab(train_words_flat, vocab_size=vocab_size)
    train_ids_by_doc = [tokenize(words, word_to_id) for words in train_data]
    val_ids_by_doc = [tokenize(words, word_to_id) for words in val_data]
    test_ids_by_doc = [tokenize(words, word_to_id) for words in test_data]

    train_ids = [x for file_ids in train_ids_by_doc for x in file_ids]
    val_ids = [x for file_ids in val_ids_by_doc for x in file_ids]
    test_ids = [x for file_ids in test_ids_by_doc for x in file_ids]

    distinct_tokens_by_split = {
        "train": len(set(w for words in train_data for w in words)),
        "val": len(set(w for words in val_data for w in words)),
        "test": len(set(w for words in test_data for w in words)),
    }
    num_distinct_tokens = len(
        set(w for words in (train_data + val_data + test_data) for w in words)
    )
    total_tokens_pre_split = len(train_ids) + len(val_ids) + len(test_ids)
    total_files_pre_split = (
        len(train_data) + len(val_data) + len(test_data)
    )

    plot_top_k(counts, out / "top_tokens.png", k=30)
    plot_split_sizes(train_data, val_data, test_data, out / "split_sizes.png")
    plot_unk_ratio(
        word_to_id,
        train_ids,
        val_ids,
        test_ids,
        out / "unk_ratio.png",
    )
    plot_label_distribution(
        label_distributions,
        out / "label_distribution.png",
    )
    plot_overall_label_distribution_pie(
        label_distributions,
        out / "overall_label_distribution_pie.png",
    )

    summary = {
        "splits": {
            "num_files": {
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
            },
            "num_tokens": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "num_distinct_tokens": distinct_tokens_by_split,
        },
        "dataset": {
            "num_distinct_tokens": num_distinct_tokens,
        },
        "pre_split": {
            "num_files": total_files_pre_split,
            "num_tokens": total_tokens_pre_split,
            "num_distinct_tokens": num_distinct_tokens,
        },
        "vocab": {"vocab_size": len(word_to_id), "unk_token": UNK_TOKEN},
        "labels": label_distributions,
        "outputs": {
            "top_tokens": str(out / "top_tokens.png"),
            "tokens_per_file_train": str(out / "tokens_per_file_train.png"),
            "split_sizes": str(out / "split_sizes.png"),
            "unk_ratio": str(out / "unk_ratio.png"),
            "label_distribution": str(out / "label_distribution.png"),
            "overall_label_distribution_pie": str(
                out / "overall_label_distribution_pie.png"
            ),
            "windows_per_file": str(out / "windows_per_file.png"),
        },
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Visualise dataset structure and token statistics."
    )
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output directory for plots. Defaults to the directory "
            "containing this file."
        ),
    )
    args = parser.parse_args()
    data_path = kagglehub.dataset_download("nltkdata/brown-corpus")

    summary = visualise_dataset(
        data_path=data_path,
        vocab_size=args.vocab_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        out_dir=(args.out_dir or None),
    )

    print(json.dumps(summary["splits"], indent=2))
    print(json.dumps(summary["pre_split"], indent=2))
    print(json.dumps(summary["vocab"], indent=2))
    print(json.dumps(summary["labels"], indent=2))
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
