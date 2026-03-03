import json
from pathlib import Path

import kagglehub
import numpy as np
import typer

from .evaluate import test_nearest_neighbors, test_sum_words
from .model import (CBOW, SkipGram, SkipGramNegativeSampling)
from .utils import (
    build_vocab,
    grid_search_cbow,
    grid_search_negative_sampling,
    load_csv_train_val_test,
    plot_loss_history,
    tokenize,
)

app = typer.Typer(
    pretty_exceptions_enable=False,
    help=(
        "Train and evaluate a word2vec implementation on "
        "Brown-corpus dataset"
    ),
)


@app.command()
def main(
    grid_search: str = typer.Option(
        None,
        "--grid-search",
        help=(
            "If specified, should be either cbow (Continous Bag of "
            "Words), sgns (Skip gram negative sampling), performs grid search on best hyperparameters combination"
        )
    ),
    train: str = typer.Option(
        None,
        "--train",
        help=(
            "If specified, should be either cbow (Continous Bag of "
            "Words), sgns (Skip gram negative sampling)"
        ),
    ),
    load_model: Path | None = typer.Option(
        None,
        "--load-model",
        help="Path from where to load model embeddings (.npy).",
    ),
    epochs: int = typer.Option(
        5,
        "--epochs",
        help="Number of epochs for training"
    )
):
    if not train and not load_model and not grid_search:
        raise typer.BadParameter(
            "Nothing to do. Use --train, --grid-search or --load-model."
        )

    print("Downloading dataset...")
    path = kagglehub.dataset_download("nltkdata/brown-corpus")

    print("Loading CSV and splitting into train/val/test (80/10/10)...")
    train_files, val_files, test_files = load_csv_train_val_test(
        path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    print(f"Found {len(train_files)} many files")

    vocab_size = 3000
    embed_size = 50
    window_size = 2
    learning_rate = 0.1
    subsample_t = 1e-2
    negative_samples = 5

    train_words_flat = [word for words in train_files for word in words]
    print(f"Total trainwords are {len(train_words_flat)}")
    word_to_id, _ = build_vocab(
        train_words_flat,
        vocab_size=vocab_size,
    )
    vocab_len = len(word_to_id)

    train_data = [tokenize(words, word_to_id) for words in train_files]
    val_data = [tokenize(words, word_to_id) for words in val_files]
    test_data = [tokenize(words, word_to_id) for words in test_files]
    model: SkipGram

    if grid_search == "cbow":
        print("Running CBOW grid search...")
        summary = grid_search_cbow(
            train_data=train_data,
            val_data=val_data,
            word_to_id=word_to_id,
            epochs=epochs
        )
        print(json.dumps(summary["best_by_val"], indent=2))
        print(json.dumps(summary["top_k_saved"], indent=2))
        print("Finished.")
        return
    elif grid_search == "sgns":
        print("Running CBOW grid search...")
        summary = grid_search_negative_sampling(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            word_to_id=word_to_id
        )
        print(json.dumps(summary["best_by_val"], indent=2))
        print(json.dumps(summary["top_k_saved"], indent=2))
        print("Finished.")
        return

    if train == "sgns":
        model = SkipGramNegativeSampling(
            vocab_size=vocab_len,
            embed_size=embed_size,
            window_size=window_size,
            learning_rate=learning_rate,
            negative_samples=negative_samples,
            subsample_t=subsample_t,
        )
    else:
        model = CBOW(
            vocab_size=vocab_len,
            embed_size=embed_size,
            window_size=window_size,
            learning_rate=learning_rate,
            subsample_t=subsample_t,
        )

    if load_model:
        loaded_w1 = np.load(load_model)
        if loaded_w1.shape != model.W1.shape:
            raise typer.BadParameter(
                "Loaded embeddings shape does not match model config. "
                f"Expected {model.W1.shape}, got {loaded_w1.shape}."
            )

        model.W1 = loaded_w1
        print(f"Loaded embeddings from {load_model}")
    else:
        print("Starting training...")
        loss_history = model.fit(train_data, epochs=epochs)

        if loss_history:
            loss_history_file = Path(f"{train}_loss_history.npz")
            np.savez(
                loss_history_file,
                across_epochs=np.asarray(
                    loss_history["across_epochs"],
                    dtype=np.float64,
                ),
            )
            plot_path = plot_loss_history(loss_history_file)
            print(f"Saved training loss plot to {plot_path}")

        val_loss = model.evaluate(val_data)
        test_loss = model.evaluate(test_data)
        print(f"Val loss: {val_loss:.4f}, Test loss: {test_loss:.4f}")

        np.save(f"{train}_embeddings.npy", model.embeddings)
        print("Saved trained embeddings to embeddings.npy")

    test_nearest_neighbors(word_to_id, model.embeddings)
    test_sum_words(word_to_id, model.embeddings)

    print("Finished.")


if __name__ == "__main__":
    app()
