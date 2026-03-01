from pathlib import Path

import kagglehub
import numpy as np
import typer

from evaluate import test_nearest_neighbors, test_sum_words
from model import CBOW, SkipGram, SkipGramNegativeSampling
from utils import UNK_TOKEN, build_vocab, load_csv_train_val_test, tokenize

app = typer.Typer(
    pretty_exceptions_enable=False,
    help=(
        "Train and evaluate a word2vec implementation on "
        "Brown-corpus dataset"
    ),
)


@app.command()
def main(
    train: str = typer.Option(
        None,
        "--train",
        help="If specified, should be either cbow (Continous Bag of Words) or sgns (Skip gram negative sampling)",
    ),
    load_model: Path | None = typer.Option(
        None,
        "--load-model",
        help="Path from where to load model embeddings (.npy).",
    ),
    evaluate: bool = typer.Option(
        True,
        "--evaluate/--no-evaluate",
        help="Calls evaluation pipeline.",
    ),
):
    if not train and not load_model:
        raise typer.BadParameter(
            "Nothing to do. Use --train, --load-model, or both."
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
    learning_rate = 0.05
    subsample_t = 1e-2
    epochs = 5
    negative_samples = 5
    model_type = "sgns"  # "cbow" or "sgns"

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

    train_token_count = sum(len(doc) for doc in train_data)
    val_token_count = sum(len(doc) for doc in val_data)
    test_token_count = sum(len(doc) for doc in test_data)
    unk_id = word_to_id[UNK_TOKEN]
    unk_count = sum(1 for doc in train_data for wid in doc if wid == unk_id)
    unk_ratio = (
        (100 * unk_count / train_token_count)
        if train_token_count > 0
        else 0.0
    )

    print(
        f"Train: {train_token_count} tokens, "
        f"Val: {val_token_count} tokens, "
        f"Test: {test_token_count} tokens"
    )
    print(
        f"Vocabulary: {vocab_len} (incl. {UNK_TOKEN}). "
        f"Uknown ratio: {unk_ratio}%"
    )

    model: SkipGram
    if model_type == "sgns":
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

    loaded_embeddings = False
    if load_model:
        if not load_model.exists():
            raise typer.BadParameter(
                f"Model file does not exist: {load_model}"
            )

        loaded_w1 = np.load(load_model)
        if loaded_w1.shape != model.W1.shape:
            raise typer.BadParameter(
                "Loaded embeddings shape does not match model config. "
                f"Expected {model.W1.shape}, got {loaded_w1.shape}."
            )

        model.W1 = loaded_w1
        loaded_embeddings = True
        print(f"Loaded embeddings from {load_model}")

    if train:
        print("Starting training...")
        model.fit(train_data, epochs=epochs)

        val_loss = model.evaluate(val_data)
        test_loss = model.evaluate(test_data)
        print(f"Val loss: {val_loss:.4f}, Test loss: {test_loss:.4f}")

        np.save("embeddings.npy", model.embeddings)
        print("Saved trained embeddings to embeddings.npy")

    if evaluate:
        print("Running evaluation pipeline...")
        if train:
            print(
                "Evaluation includes nearest neighbors on trained "
                "embeddings."
            )
        elif loaded_embeddings:
            print(
                "Loaded input embeddings only; running nearest-neighbor "
                "evaluation."
            )

        test_nearest_neighbors(word_to_id, model.embeddings)
        test_sum_words(word_to_id, model.embeddings)

    print("Finished.")


if __name__ == "__main__":
    app()
