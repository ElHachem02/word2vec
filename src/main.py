from utils import (
    UNK_TOKEN,
    load_csv_train_val_test,
    build_vocab,
    tokenize,
)
from model import SkipGram, CBOW, SkipGramNegativeSampling
from evaluate import test_nearest_neighbors
import kagglehub
import numpy as np


if __name__ == "__main__":
    print("Downloading dataset...")
    path = kagglehub.dataset_download("nltkdata/brown-corpus")

    print("Loading CSV and splitting into train/val/test (80/10/10)...")
    train_files, val_files, test_files = load_csv_train_val_test(
        path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    print(f"Found {len(train_files)} many files")

    VOCAB_SIZE = 3000
    EMBED_SIZE = 50
    WINDOW_SIZE = 2
    LEARNING_RATE = 0.05
    SUBSAMPLE_T = 1e-2
    EPOCHS = 5
    NEGATIVE_SAMPLES = 5
    MODEL_TYPE = "sgns"  # "cbow" or "sgns"

    # Build vocab from TRAIN only (avoid leakage).
    # OOV maps to UNK to preserve word order.
    train_words_flat = [w for words in train_files for w in words]
    print(f"Total trainwords are {len(train_words_flat)}")
    word_to_id, id_to_word = build_vocab(
        train_words_flat,
        vocab_size=VOCAB_SIZE,
    )
    V = len(word_to_id)

    # Tokenize per document: OOV -> UNK.
    # Preserves adjacency and file boundaries.
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
    print(f"Vocabulary: {V} (incl. {UNK_TOKEN}). Uknown ratio: {unk_ratio}%")

    model: SkipGram
    if MODEL_TYPE == "sgns":
        model = SkipGramNegativeSampling(
            vocab_size=V,
            embed_size=EMBED_SIZE,
            window_size=WINDOW_SIZE,
            learning_rate=LEARNING_RATE,
            negative_samples=NEGATIVE_SAMPLES,
            subsample_t=SUBSAMPLE_T,
        )
    else:
        model = CBOW(
            vocab_size=V,
            embed_size=EMBED_SIZE,
            window_size=WINDOW_SIZE,
            learning_rate=LEARNING_RATE,
            subsample_t=SUBSAMPLE_T,
        )

    print("Starting training...")
    model.fit(train_data, epochs=EPOCHS)

    val_loss = model.evaluate(val_data)
    test_loss = model.evaluate(test_data)
    print(f"Val loss: {val_loss:.4f}, Test loss: {test_loss:.4f}")

    # Persist embeddings for later use
    np.save("embeddings.npy", model.embeddings)

    print("Training finished! Access embeddings via model.embeddings")
    test_nearest_neighbors(word_to_id, model.embeddings)
