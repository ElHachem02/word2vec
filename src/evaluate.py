import numpy as np


def _get_embedding(
    word: str,
    word_to_id: dict[str, int],
    W1: np.ndarray,
) -> np.ndarray | None:
    """Retrieves the trained vector for a given word."""
    if word not in word_to_id:
        return None
    word_id = word_to_id[word]
    # W1 is traditionally used as the final word embedding matrix
    return W1[word_id]


def _cosine_similarity(vec_a, vec_b):
    """Calculates the cosine of the angle between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Prevent division by zero just in case
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _find_nearest_neighbors(
    word: str,
    word_to_id: dict[str, int],
    W1: np.ndarray,
    top_n: int = 5,
):
    """Finds the most similar words in the vocabulary."""
    if word not in word_to_id:
        print(f"Word '{word}' is out of vocabulary.")
        return

    query_vec = _get_embedding(word, word_to_id, W1)
    similarities = []

    # Compare the target word against every other word in the vocab
    for vocab_word, vocab_id in word_to_id.items():
        if vocab_word == word:
            continue  # Skip comparing the word to itself

        vocab_vec = W1[vocab_id]
        sim = _cosine_similarity(query_vec, vocab_vec)
        similarities.append((vocab_word, sim))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\nNearest neighbors for '{word}':")
    for neighbor, sim in similarities[:top_n]:
        print(f"  - {neighbor}: {sim:.4f}")


def test_nearest_neighbors(
    word_to_id: dict[str, int],
    W1: np.ndarray,
    top_n: int = 5,
    test_words: list[str] = ["man", "time", "year", "good"],
):
    for word in test_words:
        _find_nearest_neighbors(word, word_to_id, W1, top_n)
