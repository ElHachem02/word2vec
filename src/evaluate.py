import numpy as np


def _get_embedding(
    word: str,
    word_to_id: dict[str, int],
    W1: np.ndarray,
) -> np.ndarray:
    """Retrieves the trained vector for a given word."""
    if word not in word_to_id:
        raise ValueError(
            f"word '{word}' is not part of the vocabulary"
        )
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


def _get_nearest_neighbors_from_vec(
    word: str | None,
    query_vec: np.ndarray,
    word_to_id: dict[str, int],
    W1: np.ndarray,
):
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
    return similarities


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
    similarities = _get_nearest_neighbors_from_vec(
        word=word,
        query_vec=query_vec,
        word_to_id=word_to_id,
        W1=W1,
    )

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


def test_sum_words(
    word_to_id: dict[str, int],
    W1: np.ndarray,
    top_n: int = 5,
    left_part: list[str] = ["king", "woman", "man"],
    right_part: str = "queen",
):
    """
    Evaluate vector analogy using the pattern:
    left_part[0] + left_part[1] - left_part[2] ~= right_part

    Example:
        king + woman - man ~= queen
    """
    if len(left_part) != 3:
        raise ValueError("left_part must contain exactly 3 words")

    if right_part not in word_to_id:
        raise ValueError(f"'{right_part}' is not in the vocabulary")

    missing_words = [word for word in left_part if word not in word_to_id]
    if missing_words:
        raise ValueError(
            f"Words not in vocabulary: {', '.join(missing_words)}"
        )

    vec_a = _get_embedding(left_part[0], word_to_id, W1)
    vec_b = _get_embedding(left_part[1], word_to_id, W1)
    vec_c = _get_embedding(left_part[2], word_to_id, W1)
    query_vec = vec_a + vec_b - vec_c

    similarities = _get_nearest_neighbors_from_vec(
        word=None,
        query_vec=query_vec,
        word_to_id=word_to_id,
        W1=W1,
    )

    print(
        "\nAnalogy: "
        f"{left_part[0]} + {left_part[1]} - {left_part[2]} "
        f"~= {right_part}"
    )
    print(f"Top {top_n} nearest neighbors:")
    for neighbor, sim in similarities[:top_n]:
        print(f"  - {neighbor}: {sim:.4f}")

    target_rank = None
    for rank, (neighbor, _) in enumerate(similarities, start=1):
        if neighbor == right_part:
            target_rank = rank
            break

    if target_rank is None:
        print(f"'{right_part}' was not found in nearest neighbors.")
    else:
        print(f"Target '{right_part}' rank: {target_rank}")
