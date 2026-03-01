import numpy as np
from utils import softmax


class CBOW:
    """Continuous Bag of Words word embedding model."""

    def __init__(self, vocab_size: int, embed_size: int, window_size: int = 2, learning_rate: float = 0.01, subsample_t: float = 1e-5, seed: int = 42):
        self.V = vocab_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.subsample_t = subsample_t

        np.random.seed(seed)
        self.W1 = np.random.uniform(-0.1, 0.1, (vocab_size, embed_size))  # Input -> Hidden
        self.W2 = np.random.uniform(-0.1, 0.1, (embed_size, vocab_size))  # Hidden -> Output

    def backward_pass(self, h: np.ndarray, y_pred: np.ndarray, target_id: int, context_ids: list[int]):
        """
        Compute gradients and update W1, W2 in place.

        Args:
            h: Hidden state (sum of context embeddings), shape (embed_size,)
            y_pred: Softmax probabilities, shape (V,)
            target_id: Target word index
            context_ids: List of context word indices
        """
        # 1. Output error: prediction minus ground truth (one-hot)
        e = y_pred.copy()
        e[target_id] -= 1.0

        # 2. Gradients for W2 and hidden state
        dW2 = np.outer(h, e)   # Shape: (EMBED_SIZE, V)
        dh = np.dot(self.W2, e)  # Shape: (EMBED_SIZE,)

        # 3. Parameter updates (SGD)
        self.W2 -= self.learning_rate * dW2

        # Gradient of the sum splits equally to all context words
        for cid in context_ids:
            self.W1[cid] -= self.learning_rate * dh

    def _forward(self, context_ids: list[int], target_id: int) -> tuple[float, np.ndarray, np.ndarray]:
        """Forward pass for a single window. Returns (loss, h, y_pred)."""
        h = np.sum(self.W1[context_ids], axis=0)
        u = np.dot(h, self.W2)
        y_pred = softmax(u)
        loss = -np.log(y_pred[target_id] + 1e-9)
        return loss, h, y_pred

    def fit(
        self,
        data_paragraphs: list[list[int]],
        epochs: int = 3,
        progress_interval: int = 10000,
    ):
        """
        Train the CBOW model on tokenized paragraphs.

        Args:
            data_paragraphs: List of paragraphs, each paragraph is a list of
                word IDs.
            epochs: Number of training epochs
            progress_interval: Print progress every N trained windows
        """
        flat_tokens = [
            token_id
            for paragraph in data_paragraphs
            for token_id in paragraph
        ]
        if len(flat_tokens) == 0:
            print("No training tokens provided. Skipping training.")
            return

        word_freq = np.bincount(flat_tokens, minlength=self.V) / len(
            flat_tokens
        )
        keep_prob = np.minimum(
            1.0,
            np.sqrt(self.subsample_t / (word_freq + 1e-10)),
        )

        for epoch in range(epochs):
            total_loss = 0
            num_trained = 0

            for paragraph in data_paragraphs:
                if len(paragraph) <= 2 * self.window_size:
                    continue

                for i in range(
                    self.window_size,
                    len(paragraph) - self.window_size,
                ):
                    context_ids = (
                        paragraph[i - self.window_size: i]
                        + paragraph[i + 1: i + self.window_size + 1]
                    )
                    target_id = paragraph[i]

                    # Subsampling: skip frequent words
                    if np.random.random() > keep_prob[target_id]:
                        continue

                    num_trained += 1

                    # Forward pass
                    loss, h, y_pred = self._forward(context_ids, target_id)
                    total_loss += loss

                    # Backward pass and parameter updates
                    self.backward_pass(h, y_pred, target_id, context_ids)

                    if (
                        num_trained % progress_interval == 0
                        and num_trained > 0
                    ):
                        print(
                            f"Processed {num_trained} windows. "
                            f"Current avg loss: {total_loss / num_trained:.4f}"
                        )

            avg_loss = total_loss / num_trained if num_trained > 0 else 0.0
            print(
                f"Epoch {epoch + 1} complete. Final Average Loss: "
                f"{avg_loss:.4f} (trained on {num_trained} windows)"
            )

    def evaluate(self, data_paragraphs: list[list[int]]) -> float:
        """Compute average loss on paragraphs (no updates, no subsampling)."""
        total_loss = 0
        count = 0
        for paragraph in data_paragraphs:
            if len(paragraph) <= 2 * self.window_size:
                continue

            for i in range(
                self.window_size,
                len(paragraph) - self.window_size,
            ):
                context_ids = (
                    paragraph[i - self.window_size: i]
                    + paragraph[i + 1: i + self.window_size + 1]
                )
                target_id = paragraph[i]
                loss, _, _ = self._forward(context_ids, target_id)
                total_loss += loss
                count += 1
        return total_loss / count if count > 0 else 0.0

    @property
    def embeddings(self):
        """Return the input embedding matrix W1 (vocab_size, embed_size)."""
        return self.W1
