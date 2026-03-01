import numpy as np
from utils import softmax, sigmoid
from abc import ABC, abstractmethod


class SkipGram(ABC):
    """Abstract base class for word2vec-style models."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        window_size: int = 2,
        learning_rate: float = 0.01,
        subsample_t: float = 1e-5,
        adagrad_epsilon: float = 1e-8,
        seed: int = 42,
    ):
        self.V = vocab_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.subsample_t = subsample_t
        self.adagrad_epsilon = adagrad_epsilon
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.uniform(-0.1, 0.1, (vocab_size, embed_size))
        self.W2 = self.rng.uniform(-0.1, 0.1, (embed_size, vocab_size))
        self.W1_grad_sq = np.zeros_like(self.W1)
        self.W2_grad_sq = np.zeros_like(self.W2)
        self.loss_history: dict[str, list] = {
            "within_epoch": [],
            "across_epochs": [],
        }

    def _reset_loss_history(self):
        self.loss_history = {
            "within_epoch": [],
            "across_epochs": [],
        }

    def _apply_adagrad_update(
        self,
        weights: np.ndarray,
        grad: np.ndarray,
        grad_sq_accumulator: np.ndarray,
    ):
        """Apply one AdaGrad update step in-place."""
        grad_sq_accumulator += np.square(grad)
        adjusted_lr = self.learning_rate / (
            np.sqrt(grad_sq_accumulator) + self.adagrad_epsilon
        )
        weights -= adjusted_lr * grad

    @abstractmethod
    def fit(
        self,
        data_paragraphs: list[list[int]],
        epochs: int = 3,
        progress_interval: int = 10000,
    ):
        """Train model parameters on tokenized paragraphs."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Run a model-specific forward pass for a training sample."""

    @abstractmethod
    def backward(self, *args, **kwargs):
        """Run a model-specific backward/update step."""

    @abstractmethod
    def evaluate(self, data_paragraphs: list[list[int]]) -> float:
        """Evaluate average loss on tokenized paragraphs."""

    @property
    def embeddings(self):
        """Return input embedding matrix."""
        return self.W1


class CBOW(SkipGram):
    """Continuous Bag of Words word embedding model."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        window_size: int = 2,
        learning_rate: float = 0.01,
        subsample_t: float = 1e-5,
        adagrad_epsilon: float = 1e-8,
        seed: int = 42,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_size=embed_size,
            window_size=window_size,
            learning_rate=learning_rate,
            subsample_t=subsample_t,
            adagrad_epsilon=adagrad_epsilon,
            seed=seed,
        )

    def backward(
        self,
        h: np.ndarray,
        y_pred: np.ndarray,
        target_id: int,
        context_ids: list[int],
    ):
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

        # 3. Parameter updates (AdaGrad)
        self._apply_adagrad_update(self.W2, dW2, self.W2_grad_sq)

        # Gradient of the sum is applied to each context row independently
        for cid in context_ids:
            self._apply_adagrad_update(
                self.W1[cid],
                dh,
                self.W1_grad_sq[cid],
            )

    def forward(
        self,
        context_ids: list[int],
        target_id: int,
    ) -> tuple[float, np.ndarray, np.ndarray]:
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

        self._reset_loss_history()

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
            epoch_losses: list[float] = []

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
                    loss, h, y_pred = self.forward(context_ids, target_id)
                    total_loss += loss
                    epoch_losses.append(float(loss))

                    # Backward pass and parameter updates
                    self.backward(h, y_pred, target_id, context_ids)

                    if (
                        num_trained % progress_interval == 0
                        and num_trained > 0
                    ):
                        print(
                            f"Processed {num_trained} windows. "
                            f"Current avg loss: {total_loss / num_trained:.4f}"
                        )

            avg_loss = total_loss / num_trained if num_trained > 0 else 0.0
            self.loss_history["within_epoch"].append(epoch_losses)
            self.loss_history["across_epochs"].append(float(avg_loss))
            print(
                f"Epoch {epoch + 1} complete. Final Average Loss: "
                f"{avg_loss:.4f} (trained on {num_trained} windows)"
            )

        return self.loss_history

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
                loss, _, _ = self.forward(context_ids, target_id)
                total_loss += loss
                count += 1
        return total_loss / count if count > 0 else 0.0


class SkipGramNegativeSampling(SkipGram):
    """Skip-gram model trained with negative sampling."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        window_size: int = 2,
        learning_rate: float = 0.01,
        negative_samples: int = 5,
        subsample_t: float = 1e-5,
        adagrad_epsilon: float = 1e-8,
        seed: int = 42,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_size=embed_size,
            window_size=window_size,
            learning_rate=learning_rate,
            subsample_t=subsample_t,
            adagrad_epsilon=adagrad_epsilon,
            seed=seed,
        )
        self.negative_samples = negative_samples

        # SGNS needs one output embedding per vocabulary item.
        # Shape must be (V, embed_size), not (embed_size, V) as in CBOW.
        self.W2 = self.rng.uniform(-0.1, 0.1, (vocab_size, embed_size))
        self.W2_grad_sq = np.zeros_like(self.W2)

    def _sample_negative_ids(
        self,
        distribution: np.ndarray,
        positive_id: int,
    ) -> np.ndarray:
        """Sample negative IDs, excluding the positive target ID."""
        negative_ids = self.rng.choice(
            self.V,
            size=self.negative_samples,
            p=distribution,
        )

        while np.any(negative_ids == positive_id):
            mask = negative_ids == positive_id
            negative_ids[mask] = self.rng.choice(
                self.V,
                size=np.sum(mask),
                p=distribution,
            )

        return negative_ids

    def _build_distributions(
        self,
        data_paragraphs: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build keep probabilities and negative-sampling distribution.
        """
        flat_tokens = [
            token_id
            for paragraph in data_paragraphs
            for token_id in paragraph
        ]

        if len(flat_tokens) == 0:
            return np.array([]), np.array([])

        word_freq = np.bincount(
            flat_tokens,
            minlength=self.V,
        ).astype(np.float64)
        prob = word_freq / np.sum(word_freq)
        keep_prob = np.minimum(
            1.0,
            np.sqrt(self.subsample_t / (prob + 1e-10)),
        )

        neg_weights = np.power(word_freq, 0.75)
        if np.sum(neg_weights) == 0:
            neg_distribution = np.full(self.V, 1.0 / self.V)
        else:
            neg_distribution = neg_weights / np.sum(neg_weights)

        return keep_prob, neg_distribution

    def forward(
        self,
        center_id: int,
        context_id: int,
        neg_distribution: np.ndarray,
    ) -> dict[str, np.ndarray | float | int]:
        """Compute pair-level intermediates and loss for SGNS."""
        v_center = self.W1[center_id].copy()
        u_pos = self.W2[context_id].copy()

        pos_score = np.dot(v_center, u_pos)
        pos_sigmoid = float(sigmoid(pos_score))

        negative_ids = self._sample_negative_ids(neg_distribution, context_id)
        u_negs = self.W2[negative_ids].copy()
        neg_scores = np.dot(u_negs, v_center)
        neg_sigmoids = np.asarray(sigmoid(neg_scores), dtype=np.float64)

        pair_loss = -np.log(pos_sigmoid + 1e-9)
        pair_loss -= np.sum(np.log(1.0 - neg_sigmoids + 1e-9))

        return {
            "center_id": center_id,
            "context_id": context_id,
            "v_center": v_center,
            "u_pos": u_pos,
            "negative_ids": negative_ids,
            "u_negs": u_negs,
            "pos_sigmoid": pos_sigmoid,
            "neg_sigmoids": neg_sigmoids,
            "pair_loss": float(pair_loss),
        }

    def backward(self, cache: dict[str, np.ndarray | float | int]):
        """Apply SGNS parameter updates using a forward-pass cache."""
        center_id = int(cache["center_id"])
        context_id = int(cache["context_id"])
        v_center = np.asarray(cache["v_center"])
        u_pos = np.asarray(cache["u_pos"])
        negative_ids = np.asarray(cache["negative_ids"], dtype=np.int64)
        u_negs = np.asarray(cache["u_negs"])
        pos_sigmoid = float(cache["pos_sigmoid"])
        neg_sigmoids = np.asarray(cache["neg_sigmoids"], dtype=np.float64)

        grad_center = (pos_sigmoid - 1.0) * u_pos
        grad_center += np.dot(neg_sigmoids, u_negs)

        grad_context = (pos_sigmoid - 1.0) * v_center
        self._apply_adagrad_update(
            self.W2[context_id],
            grad_context,
            self.W2_grad_sq[context_id],
        )

        for idx, neg_id in enumerate(negative_ids):
            grad_negative = neg_sigmoids[idx] * v_center
            self._apply_adagrad_update(
                self.W2[neg_id],
                grad_negative,
                self.W2_grad_sq[neg_id],
            )

        self._apply_adagrad_update(
            self.W1[center_id],
            grad_center,
            self.W1_grad_sq[center_id],
        )

    def fit(
        self,
        data_paragraphs: list[list[int]],
        epochs: int = 3,
        progress_interval: int = 10000,
    ):
        """
        Train SGNS on tokenized paragraphs.

        Args:
            data_paragraphs: List of paragraphs, each paragraph is a list of
                word IDs.
            epochs: Number of training epochs.
            progress_interval: Print progress every N trained pairs.
        """
        keep_prob, neg_distribution = self._build_distributions(
            data_paragraphs
        )
        if keep_prob.size == 0:
            print("No training tokens provided. Skipping training.")
            return

        self._reset_loss_history()

        for epoch in range(epochs):
            total_loss = 0.0
            num_trained_pairs = 0
            epoch_losses: list[float] = []

            for paragraph in data_paragraphs:
                if len(paragraph) <= 1:
                    continue

                for i, center_id in enumerate(paragraph):
                    if self.rng.random() > keep_prob[center_id]:
                        continue

                    left = max(0, i - self.window_size)
                    right = min(len(paragraph), i + self.window_size + 1)

                    for j in range(left, right):
                        if j == i:
                            continue

                        context_id = paragraph[j]
                        cache = self.forward(
                            center_id,
                            context_id,
                            neg_distribution,
                        )
                        self.backward(cache)
                        loss = float(cache["pair_loss"])
                        total_loss += loss
                        epoch_losses.append(loss)
                        num_trained_pairs += 1

                        if (
                            num_trained_pairs % progress_interval == 0
                            and num_trained_pairs > 0
                        ):
                            print(
                                f"Processed {num_trained_pairs} pairs. "
                                "Current avg loss: "
                                f"{total_loss / num_trained_pairs:.4f}"
                            )

            avg_loss = (
                total_loss / num_trained_pairs
                if num_trained_pairs > 0
                else 0.0
            )
            self.loss_history["within_epoch"].append(epoch_losses)
            self.loss_history["across_epochs"].append(float(avg_loss))
            print(
                f"Epoch {epoch + 1} complete. Final Average Loss: "
                f"{avg_loss:.4f} (trained on {num_trained_pairs} pairs)"
            )

        return self.loss_history

    def evaluate(self, data_paragraphs: list[list[int]]) -> float:
        """Compute average SGNS loss on paragraphs (no updates)."""
        keep_prob, neg_distribution = self._build_distributions(
            data_paragraphs
        )
        if keep_prob.size == 0:
            return 0.0

        total_loss = 0.0
        total_pairs = 0

        for paragraph in data_paragraphs:
            if len(paragraph) <= 1:
                continue

            for i, center_id in enumerate(paragraph):
                if self.rng.random() > keep_prob[center_id]:
                    continue

                left = max(0, i - self.window_size)
                right = min(len(paragraph), i + self.window_size + 1)

                for j in range(left, right):
                    if j == i:
                        continue

                    context_id = paragraph[j]
                    v_center = self.W1[center_id]
                    u_pos = self.W2[context_id]

                    pos_score = np.dot(v_center, u_pos)
                    pos_sigmoid = float(sigmoid(pos_score))

                    negative_ids = self._sample_negative_ids(
                        neg_distribution,
                        context_id,
                    )
                    u_negs = self.W2[negative_ids]
                    neg_scores = np.dot(u_negs, v_center)
                    neg_sigmoids = np.asarray(
                        sigmoid(neg_scores),
                        dtype=np.float64,
                    )

                    pair_loss = -np.log(pos_sigmoid + 1e-9)
                    pair_loss -= np.sum(np.log(1.0 - neg_sigmoids + 1e-9))

                    total_loss += float(pair_loss)
                    total_pairs += 1

        return total_loss / total_pairs if total_pairs > 0 else 0.0
