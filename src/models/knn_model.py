from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class ItemCosineKNN:
    def __init__(self, topk: int = 50):
        self.topk = topk
        self.item_norm = None
        self.item_vectors = None
        self.sim = None  # optional precomputed similarities

    def fit(self, R: csr_matrix):
        # item vectors are columns; use transpose to normalize by item
        X = R.T.tocsr().astype(np.float32)
        self.item_vectors = normalize(X, axis=1)  # L2 normalize each item vector
        return self

    def similar_items(self, item_idx: int, k: int = 10) -> np.ndarray:
        v = self.item_vectors[item_idx]
        sims = self.item_vectors @ v.T  # cosine similarity
        sims = np.asarray(sims.todense()).ravel()
        sims[item_idx] = -np.inf
        top = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
        return top[np.argsort(sims[top])[::-1]]

    def score_user(self, user_vector, exclude_indices) -> np.ndarray:
        # simple item-based scoring: weighted sum of neighbors by user's existing ratings
        # user_vector: dense vector of user's ratings (zeros for unknown)
        rated = np.where(user_vector > 0)[0]
        if len(rated) == 0:
            return np.zeros(self.item_vectors.shape[0], dtype=np.float32)
        sims = self.item_vectors[rated] @ self.item_vectors.T  # (r x n_items)
        weights = user_vector[rated][:, None]
        scores = (weights * sims).sum(axis=0).A.ravel()
        scores[exclude_indices] = -np.inf
        return scores

    def recommend_for_user(self, user_idx: int, R: csr_matrix, k: int = 10) -> np.ndarray:
        user_vec = R[user_idx].toarray().ravel().astype(np.float32)
        known = R[user_idx].indices
        scores = self.score_user(user_vec, exclude_indices=known)
        top = np.argpartition(-scores, kth=min(k, len(scores)-1))[:k]
        return top[np.argsort(scores[top])[::-1]]
