from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from typing import Optional

class SVDRecommender:
    def __init__(self, n_components: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None
        self.user_means: Optional[np.ndarray] = None
        self.VT: Optional[np.ndarray] = None  # item factors transposed

    def fit(self, R: csr_matrix):
        # center by user means (classic baseline)
        R = R.tocsr().astype(np.float32)
        sums = np.array(R.sum(axis=1)).flatten()
        counts = np.diff(R.indptr)
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts>0, sums / counts, 0.0).astype(np.float32)
        self.user_means = means

        R_centered = R.copy().astype(np.float32)
        # subtract user means for observed entries
        for u in range(R_centered.shape[0]):
            start, end = R_centered.indptr[u], R_centered.indptr[u+1]
            R_centered.data[start:end] -= means[u]

        self.svd = TruncatedSVD(n_components=min(self.n_components, min(R.shape)-1),
                                random_state=self.random_state)
        U = self.svd.fit_transform(R_centered)   # (n_users x k)
        VT = self.svd.components_               # (k x n_items)
        self.VT = VT.astype(np.float32)
        self.U = U.astype(np.float32)
        return self

    def predict_all(self) -> np.ndarray:
        # reconstruct (approx) and add back user means
        approx = self.U @ self.VT
        approx += self.user_means[:, None]
        return approx

    def recommend_for_user(self, user_idx: int, known_item_indices: np.ndarray, k: int = 10) -> np.ndarray:
        scores = self.predict_all()[user_idx]
        scores[known_item_indices] = -np.inf  # don't recommend already-rated items
        top = np.argpartition(-scores, kth=min(k, len(scores)-1))[:k]
        return top[np.argsort(scores[top])[::-1]]
