from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

def rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth) ** 2))

def precision_at_k(recommended: np.ndarray, heldout_true: set, k: int = 10) -> float:
    if k == 0: 
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in heldout_true)
    return hits / k

def user_stratified_split(ratings_df, test_size=0.2, random_state=42):
    # per-user holdout of one rating if possible; rest to train
    rng = np.random.default_rng(random_state)
    test_rows = []
    train_mask = np.ones(len(ratings_df), dtype=bool)
    for uid, grp in ratings_df.groupby("userId"):
        idxs = grp.index.to_list()
        if len(idxs) < 2:
            continue
        test_i = rng.choice(idxs, 1)[0]
        test_rows.append(test_i)
        train_mask[test_i] = False
    test_df = ratings_df.loc[test_rows]
    train_df = ratings_df.loc[train_mask]
    return train_df, test_df
