from __future__ import annotations
from dataclasses import dataclass
import os, joblib, numpy as np
from typing import Dict, Any, Tuple, List

from .config import ARTIFACT_DIR, ALPHA, KNN_TOPK, SVD_COMPONENTS
from .models.svd_model import SVDRecommender
from .models.knn_model import ItemCosineKNN

@dataclass
class Artifacts:
    svd: SVDRecommender
    knn: ItemCosineKNN
    R: Any
    u_index: dict
    i_index: dict
    id_to_title: dict
    users_sorted: list
    items_sorted: list

MODEL_PATH = os.path.join(ARTIFACT_DIR, "recsys.joblib")

def build_hybrid_score(svd_scores: np.ndarray, knn_scores: np.ndarray, alpha: float = ALPHA):
    # weighted combination; normalize to comparable scale
    s1 = svd_scores
    s2 = knn_scores
    if np.isinf(s1).any():
        s1 = np.where(np.isinf(s1), -1e9, s1)
    if np.isinf(s2).any():
        s2 = np.where(np.isinf(s2), -1e9, s2)
    # z-score normalize
    def z(x):
        mu = np.nanmean(x[np.isfinite(x)])
        std = np.nanstd(x[np.isfinite(x)]) + 1e-8
        return (x - mu) / std
    return alpha * z(s1) + (1-alpha) * z(s2)

def train_and_pack(R, u_index, i_index, id_to_title) -> Artifacts:
    svd = SVDRecommender(n_components=SVD_COMPONENTS).fit(R)
    knn = ItemCosineKNN(topk=KNN_TOPK).fit(R)

    users_sorted = sorted(u_index.keys())
    items_sorted = sorted(i_index.keys())

    art = Artifacts(svd=svd, knn=knn, R=R, u_index=u_index, i_index=i_index,
                    id_to_title=id_to_title, users_sorted=users_sorted, items_sorted=items_sorted)
    return art

def save_artifacts(art: Artifacts):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(art, MODEL_PATH)

def load_artifacts() -> Artifacts:
    return joblib.load(MODEL_PATH)

def load_or_train(build_fn) -> Artifacts:
    if os.path.exists(MODEL_PATH):
        return load_artifacts()
    art = build_fn()
    save_artifacts(art)
    return art

def recommend_for_user(art: Artifacts, raw_user_id: int, k: int = 10) -> List[Tuple[int, str]]:
    if raw_user_id not in art.u_index:
        raise ValueError(f"Unknown user_id {raw_user_id}")
    uidx = art.u_index[raw_user_id]
    svd_top = art.svd.recommend_for_user(uidx, known_item_indices=art.R[uidx].indices, k=max(k, 50))
    knn_top = art.knn.recommend_for_user(uidx, art.R, k=max(k, 50))

    # build hybrid score array across all items
    svd_scores = art.svd.predict_all()[uidx]
    knn_scores = art.knn.score_user(art.R[uidx].toarray().ravel(), exclude_indices=art.R[uidx].indices)
    hybrid = build_hybrid_score(svd_scores, knn_scores)
    # mask out known
    hybrid[art.R[uidx].indices] = -1e9

    top = np.argpartition(-hybrid, kth=min(k, len(hybrid)-1))[:k]
    top = top[np.argsort(hybrid[top])[::-1]]
    # map indices back to movieIds
    inv_i = {v:k for k,v in art.i_index.items()}
    results = [(int(inv_i[i]), art.id_to_title[int(inv_i[i])]) for i in top]
    return results

def similar_items(art: Artifacts, raw_movie_id: int, k: int = 10):
    if raw_movie_id not in art.i_index:
        raise ValueError(f"Unknown movie_id {raw_movie_id}")
    midx = art.i_index[raw_movie_id]
    top = art.knn.similar_items(midx, k=k)
    inv_i = {v:k for k,v in art.i_index.items()}
    return [(int(inv_i[i]), art.id_to_title[int(inv_i[i])]) for i in top]
