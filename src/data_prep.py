from __future__ import annotations
import os, zipfile, io, requests
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict

from .config import DATA_DIR, MIN_USER_RATINGS, MIN_ITEM_RATINGS

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_movielens_if_needed(data_dir: str = DATA_DIR) -> str:
    os.makedirs(data_dir, exist_ok=True)
    ratings_path = os.path.join(data_dir, "ml-latest-small", "ratings.csv")
    movies_path = os.path.join(data_dir, "ml-latest-small", "movies.csv")
    if os.path.exists(ratings_path) and os.path.exists(movies_path):
        return os.path.join(data_dir, "ml-latest-small")

    # Download
    resp = requests.get(MOVIELENS_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(data_dir)
    return os.path.join(data_dir, "ml-latest-small")

def load_movielens(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(os.path.join(data_root, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_root, "movies.csv"))
    return ratings, movies

def filter_min_counts(ratings: pd.DataFrame) -> pd.DataFrame:
    # filter users/items with too few ratings to reduce sparsity and runtime (~30% improvement typical)
    u_counts = ratings["userId"].value_counts()
    i_counts = ratings["movieId"].value_counts()
    r = ratings[ratings["userId"].isin(u_counts[u_counts >= MIN_USER_RATINGS].index)]
    r = r[r["movieId"].isin(i_counts[i_counts >= MIN_ITEM_RATINGS].index)]
    return r

def build_user_item_matrix(ratings: pd.DataFrame) -> Tuple[csr_matrix, Dict[int,int], Dict[int,int]]:
    users = np.sort(ratings["userId"].unique())
    items = np.sort(ratings["movieId"].unique())
    u_index = {u:i for i,u in enumerate(users)}
    i_index = {m:i for i,m in enumerate(items)}

    row = ratings["userId"].map(u_index).to_numpy()
    col = ratings["movieId"].map(i_index).to_numpy()
    data = ratings["rating"].astype(float).to_numpy()
    mat = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    return mat, u_index, i_index

def join_titles(movies: pd.DataFrame) -> Dict[int,str]:
    return dict(zip(movies["movieId"], movies["title"]))

def compute_sparsity(mat: csr_matrix) -> float:
    nnz = mat.nnz
    total = mat.shape[0] * mat.shape[1]
    return 1.0 - nnz / total if total else 0.0

def search_titles(movies: pd.DataFrame, q: str, top: int = 10) -> pd.DataFrame:
    ql = q.lower()
    m = movies[movies["title"].str.lower().str.contains(ql, na=False)]
    return m.head(top).copy()
