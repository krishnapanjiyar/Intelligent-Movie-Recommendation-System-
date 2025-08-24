import os, joblib
import pandas as pd
from src.config import DATA_DIR, ARTIFACT_DIR, RANDOM_SEED, TEST_SIZE
from src.data_prep import download_movielens_if_needed, load_movielens, filter_min_counts, build_user_item_matrix, join_titles
from src.recommender import train_and_pack, save_artifacts

def main():
    root = download_movielens_if_needed(DATA_DIR)
    ratings, movies = load_movielens(root)
    ratings = filter_min_counts(ratings)

    R, u_index, i_index = build_user_item_matrix(ratings)
    id_to_title = join_titles(movies)

    art = train_and_pack(R, u_index, i_index, id_to_title)
    save_artifacts(art)
    print(f"Artifacts saved to {ARTIFACT_DIR}/")

if __name__ == "__main__":
    main()
