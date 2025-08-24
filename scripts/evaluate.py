import numpy as np
from src.config import DATA_DIR, RANDOM_SEED
from src.data_prep import download_movielens_if_needed, load_movielens, filter_min_counts, build_user_item_matrix
from src.recommender import train_and_pack, recommend_for_user
from src.evaluate import rmse, precision_at_k, user_stratified_split

def main():
    root = download_movielens_if_needed()
    ratings, movies = load_movielens(root)
    ratings = filter_min_counts(ratings)

    # simple user-held-out split
    train_df, test_df = user_stratified_split(ratings, random_state=RANDOM_SEED)
    R, u_index, i_index = build_user_item_matrix(train_df)
    id_to_title = dict(zip(movies.movieId, movies.title))

    art = train_and_pack(R, u_index, i_index, id_to_title)
    preds = art.svd.predict_all()

    # RMSE on entries that exist in test set
    inv_u = {v:k for k,v in u_index.items()}
    inv_i = {v:k for k,v in i_index.items()}
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        uid, mid, r = int(row.userId), int(row.movieId), float(row.rating)
        if uid in u_index and mid in i_index:
            y_true.append(r)
            y_pred.append(preds[u_index[uid], i_index[mid]])
    rmse_val = rmse(np.array(y_pred), np.array(y_true)) if y_true else float("nan")

    # Precision@10 by recommending and checking if held-out item is in top‑k
    # Build ground truth set per user (held-out items)
    heldout_by_user = {}
    for _, row in test_df.iterrows():
        heldout_by_user.setdefault(int(row.userId), set()).add(int(row.movieId))

    p_at_10 = []
    for uid in heldout_by_user.keys():
        if uid not in u_index:
            continue
        recs = recommend_for_user(art, uid, k=10)
        rec_movie_ids = [m for m,_ in recs]
        p = precision_at_k(rec_movie_ids, heldout_by_user[uid], k=10)
        p_at_10.append(p)
    prec10 = float(np.mean(p_at_10)) if p_at_10 else float("nan")

    print(f"RMSE: {rmse_val:.4f}")
    print(f"Precision@10: {prec10:.4f}")

if __name__ == "__main__":
    main()
