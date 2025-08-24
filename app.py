import os, json
from flask import Flask, request, jsonify
from src.config import DATA_DIR
from src.data_prep import download_movielens_if_needed, load_movielens, filter_min_counts, build_user_item_matrix, join_titles, search_titles
from src.recommender import load_or_train, train_and_pack, recommend_for_user, similar_items
from src.llm_interface import parse_with_openai

app = Flask(__name__)

def _build_artifacts():
    root = download_movielens_if_needed(DATA_DIR)
    ratings, movies = load_movielens(root)
    ratings = filter_min_counts(ratings)
    R, u_idx, i_idx = build_user_item_matrix(ratings)
    id_to_title = join_titles(movies)
    return train_and_pack(R, u_idx, i_idx, id_to_title)

ART = load_or_train(_build_artifacts)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/search")
def search():
    q = request.args.get("q", "")
    root = download_movielens_if_needed(DATA_DIR)
    _, movies = load_movielens(root)
    res = search_titles(movies, q, top=15)
    return jsonify(res.to_dict(orient="records"))

@app.get("/recommend/user/<int:user_id>")
def rec_user(user_id: int):
    k = int(request.args.get("k", 10))
    try:
        recs = recommend_for_user(ART, user_id, k=k)
        return jsonify([{"movieId": mid, "title": title} for mid, title in recs])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/similar/<int:movie_id>")
def similar(movie_id: int):
    k = int(request.args.get("k", 10))
    try:
        sims = similar_items(ART, movie_id, k=k)
        return jsonify([{"movieId": mid, "title": title} for mid, title in sims])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/llm")
def llm():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "")
    k = int(data.get("k", 10))
    parsed = parse_with_openai(query)
    intent = parsed.get("intent", "recommend")
    seed = parsed.get("seed_movie")
    genres = parsed.get("genres", [])
    k = int(parsed.get("k", k))

    # Resolve seed movie if provided
    seed_movie_id = None
    if seed:
        root = download_movielens_if_needed(DATA_DIR)
        _, movies = load_movielens(root)
        match = movies[movies['title'].str.contains(seed, case=False, na=False)]
        if not match.empty:
            seed_movie_id = int(match.iloc[0]['movieId'])

    if intent in ("similar",) and seed_movie_id:
        sims = similar_items(ART, seed_movie_id, k=k)
        return jsonify({"parsed": parsed, "results": [{"movieId": mid, "title": title} for mid, title in sims]})
    else:
        # default: user‑agnostic popular-ish recommendations filtered by genre via cosine anchors
        # pick an arbitrary existing user with many ratings or fallback to user 1
        try_users = sorted(ART.u_index.keys())
        user_id = try_users[0] if try_users else 1
        recs = recommend_for_user(ART, user_id, k=50)
        results = [{"movieId": mid, "title": title} for mid,title in recs]

        if genres:
            gl = [g.lower() for g in genres]
            root = download_movielens_if_needed(DATA_DIR)
            _, movies = load_movielens(root)
            gmap = dict(zip(movies.movieId, movies.title.str.lower()))
            # crude genre filtering: title keywords as proxy
            filt = [r for r in results if any(g in gmap.get(r["movieId"], "") for g in gl)]
            results = filt or results  # fallback if empty

        return jsonify({"parsed": parsed, "results": results[:k]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
