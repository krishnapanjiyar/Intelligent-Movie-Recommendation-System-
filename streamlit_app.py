import streamlit as st
import pandas as pd
from src.recommender import load_or_train, train_and_pack, recommend_for_user, similar_items
from src.data_prep import download_movielens_if_needed, load_movielens, filter_min_counts, build_user_item_matrix, search_titles
from src.config import DATA_DIR

st.set_page_config(page_title="Intelligent Movie Recommender", layout="wide")

@st.cache_resource
def load_artifacts():
    def _build():
        root = download_movielens_if_needed(DATA_DIR)
        ratings, movies = load_movielens(root)
        ratings = filter_min_counts(ratings)
        R, u_idx, i_idx = build_user_item_matrix(ratings)
        id_to_title = dict(zip(movies.movieId, movies.title))
        return train_and_pack(R, u_idx, i_idx, id_to_title)
    return load_or_train(_build)

art = load_artifacts()

st.title("🎬 Intelligent Movie Recommendation System")
tab1, tab2, tab3 = st.tabs(["Recommend for User", "Similar Movies", "Search & Explore"])

with tab1:
    st.subheader("User-based Recommendations")
    all_users = sorted(list(art.u_index.keys()))
    user = st.selectbox("Choose a userId", all_users if all_users else [1])
    k = st.slider("How many recommendations?", 5, 30, 10, 1)
    if st.button("Get Recommendations", key="rec_user"):
        recs = recommend_for_user(art, int(user), k=k)
        df = pd.DataFrame(recs, columns=["movieId","title"])
        st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Find Similar Movies")
    q = st.text_input("Search a movie title to pick a seed (e.g., Inception)", "")
    if q:
        root = download_movielens_if_needed(DATA_DIR)
        _, movies = load_movielens(root)
        hits = search_titles(movies, q, top=10)
        if not hits.empty:
            title_map = {f"{row['title']} (id={row['movieId']})": int(row['movieId']) for _,row in hits.iterrows()}
            choice = st.selectbox("Pick one:", list(title_map.keys()))
            mid = title_map[choice]
            k2 = st.slider("How many similar?", 5, 30, 10, 1, key="k_sim")
            if st.button("Get Similar", key="rec_sim"):
                sims = similar_items(art, mid, k=k2)
                df = pd.DataFrame(sims, columns=["movieId","title"])
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No local matches, try another query.")

with tab3:
    st.subheader("Quick dataset peek")
    root = download_movielens_if_needed(DATA_DIR)
    ratings, movies = load_movielens(root)
    st.write("Sample Ratings")
    st.dataframe(ratings.head(20), use_container_width=True)
    st.write("Sample Movies")
    st.dataframe(movies.head(20), use_container_width=True)

st.caption("Tip: run with `streamlit run streamlit_app.py`")
