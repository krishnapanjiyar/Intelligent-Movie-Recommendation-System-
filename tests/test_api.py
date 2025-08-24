import os, shutil, pandas as pd, numpy as np
import joblib, tempfile, types
import pytest

def build_mini_artifacts(tmpdir):
    # Minimal dataset with 5 users x 6 items
    from scipy.sparse import csr_matrix
    from src.models.svd_model import SVDRecommender
    from src.models.knn_model import ItemCosineKNN
    from src.recommender import Artifacts, save_artifacts, MODEL_PATH
    from src.config import ARTIFACT_DIR

    # Ratings matrix
    data = np.array([5,4,3,4,5,2,1,4,5,3,2,4,5,3,4,2], dtype=float)
    rows = np.array([0,0,1,1,2,2,2,3,3,3,4,4,1,2,3,4])
    cols = np.array([0,1,1,2,2,3,4,1,4,5,0,5,3,0,2,4])
    R = csr_matrix((data, (rows, cols)), shape=(5,6))

    u_index = {u:u for u in range(1,6)}  # userIds 1..5 -> 0..4? map directly for simplicity
    # Actually map raw ids 1..5 to indices 0..4
    u_index = {raw:(raw-1) for raw in range(1,6)}
    i_index = {mid:idx for idx, mid in enumerate([10,20,30,40,50,60])}
    id_to_title = {10:"The Matrix (1999)", 20:"Inception (2010)", 30:"Toy Story (1995)",
                   40:"The Dark Knight (2008)", 50:"Interstellar (2014)", 60:"Spirited Away (2001)"}

    svd = SVDRecommender(n_components=3, random_state=0).fit(R)
    knn = ItemCosineKNN(topk=3).fit(R)

    art = Artifacts(svd=svd, knn=knn, R=R, u_index=u_index, i_index=i_index,
                    id_to_title=id_to_title, users_sorted=sorted(u_index.keys()),
                    items_sorted=sorted(i_index.keys()))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_artifacts(art)

@pytest.fixture(autouse=True)
def env_isolated(tmp_path, monkeypatch):
    # Prepare isolated ARTIFACT_DIR and DATA_DIR
    art_dir = tmp_path / "artifacts"
    data_dir = tmp_path / "data"
    art_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.setenv("ARTIFACT_DIR", str(art_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    # Create minimal MovieLens-style CSVs
    ratings = pd.DataFrame({
        "userId":[1,1,2,2,3,3,3,4,4,4,5,5,2,3,4,5],
        "movieId":[10,20,20,30,30,40,50,20,50,60,10,60,40,10,30,50],
        "rating":[5,4,3,4,5,2,1,4,5,3,2,4,5,3,4,2],
        "timestamp":[0]*16
    })
    movies = pd.DataFrame({
        "movieId":[10,20,30,40,50,60],
        "title":["The Matrix (1999)", "Inception (2010)", "Toy Story (1995)",
                 "The Dark Knight (2008)", "Interstellar (2014)", "Spirited Away (2001)"],
        "genres":["Action|Sci-Fi","Action|Sci-Fi","Animation|Children","Action|Crime","Sci-Fi|Drama","Animation|Fantasy"]
    })
    ml_dir = data_dir / "ml-latest-small"
    ml_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(ml_dir / "ratings.csv", index=False)
    movies.to_csv(ml_dir / "movies.csv", index=False)

    # Build tiny artifacts so app loads without network
    build_mini_artifacts(tmp_path)
    yield

def test_endpoints():
    # Import after fixtures set env and artifacts exist
    from app import app
    client = app.test_client()

    # health
    r = client.get("/healthz")
    assert r.status_code == 200 and r.get_json().get("status") == "ok"

    # search
    r = client.get("/search?q=Matrix")
    assert r.status_code == 200
    data = r.get_json()
    assert any("Matrix" in row["title"] for row in data)

    # similar
    r = client.get("/similar/20?k=5")  # similar to Inception (2010)
    assert r.status_code == 200
    sim = r.get_json()
    assert isinstance(sim, list) and len(sim) > 0

    # recommend for user
    r = client.get("/recommend/user/1?k=5")
    assert r.status_code == 200
    recs = r.get_json()
    assert isinstance(recs, list) and len(recs) > 0

    # llm route (fallback, no key)
    r = client.post("/llm", json={"query":"Suggest action movies like Inception", "k":5})
    assert r.status_code == 200
    payload = r.get_json()
    assert "parsed" in payload and "results" in payload
    assert isinstance(payload["results"], list)
