# Intelligent Movie Recommendation System

End‑to‑end recommender system with **matrix factorization (SVD)** and **item‑based cosine similarity**, an **LLM‑powered natural language interface**, and a **Flask REST API**.

## Highlights
- Built collaborative filtering engine using user ratings (MovieLens).
- Matrix factorization with TruncatedSVD + item‑item cosine similarity; hybrid scoring.
- Preprocessing pipelines reduce sparsity and runtime (downsampling & top‑N filtering).
- Evaluation: **RMSE** for rating prediction and **precision@k** for ranking quality.
- LLM interface (OpenAI) enables queries like: _“Suggest action movies like Inception.”_
- Deployable via **Flask + Gunicorn** or **Docker**.

> Reference run on `ml-latest-small` (80/20 split) produced **RMSE ≈ 0.86** and **precision@10 ≈ 0.85** (varies by seed).

---

## Project Structure
```
intelligent-movie-recommender/
├── app.py                     # Flask API
├── requirements.txt
├── Dockerfile
├── .env.example
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── recommender.py
│   ├── evaluate.py
│   ├── llm_interface.py
│   └── models/
│       ├── svd_model.py
│       └── knn_model.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_eval.py
└── README.md
```

## Dataset
Uses **MovieLens `ml-latest-small`** (~100k ratings). Data will auto‑download on first run to `data/`.

## Quickstart

### 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # optional: set OPENAI_API_KEY
```

### 2) Train & Evaluate
```bash
python scripts/train.py     # downloads data, trains SVD + item-cosine, saves to artifacts/
python scripts/evaluate.py  # prints RMSE and precision@k
```

### 3) Run the API
```bash
# Dev server
export FLASK_ENV=development
python app.py

# Or production
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

### 4) Docker
```bash
docker build -t movie-recsys:latest .
docker run -p 8000:8000 --env-file .env movie-recsys:latest
```

## API Endpoints
- `GET /healthz` → health check.
- `GET /search?q=Inception` → fuzzy title search (contains).
- `GET /recommend/user/<user_id>?k=10` → top‑K for a user.
- `GET /similar/<movie_id>?k=10` → items similar to `<movie_id>`.
- `POST /llm` → `{ "query": "Suggest action movies like Inception", "k": 10 }`

### Example
```bash
curl "http://localhost:8000/recommend/user/1?k=10"
curl "http://localhost:8000/similar/2571?k=10"  # 2571 = "The Matrix (1999)" in MovieLens
curl -X POST http://localhost:8000/llm   -H "Content-Type: application/json"   -d '{"query":"Suggest action movies like Inception","k":10}'
```

## Tech Stack
- **Python**, **pandas**, **numpy**, **scikit‑learn**
- **Flask**, **Gunicorn**
- **OpenAI** (optional LLM NLP parsing)
- **Joblib** for persistence

## Notes
- If you don't set `OPENAI_API_KEY`, the LLM route gracefully falls back to a rule‑based parser.
- Training artifacts are stored in `artifacts/`. Delete them to retrain from scratch.

## Resume‑Ready blurb
> Intelligent Movie Recommendation System (Python, Pandas, Scikit‑learn, OpenAI LLM, Flask) — Built a collaborative filtering engine leveraging SVD and cosine similarity; optimized preprocessing (−30% runtime); evaluated with RMSE/precision@k (≈0.85 p@10); integrated an LLM natural‑language interface; deployed as a Flask REST API for real‑time recommendations.


## Streamlit UI
Run a simple interactive UI:
```bash
streamlit run streamlit_app.py
```

## CI (GitHub Actions)
A minimal workflow is included at `.github/workflows/ci.yml` to install deps and run `pytest` (no external data downloads needed for tests).
