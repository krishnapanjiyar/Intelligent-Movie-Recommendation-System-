import os

DATA_DIR = os.getenv("DATA_DIR", "data")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

# Recommender weights
ALPHA = float(os.getenv("ALPHA", "0.6"))  # weight for SVD vs KNN hybrid

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Evaluation
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# Training
MIN_USER_RATINGS = int(os.getenv("MIN_USER_RATINGS", "5"))
MIN_ITEM_RATINGS = int(os.getenv("MIN_ITEM_RATINGS", "5"))
KNN_TOPK = int(os.getenv("KNN_TOPK", "50"))
SVD_COMPONENTS = int(os.getenv("SVD_COMPONENTS", "100"))
