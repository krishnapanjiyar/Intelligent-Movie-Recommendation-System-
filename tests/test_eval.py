from src.evaluate import precision_at_k

def test_precision_at_k_basic():
    recs = [1,2,3,4,5]
    truth = {2,5,9}
    assert abs(precision_at_k(recs, truth, k=5) - 0.4) < 1e-6
