from fastapi.testclient import TestClient
from fivedreg.main import app

client = TestClient(app)

def test_benchmark_endpoint():
    r = client.post(
        "/benchmark",
        json={
            "layers": [8, 4],
            "lr": 0.001,
            "epochs": 1,
            "dataset_sizes": [100]
        }
    )

    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) == 1
    assert "training_time_s" in body["results"][0]
