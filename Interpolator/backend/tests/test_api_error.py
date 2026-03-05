from fastapi.testclient import TestClient
from fivedreg.main import app
import numpy as np
import pickle

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_upload_and_train():

    # generte temp dataset
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    pickle.dump({"X": X, "y": y}, open("temp.pkl", "wb"))

    with open("temp.pkl", "rb") as f:
        r = client.post("/upload", files={"file": ("temp.pkl", f, "application/octet-stream")})
    assert r.status_code == 200

    r = client.post("/train", json={"layers": [16,8], "lr":0.001, "epochs":2})
    assert r.status_code == 200

    print("UPLOAD RESPONSE:", r.json())


def test_predict():
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    pickle.dump({"X": X, "y": y}, open("temp.pkl", "wb"))

    with open("temp.pkl", "rb") as f:
        r = client.post("/upload", files={"file": ("temp.pkl", f, "application/octet-stream")})
    assert r.status_code == 200

    r = client.post("/train", json={"layers": [16,8], "lr": 0.001, "epochs": 1})
    assert r.status_code == 200

    r = client.post("/predict", json={"X": [0,0,0,0,0]})
    assert r.status_code == 200
    assert "y_pred" in r.json()
