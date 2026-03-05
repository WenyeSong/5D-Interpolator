from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, pickle
import io
import torch
import time
import tracemalloc
from contextlib import redirect_stdout
import numpy as np
from fivedreg.data import prepare_datasets
from fivedreg.model import FiveDRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],  # allow cross-domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path of this file 
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")
DATA_PATH = os.path.join(DATA_DIR, "uploaded_dataset.pkl")

os.makedirs(DATA_DIR, exist_ok=True)


# GET health
@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
        dict: Status message indicating the service is running
    """
    return {"status": "ok"}


# POST upload

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset file in pickle format.

    Args:
        file: Pickle file containing dataset with 'X' (N×5) and 'y' keys

    Returns:
        dict: Dataset information including data points, features, target range, and preview

    Raises:
        HTTPException: If file format is invalid or dataset structure is incorrect
    """
    print("filename:", file.filename)
    print("content_type:", file.content_type)

    # content-type check
    if file.content_type not in [
        "application/octet-stream",
        "application/x-pickle",
        "application/python-pickle"
    ]:
        raise HTTPException(status_code=400, detail="File must be a pickle (.pkl)")

    # read file
    contents = await file.read()

    # save to disk
    with open(DATA_PATH, "wb") as f:
        f.write(contents)

    # load pickle safely from disk
    try:
        with open(DATA_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print("pickle load error:", e)
        raise HTTPException(status_code=400, detail=f"Invalid dataset format: {e}")

    # validate dataset
    if not isinstance(data, dict) or "X" not in data or "y" not in data:
        raise HTTPException(400, "Pickle must contain keys 'X' and 'y'")

    X = np.asarray(data["X"])
    y = np.asarray(data["y"])

    if X.ndim != 2 or X.shape[1] != 5:
        raise HTTPException(400, "X must have shape (N,5)")

    if y.ndim != 1 or len(y) != len(X):
        raise HTTPException(400, "y must have same length as X")

    # show preview
    # show preview: X + y (last column is y)
    preview = [
    [float(v) for v in X[i]] + [float(y[i])]
    for i in range(min(5, len(X)))
    ]


    print("DEBUG preview row length:", len(preview[0]))
    print("DEBUG preview row:", preview[0])


    target_min = float(y.min())
    target_max = float(y.max())
    target_range = [target_min, target_max]

    return {
    "message": "Dataset uploaded successfully!",
    "data_points": int(X.shape[0]),          # = data points
    "number_features": int(X.shape[1]),    # feature count
    "shape_X": list(X.shape),
    "target_min": target_min,
    "target_max": target_max,
    "target_range": target_range,
    "preview": preview
    }

# POST train

class TrainRequest(BaseModel):
    layers: List[int] = [64, 32, 16]
    lr: float = 1e-3
    epochs: int = 200


@app.post("/train")
def train(req: TrainRequest):  # to accept input
    """
    Train the neural network model with configurable hyperparameters.

    Args:
        req: Training request containing layers, learning rate, and epochs

    Returns:
        dict: Training results including message, hyperparameters, logs, and early_stopped flag

    Raises:
        HTTPException: If no dataset has been uploaded
    """
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]

    splits = prepare_datasets(X, y)

    # create configurable model
    model = FiveDRegressor(
        layers=req.layers,
        lr=req.lr,
        epochs=req.epochs
    )

    f = io.StringIO()
    with redirect_stdout(f):   # collect stdout
        fit_result = model.fit(               # train
            splits["X_train"],
            splits["y_train"],
            splits["X_val"],
            splits["y_val"]
    )
    
    logs = f.getvalue().splitlines()

    # attach normalization
    model.scaler = splits["scaler"]
    model.y_mean = splits["y_mean"]
    model.y_std = splits["y_std"]

    # save
    model.save(MODEL_PATH)

    return {
        "message": "Training completed.",
        "layers": req.layers,
        "lr": req.lr,
        "epochs": req.epochs,
        "logs": logs,
        "early_stopped": fit_result["early_stopped"],
        "stopped_epoch": fit_result["stopped_epoch"],
        "best_epoch": fit_result["best_epoch"],
        "best_val_loss": fit_result["best_val_loss"],
        "total_epochs": fit_result["total_epochs"]
    }


# POST predict

class PredictRequest(BaseModel):
    X: list  # must be length 5

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict a scalar output for a given 5D input.

    Args:
        req: Prediction request containing X (list of 5 values)

    Returns:
        dict: Predicted value (y_pred)

    Raises:
        HTTPException: If model has not been trained yet
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    # load model

    model = FiveDRegressor()
    model.load(MODEL_PATH)

    X = np.array(req.X, dtype=np.float32).reshape(1, -1)

    y_pred = model.predict(X)

    return {"y_pred": float(y_pred[0])}


# POST benchmark

class BenchmarkRequest(BaseModel):
    layers: List[int] = [64, 32, 16]
    lr: float = 1e-3
    epochs: int = 200
    dataset_sizes: List[int] = [1000, 5000, 10000]

def ground_truth_function(X):
    x1, x2, x3, x4, x5 = X.T
    y = (
        1.2 * x1
        + 0.8 * x2
        - 0.5 * x3
        + 0.3 * x4
        + 0.2 * x5
        + 0.5 * x1 * x2
    )
    return y

def generate_benchmark_dataset(n_samples, noise_std=0.1, random_seed=26):
    np.random.seed(random_seed)
    X = np.random.uniform(-1.0, 1.0, size=(n_samples, 5))
    y_clean = ground_truth_function(X)
    noise = np.random.normal(0.0, noise_std, size=n_samples)
    y = y_clean + noise
    return X.astype(np.float32), y.astype(np.float32)

def profile_training_benchmark(model, splits):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    model.fit(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"]
    )
    
    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return end_time - start_time, peak_memory / 1024 / 1024

def profile_prediction_benchmark(model, X_sample):
    tracemalloc.start()
    _ = model.predict(X_sample)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_memory / 1024 / 1024

def evaluate_accuracy_benchmark(model, X_val, y_true):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def warmup_benchmark(layers, lr, epochs):
    X_warmup, y_warmup = generate_benchmark_dataset(100, random_seed=26)
    splits_warmup = prepare_datasets(X_warmup, y_warmup)
    model_warmup = FiveDRegressor(layers=layers, lr=lr, epochs=10)
    _ = model_warmup.fit(
        splits_warmup["X_train"],
        splits_warmup["y_train"],
        splits_warmup["X_val"],
        splits_warmup["y_val"]
    )
    model_warmup.scaler = splits_warmup["scaler"]
    model_warmup.y_mean = splits_warmup["y_mean"]
    model_warmup.y_std = splits_warmup["y_std"]
    _ = model_warmup.predict(splits_warmup["X_val"][:1])

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    """
    Run performance benchmarking across different dataset sizes.

    Args:
        req: Benchmark request containing model config and dataset sizes

    Returns:
        dict: Benchmark results with training time, memory usage, and accuracy metrics
    """
    try:
        results = []
        
        warmup_benchmark(req.layers, req.lr, req.epochs)
        
        for n_samples in req.dataset_sizes:
            X, y = generate_benchmark_dataset(n_samples, random_seed=26)
            splits = prepare_datasets(X, y)
            
            model = FiveDRegressor(
                layers=req.layers,
                lr=req.lr,
                epochs=req.epochs
            )
            
            train_time, train_mem = profile_training_benchmark(model, splits)
            
            model.scaler = splits["scaler"]
            model.y_mean = splits["y_mean"]
            model.y_std = splits["y_std"]
            
            pred_mem = profile_prediction_benchmark(model, splits["X_val"][:1])
            
            mse, r2 = evaluate_accuracy_benchmark(
                model,
                splits["X_val"],
                splits["y_val"]
            )
            
            results.append({
                "dataset_size": n_samples,
                "training_time_s": float(train_time),
                "training_peak_memory_mb": float(train_mem),
                "prediction_peak_memory_mb": float(pred_mem),
                "mse": float(mse),
                "r2": float(r2)
            })
        
        return {
            "message": "Benchmark completed.",
            "results": results
        }
    except Exception as e:
        import traceback
        error_detail = str(e)
        traceback_str = traceback.format_exc()
        print(f"Benchmark error: {error_detail}")
        print(traceback_str)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {error_detail}")
