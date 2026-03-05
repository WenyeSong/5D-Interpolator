import time
import tracemalloc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from fivedreg.data import prepare_datasets
from fivedreg.model import FiveDRegressor




DATASET_SIZES = [1000, 5000, 10000]
RANDOM_SEED = 26

MODEL_CONFIG = {
    "layers": [64, 32, 16],
    "lr": 1e-3,
    "epochs": 200,
}

np.random.seed(RANDOM_SEED)



# Dataset generation
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

def generate_dataset(n_samples, noise_std=0.1):
    X = np.random.uniform(-1.0, 1.0, size=(n_samples, 5))
    y_clean = ground_truth_function(X)

    # add noise
    noise = np.random.normal(0.0, noise_std, size=n_samples)
    y = y_clean + noise

    return X.astype(np.float32), y.astype(np.float32)


# profile

def profile_training(model, splits):
    tracemalloc.start()
    start_time = time.perf_counter()

    model.fit(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"]
    )
        # get training memory
    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return end_time - start_time, peak_memory / 1024 / 1024


def profile_prediction(model, X_sample):
    #trave memeory alloc
    tracemalloc.start()
    _ = model.predict(X_sample)
    _, peak_memory = tracemalloc.get_traced_memory()  # get wordt case
    tracemalloc.stop()
    return peak_memory / 1024 / 1024  # bytes to MB

# calc mse and r2
def evaluate_accuracy(model, X_val, y_true):

    y_pred = model.predict(X_val)  # already de-normalized
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2




# Warm-up function to eliminate startup overhead
def warmup():
    print("Warming up (eliminating startup overhead)...")
    X_warmup, y_warmup = generate_dataset(100)
    splits_warmup = prepare_datasets(X_warmup, y_warmup)
    model_warmup = FiveDRegressor(
        layers=MODEL_CONFIG["layers"],
        lr=MODEL_CONFIG["lr"],
        epochs=10
    )
    model_warmup.fit(
        splits_warmup["X_train"],
        splits_warmup["y_train"],
        splits_warmup["X_val"],
        splits_warmup["y_val"]
    )

    model_warmup.scaler = splits_warmup["scaler"]
    model_warmup.y_mean = splits_warmup["y_mean"]
    model_warmup.y_std  = splits_warmup["y_std"]

    _ = model_warmup.predict(splits_warmup["X_val"][:1])
    print("Warm-up completed.\n")

# Main experiment
def main():
    results = []
    
    # Warm-up to eliminate startup overhead
    warmup()

    for n_samples in DATASET_SIZES:
        print("=" * 60)
        print(f"Dataset size: {n_samples}")
        print("=" * 60)

        # generate dataset
        X, y = generate_dataset(n_samples)
        splits = prepare_datasets(X, y)

        # initialize model
        model = FiveDRegressor(
            layers=MODEL_CONFIG["layers"],
            lr=MODEL_CONFIG["lr"],
            epochs=MODEL_CONFIG["epochs"]
        )

        # training profiling
        train_time, train_mem = profile_training(model, splits)

        model.scaler = splits["scaler"]
        model.y_mean = splits["y_mean"]
        model.y_std = splits["y_std"]

        # worst case memory
        pred_mem = profile_prediction(model, splits["X_val"][:1])

        # accuracy evaluation using validation set
        mse, r2 = evaluate_accuracy(
            model,
            splits["X_val"],
            splits["y_val"]
        )

        results.append({
            "dataset_size": n_samples,
            "training_time_s": train_time,
            "training_peak_memory_mb": train_mem,
            "prediction_peak_memory_mb": pred_mem,
            "mse": mse,
            "r2": r2
        })

        # print per-dataset summary
        print(f"Training time: {train_time:.2f} s")
        print(f"Training peak memory: {train_mem:.2f} MB")
        print(f"Prediction peak memory: {pred_mem:.2f} MB")
        print(f"MSE: {mse:.6f}")
        print(f"R2: {r2:.4f}")
        print()

    # final summary
    print("\n========== Final Summary ==========")
    for r in results:
        print(
            f"{r['dataset_size']:>5} samples | "
            f"time: {r['training_time_s']:.2f}s | "
            f"train mem: {r['training_peak_memory_mb']:.2f}MB | "
            f"pred mem: {r['prediction_peak_memory_mb']:.2f}MB | "
            f"MSE: {r['mse']:.6f} | "
            f"R2: {r['r2']:.4f}"
        )
    
    # Plot comparison charts
    plot_results(results)

def plot_results(results):
    """Plot comparison charts for training time, memory usage, and accuracy metrics."""
    dataset_sizes = [r['dataset_size'] for r in results]
    train_times = [r['training_time_s'] for r in results]
    train_mems = [r['training_peak_memory_mb'] for r in results]
    pred_mems = [r['prediction_peak_memory_mb'] for r in results]
    mse_values = [r['mse'] for r in results]
    r2_values = [r['r2'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Time vs Dataset Size
    axes[0, 0].plot(dataset_sizes, train_times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Dataset Size', fontsize=12)
    axes[0, 0].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[0, 0].set_title('Training Time vs Dataset Size\n(Startup Overhead Eliminated)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(dataset_sizes)
    for i, (size, time) in enumerate(zip(dataset_sizes, train_times)):
        axes[0, 0].annotate(f'{time:.2f}s', (size, time), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Memory Usage vs Dataset Size
    axes[0, 1].plot(dataset_sizes, train_mems, 'o-', linewidth=2, markersize=8, 
                label='Training Memory', color='#A23B72')
    axes[0, 1].plot(dataset_sizes, pred_mems, 's-', linewidth=2, markersize=8, 
                label='Prediction Memory', color='#F18F01')
    axes[0, 1].set_xlabel('Dataset Size', fontsize=12)
    axes[0, 1].set_ylabel('Peak Memory (MB)', fontsize=12)
    axes[0, 1].set_title('Memory Usage vs Dataset Size\n(Startup Overhead Eliminated)', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(dataset_sizes)
    for i, (size, mem) in enumerate(zip(dataset_sizes, train_mems)):
        axes[0, 1].annotate(f'{mem:.2f}MB', (size, mem), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 3: MSE vs Dataset Size
    axes[1, 0].plot(dataset_sizes, mse_values, 'o-', linewidth=2, markersize=8, color='#C73E1D')
    axes[1, 0].set_xlabel('Dataset Size', fontsize=12)
    axes[1, 0].set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    axes[1, 0].set_title('MSE vs Dataset Size', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(dataset_sizes)
    for i, (size, mse) in enumerate(zip(dataset_sizes, mse_values)):
        axes[1, 0].annotate(f'{mse:.4f}', (size, mse), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 4: R² vs Dataset Size
    axes[1, 1].plot(dataset_sizes, r2_values, 'o-', linewidth=2, markersize=8, color='#06A77D')
    axes[1, 1].set_xlabel('Dataset Size', fontsize=12)
    axes[1, 1].set_ylabel('R² Score', fontsize=12)
    axes[1, 1].set_title('R² Score vs Dataset Size', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(dataset_sizes)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for i, (size, r2) in enumerate(zip(dataset_sizes, r2_values)):
        axes[1, 1].annotate(f'{r2:.4f}', (size, r2), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save to current directory
    output_path = 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPerformance comparison chart saved as '{output_path}'")
    
    # Also save to docs directory if it exists
    import os
    docs_static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'source', '_static', 'performance_comparison.png')
    if os.path.exists(os.path.dirname(docs_static_path)):
        plt.savefig(docs_static_path, dpi=300, bbox_inches='tight')
        print(f"Also saved to '{docs_static_path}' for documentation")
    
    plt.close()  # Close figure to free memory


if __name__ == "__main__":
    main()
