import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(filepath):
    """
    Load dataset from a pickle file.

    Args:
        filepath: Path to the pickle file containing the dataset

    Returns:
        tuple: (X, y) where X is the input features (N, 5) and y is the target values (N,)

    Raises:
        ValueError: If the dataset format is invalid
    """
    with open(filepath, "rb") as f:  # .pkl is binary, to read binary: rb
        data = pickle.load(f)

    X = np.asarray(data["X"])
    y = np.asarray(data["y"])

    # validate dimensions
    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError(f"Expected X of shape (N, 5), but here it is {X.shape}")

    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(f"Expected y of length N={len(X)}, but here it is {y.shape}")

    # handle missing values
    if np.isnan(X).any():
        X = fill_missing_with_neighbors(X)

    return X, y


# fill with closest value
def fill_missing_with_neighbors(X):
    """
    Fill missing values in the dataset using neighbor values.

    Args:
        X: Input features array that may contain NaN values

    Returns:
        numpy.ndarray: Array with missing values filled
    """
    X = X.copy()
    n_samples, n_features = X.shape
    col_means = np.nanmean(X, axis=0) # average of each column(feature), skip Nan

    for i in range(n_samples):
        for j in range(n_features):
            if np.isnan(X[i, j]):
                # prefer previous value if available, including last value
                if i > 0 and not np.isnan(X[i - 1, j]):
                    X[i, j] = X[i - 1, j]
                # otherwise, use next value, including first value
                elif i < n_samples - 1 and not np.isnan(X[i + 1, j]):
                    X[i, j] = X[i + 1, j]
                else:
                    #set mean of the whole column
                    X[i, j] = col_means[j]
    return X



def prepare_datasets(X, y, test_size=0.2, val_size=0.1, random_state=2):
    """
    Prepare train, validation, and test datasets with standardization.

    Args:
        X: Input features, shape (N, 5)
        y: Target values, shape (N,)
        test_size: Proportion of data to use for testing. Defaults to 0.2.
        val_size: Proportion of data to use for validation. Defaults to 0.1.
        random_state: Random seed for reproducibility. Defaults to 2.

    Returns:
        dict: Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test,
              scaler, y_mean, and y_std
    """
    # split to train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # split train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, shuffle=True
    )

    # standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    # Standardize y
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "y_mean": y_mean,
        "y_std": y_std,
    }
