import numpy as np
from fivedreg.data import prepare_datasets, fill_missing_with_neighbors

def test_prepare_datasets_shapes():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    splits = prepare_datasets(X, y)

    assert splits["X_train"].shape[1] == 5
    assert len(splits["y_train"]) == splits["X_train"].shape[0]
    assert "scaler" in splits


def test_fill_missing_values():
    X = np.array([
        [1.0, np.nan, 3.0, 4.0, 5.0],
        [2.0, 2.0, 3.0, 4.0, 5.0],
    ])

    X_filled = fill_missing_with_neighbors(X)

    assert not np.isnan(X_filled).any()
