import numpy as np
from fivedreg.model import FiveDRegressor, ConfigurableMLP
import torch

def test_configurable_mlp_forward():
    model = ConfigurableMLP(layers=[8, 4])
    x = torch.randn(10, 5)
    y = model(x)

    assert y.shape == (10, 1)


def test_fivedregressor_fit_and_predict():
    X = np.random.randn(50, 5)
    y = np.random.randn(50)

    model = FiveDRegressor(layers=[16, 8], epochs=2)

    # fake scaler & normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model.scaler = scaler
    model.y_mean = y.mean()
    model.y_std = y.std() + 1e-8

    model.fit(Xs, (y - model.y_mean) / model.y_std)

    preds = model.predict(X[:5])
    assert preds.shape == (5,)
