import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ConfigurableMLP(nn.Module):  

    def __init__(self, layers):
        """
        Initialize a configurable multi-layer perceptron.

        Args:
            layers: list of hidden layer sizes, e.g. [32, 32] or [64, 32, 16]; so that can be read by "model = FiveDRegressor(layers=[64, 32, 16])
        """
        
        #layers: list of hidden layer sizes, e.g. [32, 32] or [64, 32, 16]; so that can be read by "model = FiveDRegressor(layers=[64, 32, 16])

        super().__init__()  # inherits nn.Module, including train() evel() parameters() etc.

        input_dim = 5
        output_dim = 1

        net_layers = []
        prev_dim = input_dim  # initial layer


        # build and connect hidden layers iteratevely
        for h in layers:
            net_layers.append(nn.Linear(prev_dim, h))
            net_layers.append(nn.ReLU())
            prev_dim = h

        net_layers.append(nn.Linear(prev_dim, output_dim))  # final output layer

        self.model = nn.Sequential(*net_layers) # arrange all the layers to a complete NN

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor from the model
        """
        return self.model(x)
        

class FiveDRegressor:

    def __init__(self, layers = [64,32,16], lr = 1e-3, epochs = 300):
        """
        Initialize the 5D regressor model.

        Args:
            layers: List of hidden layer sizes. Defaults to [64, 32, 16].
            lr: Learning rate. Defaults to 1e-3.
            epochs: Number of training epochs. Defaults to 300.
        """
        self.device = torch.device("cpu") # run on CPU
        self.model = ConfigurableMLP(layers).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.layers = layers


    def fit(self, X_train, y_train, X_val = None, y_val = None):  # need val?
        """
        Train the model on the provided dataset.

        Args:
            X_train: Training input features, shape (N, 5)
            y_train: Training target values, shape (N,)
            X_val: Optional validation input features, shape (M, 5)
            y_val: Optional validation target values, shape (M,)

        Returns:
            dict: Dictionary containing early_stopped (bool), stopped_epoch (int), 
                  best_epoch (int), best_val_loss (float), and total_epochs (int)
        """
        #numpy to tensor
        X_train = torch.tensor(X_train, dtype = torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype = torch.float32).view(-1,1).to(self.device)  # view change (N,) to (N,1)

        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()  

        patience = 3
        min_delta = 0.0001
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False
        stopped_epoch = None
        best_epoch = 0
        check_interval = 20

        for epoch in range(self.epochs):
            optimizer.zero_grad()   # reset gradient
            pred = self.model(X_train)  # call ConfigurableMLP 
            loss = loss_fn(pred, y_train)
            loss.backward()  # calc grad for each w
            optimizer.step() # update w
            if epoch % check_interval == 0:
                log = f"Epoch {epoch}: Train={loss.item():.4f}"
                if X_val is not None:
                    with torch.no_grad():
                        pred_val = self.model(X_val)
                        val_loss = loss_fn(pred_val, y_val)
                    log += f", Val={val_loss.item():.4f}"
                    print(log)
                    
                    if val_loss.item() < best_val_loss - min_delta:
                        best_val_loss = val_loss.item()
                        best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            early_stopped = True
                            stopped_epoch = epoch
                            epochs_without_improvement = patience * check_interval
                            print(f"Early stopping at epoch {epoch} (no improvement for {patience} checks = {epochs_without_improvement} epochs, best at epoch {best_epoch}, best val loss: {best_val_loss:.6f})")
                            break
                else:
                    print(log)

        return {
            "early_stopped": early_stopped,
            "stopped_epoch": stopped_epoch if early_stopped else self.epochs - 1,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss) if X_val is not None else None,
            "total_epochs": self.epochs
        }


    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input features, shape (N, 5)

        Returns:
            numpy.ndarray: Predicted values, shape (N,)
        """
        X = self.scaler.transform(X)
        # to tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # inference
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        result = pred.cpu().numpy().flatten()

        result_norm = result * self.y_std + self.y_mean  # inverse norm
        return result_norm



    # save weight/bias and use

    def save(self, path="model.pt"):
        """
        Save the model to disk.

        Args:
            path: Path to save the model file. Defaults to "model.pt".
        """
        torch.save({
            "state_dict": self.model.state_dict(),   # all paras
            "scaler": self.scaler,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "layers": self.layers
        }, path)

    def load(self, path="model.pt"):
        """
        Load the model from disk.

        Args:
            path: Path to the model file. Defaults to "model.pt".
        """
        saved = torch.load(path, map_location="cpu", weights_only=False)

        self.layers = saved["layers"]       # load architecture
        self.model = ConfigurableMLP(self.layers).to(self.device)   # build
        self.model.load_state_dict(saved["state_dict"])  # load paras
        self.scaler = saved["scaler"]
        self.y_mean = saved["y_mean"]
        self.y_std = saved["y_std"]
