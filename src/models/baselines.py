"""
Comparative Benchmark Model Zoo.
Implements:
- Model 0: Physics-Only Baseline (Zero-data analytical 1st Law ODE)
- Model 1: Polynomial Ridge Regression
- Model 2: Random Forest Regressor
- Model 3: Standard Deep MLP (Pure Data Loss, No Physics)
- Model 4: Recurrent LSTM Neural Network
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.physics.boiler_thermo import BoilerThermodynamics


class PhysicsOnlyBaseline:
    """
    Model 0: Pure First-Principles Analytical Physics.
    Uses 1st Law steady-state heat exchanger equations.
    Requires ZERO training data.
    """
    def __init__(self):
        self.thermo = BoilerThermodynamics()
        self.name = "Physics-Only (Analytical 1st Law)"

    def predict(self, X_raw):
        """
        X_raw columns: [Fuel_Mdot, Tair, Treturn, Water_Mdot]
        """
        preds = []
        for row in X_raw:
            fuel_mdot, t_air, t_return, water_mdot = row[0], row[1], row[2], row[3]
            t_pred = self.thermo.compute_steady_state_t_supply(
                fuel_mdot=fuel_mdot,
                t_air=t_air,
                t_return=t_return,
                water_mdot=water_mdot,
                r_fouling=0.0
            )
            preds.append(t_pred)
        return np.array(preds).reshape(-1, 1)


class PolynomialRidgeBaseline:
    """
    Model 1: Degree-2 Polynomial Features + Ridge Regularization.
    """
    def __init__(self, degree=2, alpha=1.0):
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("ridge", Ridge(alpha=alpha))
        ])
        self.name = "Polynomial Ridge Regression"

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class RandomForestBaseline:
    """
    Model 2: Ensemble Random Forest Regressor.
    """
    def __init__(self, n_estimators=100, max_depth=12, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
        self.name = "Random Forest Regressor"

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class StandardDeepMLP(nn.Module):
    """
    Model 3: Standard Deep Neural Network trained purely on MSE data loss.
    Has NO knowledge of thermodynamics or energy balance.
    """
    def __init__(self, in_features=4, hidden_dim=64, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_features)
        )
        self.name = "Standard Deep MLP (Data-Only)"

    def forward(self, x):
        return self.net(x)

    def fit_model(self, X_train, y_train, X_val=None, y_val=None, epochs=250, lr=0.005, batch_size=256):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.train()
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()


class LSTMBaseline(nn.Module):
    """
    Model 4: Recurrent LSTM Sequence Predictor.
    """
    def __init__(self, in_features=4, hidden_dim=48, num_layers=2, out_features=1):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_features)
        self.name = "LSTM Recurrent Network"

    def forward(self, x):
        # If x is 2D (batch, features), add sequence dim: (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        # Take last time step
        pred = self.fc(out[:, -1, :])
        return pred

    def fit_model(self, X_train, y_train, epochs=200, lr=0.005, batch_size=256):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.train()
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()


def evaluate_model_metrics(y_true, y_pred):
    """Computes standard regression metrics."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
