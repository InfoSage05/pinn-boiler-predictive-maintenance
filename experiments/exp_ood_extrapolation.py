"""
Experiment: Out-of-Distribution (OOD) Extrapolation Benchmark.
Evaluates model behavior on operational regimes outside the training distribution:
High-load conditions (Fuel_Mdot >= 3.5 kg/s, Water_Mdot >= 10.5 kg/s).
Tests whether pure empirical models produce unphysical predictions,
while PINN remains constrained by the 1st Law of Thermodynamics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.preprocessor import BoilerDataPipeline
from src.models.baselines import (
    PhysicsOnlyBaseline,
    PolynomialRidgeBaseline,
    RandomForestBaseline,
    StandardDeepMLP,
    evaluate_model_metrics
)
from src.models.pinn_model import BoilerPINN
from src.physics.boiler_thermo import BoilerThermodynamics


def run_ood_experiment(epochs=180):
    print("=" * 70)
    print("RUNNING OUT-OF-DISTRIBUTION (OOD) EXTRAPOLATION BENCHMARK")
    print("=" * 70)
    
    pipeline = BoilerDataPipeline()
    data = pipeline.get_train_val_test_splits()
    thermo = BoilerThermodynamics()
    
    X_train_raw = data["X_train_raw"]
    y_train_raw = data["y_train_raw"]
    X_ood_raw = data["X_ood_raw"]
    y_ood_raw = data["y_ood_raw"]
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    deg_train = data["deg_train"]
    X_ood = data["X_ood"]
    
    print(f"Train samples (In-Distribution): {len(X_train):,}")
    print(f"OOD Evaluation samples (Extreme Peak Load): {len(X_ood):,}")
    
    results = {}
    
    # 1. Physics-Only
    print("\nEvaluating Model 0 (Physics-Only Analytical)...")
    m0 = PhysicsOnlyBaseline()
    pred_m0 = m0.predict(X_ood_raw)
    results["Physics-Only ODE"] = evaluate_model_metrics(y_ood_raw, pred_m0)
    
    # 2. Polynomial Ridge
    print("Evaluating Model 1 (Polynomial Ridge)...")
    m1 = PolynomialRidgeBaseline(degree=2)
    m1.fit(X_train_raw, y_train_raw)
    pred_m1 = m1.predict(X_ood_raw)
    results["Polynomial Ridge"] = evaluate_model_metrics(y_ood_raw, pred_m1)
    
    # 3. Random Forest
    print("Evaluating Model 2 (Random Forest)...")
    m2 = RandomForestBaseline(n_estimators=80, max_depth=12)
    m2.fit(X_train_raw[:6000], y_train_raw[:6000])
    pred_m2 = m2.predict(X_ood_raw)
    results["Random Forest"] = evaluate_model_metrics(y_ood_raw, pred_m2)
    
    # 4. Standard Deep MLP
    print("Evaluating Model 3 (Standard Deep MLP - Data Only)...")
    m3 = StandardDeepMLP(in_features=4, hidden_dim=64)
    m3.fit_model(X_train, y_train, epochs=epochs, lr=0.003, batch_size=256)
    with torch.no_grad():
        m3.eval()
        pred_m3_scaled = m3(X_ood).numpy()
    pred_m3 = pipeline.inverse_transform_target(pred_m3_scaled)
    results["Deep MLP (Data-Only)"] = evaluate_model_metrics(y_ood_raw, pred_m3)
    
    # 5. PINN (Ours)
    print("Evaluating Model 5 (Physics-Informed Neural Network)...")
    m5 = BoilerPINN(in_features=4, hidden_dim=64, w_data=1.0, w_phys=0.25, w_mono=0.05)
    m5.fit_pinn(X_train, y_train, deg_train, pipeline.scaler_X, pipeline.scaler_y, epochs=epochs, lr=0.003, verbose=False)
    diag = m5.predict_with_diagnostics(X_ood, pipeline.scaler_X, pipeline.scaler_y)
    pred_m5 = diag["t_pred_k"]
    results["PINN (Proposed)"] = evaluate_model_metrics(y_ood_raw, pred_m5)
    
    df_ood = pd.DataFrame(results).T
    print("\n" + "=" * 70)
    print("OUT-OF-DISTRIBUTION (OOD) EXTRAPOLATION RESULTS")
    print("=" * 70)
    print(df_ood.to_string())
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "results_ood.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Results saved to {out_path}")
    return df_ood


if __name__ == "__main__":
    run_ood_experiment(epochs=150)
