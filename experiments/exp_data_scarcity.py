"""
Experiment: Data Scarcity Ablation.
Evaluates model accuracy under severe training data scarcity (1%, 5%, 10%, 25%, 50%, 100%).
Demonstrates that embedding 1st Law physics enables PINN to generalize
far better than purely empirical data-driven models when failure records are scarce.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.preprocessor import BoilerDataPipeline
from src.models.baselines import RandomForestBaseline, StandardDeepMLP, evaluate_model_metrics
from src.models.pinn_model import BoilerPINN


def run_data_scarcity_experiment(fractions=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0], epochs=150):
    print("=" * 65)
    print("RUNNING DATA SCARCITY EXPERIMENT (1% TO 100% TRAINING DATA)")
    print("=" * 65)
    
    pipeline = BoilerDataPipeline()
    data = pipeline.get_train_val_test_splits()
    
    X_train_raw = data["X_train_raw"]
    y_train_raw = data["y_train_raw"]
    X_test_raw = data["X_test_raw"]
    y_test_raw = data["y_test_raw"]
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    deg_train = data["deg_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    n_total = len(X_train)
    records = []
    
    for frac in fractions:
        n_sub = max(int(n_total * frac), 20)
        print(f"\n[Scarcity {frac*100:5.1f}%] Subsample size: {n_sub:,} samples...")
        
        # Subsample indices
        idx = np.random.choice(n_total, n_sub, replace=False)
        sub_X_raw, sub_y_raw = X_train_raw[idx], y_train_raw[idx]
        sub_X, sub_y, sub_deg = X_train[idx], y_train[idx], deg_train[idx]
        
        # 1. Random Forest
        rf = RandomForestBaseline(n_estimators=50, max_depth=10)
        rf.fit(sub_X_raw, sub_y_raw)
        rf_metrics = evaluate_model_metrics(y_test_raw, rf.predict(X_test_raw))
        
        # 2. Standard Deep MLP
        mlp = StandardDeepMLP(in_features=4, hidden_dim=64)
        mlp.fit_model(sub_X, sub_y, epochs=epochs, lr=0.003, batch_size=min(64, n_sub))
        with torch.no_grad():
            mlp.eval()
            mlp_pred_scaled = mlp(X_test).numpy()
        mlp_pred = pipeline.inverse_transform_target(mlp_pred_scaled)
        mlp_metrics = evaluate_model_metrics(y_test_raw, mlp_pred)
        
        # 3. PINN (Ours)
        pinn = BoilerPINN(in_features=4, hidden_dim=64, w_data=1.0, w_phys=0.20, w_mono=0.05)
        pinn.fit_pinn(sub_X, sub_y, sub_deg, pipeline.scaler_X, pipeline.scaler_y, epochs=epochs, lr=0.003, verbose=False)
        diag = pinn.predict_with_diagnostics(X_test, pipeline.scaler_X, pipeline.scaler_y)
        pinn_metrics = evaluate_model_metrics(y_test_raw, diag["t_pred_k"])
        
        record = {
            "fraction": frac,
            "train_samples": n_sub,
            "RF_RMSE": rf_metrics["RMSE"],
            "MLP_RMSE": mlp_metrics["RMSE"],
            "PINN_RMSE": pinn_metrics["RMSE"],
            "PINN_Advantage_Pct": round(((mlp_metrics["RMSE"] - pinn_metrics["RMSE"]) / mlp_metrics["RMSE"]) * 100.0, 2)
        }
        records.append(record)
        print(f"  RF RMSE: {rf_metrics['RMSE']:.3f} K | MLP RMSE: {mlp_metrics['RMSE']:.3f} K | PINN RMSE: {pinn_metrics['RMSE']:.3f} K")
        
    df_scarcity = pd.DataFrame(records)
    print("\n" + "=" * 65)
    print("DATA SCARCITY RESULTS TABLE")
    print("=" * 65)
    print(df_scarcity.to_string(index=False))
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "results_data_scarcity.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[SAVED] Results saved to {out_path}")
    return df_scarcity


if __name__ == "__main__":
    run_data_scarcity_experiment(epochs=120)
