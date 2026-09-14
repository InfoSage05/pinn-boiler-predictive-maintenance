"""
Experiment: Physics Loss Weight Ablation Study.
Quantifies the impact of the thermodynamic loss weight (lambda_physics)
on prediction accuracy and physical energy conservation residual.
Sweeps lambda_phys in [0.0, 0.01, 0.05, 0.15, 0.50, 1.0].
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.preprocessor import BoilerDataPipeline
from src.models.pinn_model import BoilerPINN
from src.models.baselines import evaluate_model_metrics


def run_physics_ablation_experiment(lambdas=[0.0, 0.01, 0.05, 0.15, 0.50, 1.0], epochs=120):
    print("=" * 70)
    print("RUNNING PHYSICS LOSS WEIGHT (LAMBDA_PHYS) ABLATION STUDY")
    print("=" * 70)
    
    pipeline = BoilerDataPipeline()
    data = pipeline.get_train_val_test_splits()
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    deg_train = data["deg_train"]
    X_test = data["X_test"]
    y_test_raw = data["y_test_raw"]
    
    records = []
    
    for l_p in lambdas:
        print(f"\nTraining PINN with lambda_phys = {l_p:.2f}...")
        pinn = BoilerPINN(in_features=4, hidden_dim=64, w_data=1.0, w_phys=l_p, w_mono=0.05)
        pinn.fit_pinn(X_train, y_train, deg_train, pipeline.scaler_X, pipeline.scaler_y, epochs=epochs, lr=0.003, verbose=False)
        
        diag = pinn.predict_with_diagnostics(X_test, pipeline.scaler_X, pipeline.scaler_y)
        metrics = evaluate_model_metrics(y_test_raw, diag["t_pred_k"])
        mean_res = float(np.mean(diag["physics_residual_kw"]))
        
        record = {
            "lambda_phys": l_p,
            "RMSE": round(metrics["RMSE"], 3),
            "MAE": round(metrics["MAE"], 3),
            "R2": round(metrics["R2"], 4),
            "Physics_Residual_kW": round(mean_res, 2)
        }
        records.append(record)
        print(f"  RMSE: {metrics['RMSE']:.3f} K | MAE: {metrics['MAE']:.3f} K | Residual: {mean_res:.2f} kW")
        
    df_ablation = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("PHYSICS LOSS WEIGHT SENSITIVITY SUMMARY")
    print("=" * 70)
    print(df_ablation.to_string(index=False))
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "results_physics_ablation.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[SAVED] Results saved to {out_path}")
    return df_ablation


if __name__ == "__main__":
    run_physics_ablation_experiment(epochs=100)
