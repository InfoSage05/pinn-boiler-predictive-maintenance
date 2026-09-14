"""
Experiment: Sensor Noise Robustness Benchmark.
Tests model resilience when physical sensor readings are contaminated
with Gaussian measurement noise (0%, 2%, 5%, 10%, 20%).
Demonstrates that the physics regularization acts as a denoising filter.
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


def run_noise_robustness_experiment(noise_levels=[0.0, 0.02, 0.05, 0.10, 0.20], epochs=120):
    print("=" * 70)
    print("RUNNING SENSOR NOISE ROBUSTNESS BENCHMARK")
    print("=" * 70)
    
    pipeline = BoilerDataPipeline()
    data = pipeline.get_train_val_test_splits()
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    deg_train = data["deg_train"]
    X_test = data["X_test"]
    y_test_raw = data["y_test_raw"]
    
    # Train models on clean training set
    print("\nPre-training models on clean baseline data...")
    rf = RandomForestBaseline(n_estimators=60, max_depth=10)
    rf.fit(data["X_train_raw"][:6000], data["y_train_raw"][:6000])
    
    mlp = StandardDeepMLP(in_features=4, hidden_dim=64)
    mlp.fit_model(X_train, y_train, epochs=epochs, lr=0.003, batch_size=256)
    
    pinn = BoilerPINN(in_features=4, hidden_dim=64, w_data=1.0, w_phys=0.20, w_mono=0.05)
    pinn.fit_pinn(X_train, y_train, deg_train, pipeline.scaler_X, pipeline.scaler_y, epochs=epochs, lr=0.003, verbose=False)
    
    records = []
    
    for noise_std in noise_levels:
        print(f"\nEvaluating with {noise_std*100:4.1f}% Gaussian Sensor Noise...")
        
        # Add noise to test inputs
        noise = torch.randn_like(X_test) * noise_std
        X_test_noisy = X_test + noise
        X_test_noisy_raw = pipeline.inverse_transform_features(X_test_noisy)
        
        # 1. Random Forest
        rf_pred = rf.predict(X_test_noisy_raw)
        rf_m = evaluate_model_metrics(y_test_raw, rf_pred)
        
        # 2. Standard Deep MLP
        with torch.no_grad():
            mlp.eval()
            mlp_pred_scaled = mlp(X_test_noisy).numpy()
        mlp_pred = pipeline.inverse_transform_target(mlp_pred_scaled)
        mlp_m = evaluate_model_metrics(y_test_raw, mlp_pred)
        
        # 3. PINN
        diag = pinn.predict_with_diagnostics(X_test_noisy, pipeline.scaler_X, pipeline.scaler_y)
        pinn_m = evaluate_model_metrics(y_test_raw, diag["t_pred_k"])
        
        record = {
            "noise_level_pct": round(noise_std * 100.0, 1),
            "RF_RMSE": round(rf_m["RMSE"], 3),
            "MLP_RMSE": round(mlp_m["RMSE"], 3),
            "PINN_RMSE": round(pinn_m["RMSE"], 3),
            "PINN_Denoising_Gain_Pct": round(((mlp_m["RMSE"] - pinn_m["RMSE"]) / mlp_m["RMSE"]) * 100.0, 2)
        }
        records.append(record)
        print(f"  RF RMSE: {rf_m['RMSE']:.3f} K | MLP RMSE: {mlp_m['RMSE']:.3f} K | PINN RMSE: {pinn_m['RMSE']:.3f} K")
        
    df_noise = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("NOISE ROBUSTNESS RESULTS")
    print("=" * 70)
    print(df_noise.to_string(index=False))
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "results_noise.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[SAVED] Results saved to {out_path}")
    return df_noise


if __name__ == "__main__":
    run_noise_robustness_experiment(epochs=120)
