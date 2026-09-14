"""
Master Benchmark Runner: Comprehensive 6-Model Comparative Evaluation.
Evaluates:
- Model 0: Physics-Only (Analytical 1st Law ODE)
- Model 1: Polynomial Ridge Regression
- Model 2: Random Forest Regressor
- Model 3: Standard Deep MLP (Pure Data Loss)
- Model 4: Recurrent LSTM Neural Network
- Model 5: Physics-Informed Neural Network (PINN)

Computes Test RMSE (K), MAE (K), R², Physics Energy Balance Residual Error (kW), and Latency.
Saves results to experiments/results_benchmark_leaderboard.json.
"""

import os
import sys
import time
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
    LSTMBaseline,
    evaluate_model_metrics
)
from src.models.pinn_model import BoilerPINN
from src.physics.boiler_thermo import BoilerThermodynamics


def evaluate_physics_residual_error(X_raw, y_pred_k, thermo):
    """
    Computes mean energy balance violation (in kW) across predictions:
    Residual = |Q_water_predicted - Q_effective_expected|
    """
    residuals = []
    for i in range(len(X_raw)):
        fuel_mdot, t_air, t_return, water_mdot = X_raw[i]
        t_supply_pred = y_pred_k[i, 0]
        
        q_clean = thermo.compute_clean_q_water(fuel_mdot, t_air)
        q_absorbed = water_mdot * thermo.water_cp * (t_supply_pred - t_return)
        res = abs(q_clean - q_absorbed)
        residuals.append(res)
    return float(np.mean(residuals))


def run_full_benchmark(epochs_nn=250):
    print("=" * 70)
    print("RUNNING COMPREHENSIVE 6-MODEL BENCHMARK LADDER")
    print("=" * 70)
    
    # 1. Load Data
    pipeline = BoilerDataPipeline()
    data = pipeline.get_train_val_test_splits()
    thermo = BoilerThermodynamics()
    
    X_train_raw = data["X_train_raw"]
    y_train_raw = data["y_train_raw"]
    X_test_raw = data["X_test_raw"]
    y_test_raw = data["y_test_raw"]
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    deg_train = data["deg_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    results = {}
    
    # -------------------------------------------------------------
    # Model 0: Physics-Only (Zero Data)
    # -------------------------------------------------------------
    print("\n[1/6] Evaluating Model 0: Physics-Only (Analytical 1st Law)...")
    m0 = PhysicsOnlyBaseline()
    t0 = time.time()
    y_pred_m0 = m0.predict(X_test_raw)
    lat_m0 = ((time.time() - t0) / len(X_test_raw)) * 1000.0
    m0_metrics = evaluate_model_metrics(y_test_raw, y_pred_m0)
    m0_res = evaluate_physics_residual_error(X_test_raw, y_pred_m0, thermo)
    results["Model 0: Physics-Only (Zero-Data)"] = {
        **m0_metrics, "Physics_Residual_kW": m0_res, "Latency_ms": round(lat_m0, 4), "Trainable_Params": 0
    }
    
    # -------------------------------------------------------------
    # Model 1: Polynomial Ridge Regression
    # -------------------------------------------------------------
    print("[2/6] Training & Evaluating Model 1: Polynomial Ridge Regression...")
    m1 = PolynomialRidgeBaseline(degree=2)
    m1.fit(X_train_raw, y_train_raw)
    t0 = time.time()
    y_pred_m1 = m1.predict(X_test_raw)
    lat_m1 = ((time.time() - t0) / len(X_test_raw)) * 1000.0
    m1_metrics = evaluate_model_metrics(y_test_raw, y_pred_m1)
    m1_res = evaluate_physics_residual_error(X_test_raw, y_pred_m1, thermo)
    results["Model 1: Polynomial Ridge"] = {
        **m1_metrics, "Physics_Residual_kW": m1_res, "Latency_ms": round(lat_m1, 4), "Trainable_Params": 15
    }
    
    # -------------------------------------------------------------
    # Model 2: Random Forest Regressor
    # -------------------------------------------------------------
    print("[3/6] Training & Evaluating Model 2: Random Forest Regressor...")
    m2 = RandomForestBaseline(n_estimators=80, max_depth=12)
    # Train on subset for fast CPU execution
    m2.fit(X_train_raw[:6000], y_train_raw[:6000])
    t0 = time.time()
    y_pred_m2 = m2.predict(X_test_raw)
    lat_m2 = ((time.time() - t0) / len(X_test_raw)) * 1000.0
    m2_metrics = evaluate_model_metrics(y_test_raw, y_pred_m2)
    m2_res = evaluate_physics_residual_error(X_test_raw, y_pred_m2, thermo)
    results["Model 2: Random Forest"] = {
        **m2_metrics, "Physics_Residual_kW": m2_res, "Latency_ms": round(lat_m2, 4), "Trainable_Params": 125000
    }
    
    # -------------------------------------------------------------
    # Model 3: Standard Deep MLP (Pure Data Loss)
    # -------------------------------------------------------------
    print("[4/6] Training & Evaluating Model 3: Standard Deep MLP (Data-Only)...")
    m3 = StandardDeepMLP(in_features=4, hidden_dim=64)
    m3.fit_model(X_train, y_train, epochs=epochs_nn, lr=0.003, batch_size=256)
    t0 = time.time()
    with torch.no_grad():
        m3.eval()
        y_pred_m3_scaled = m3(X_test).numpy()
    y_pred_m3 = pipeline.inverse_transform_target(y_pred_m3_scaled)
    lat_m3 = ((time.time() - t0) / len(X_test)) * 1000.0
    m3_metrics = evaluate_model_metrics(y_test_raw, y_pred_m3)
    m3_res = evaluate_physics_residual_error(X_test_raw, y_pred_m3, thermo)
    n_params_m3 = sum(p.numel() for p in m3.parameters())
    results["Model 3: Standard Deep MLP (Data-Only)"] = {
        **m3_metrics, "Physics_Residual_kW": m3_res, "Latency_ms": round(lat_m3, 4), "Trainable_Params": n_params_m3
    }
    
    # -------------------------------------------------------------
    # Model 4: Recurrent LSTM Network
    # -------------------------------------------------------------
    print("[5/6] Training & Evaluating Model 4: Recurrent LSTM...")
    m4 = LSTMBaseline(in_features=4, hidden_dim=48, num_layers=2)
    m4.fit_model(X_train, y_train, epochs=min(epochs_nn, 150), lr=0.003, batch_size=256)
    t0 = time.time()
    with torch.no_grad():
        m4.eval()
        y_pred_m4_scaled = m4(X_test).numpy()
    y_pred_m4 = pipeline.inverse_transform_target(y_pred_m4_scaled)
    lat_m4 = ((time.time() - t0) / len(X_test)) * 1000.0
    m4_metrics = evaluate_model_metrics(y_test_raw, y_pred_m4)
    m4_res = evaluate_physics_residual_error(X_test_raw, y_pred_m4, thermo)
    n_params_m4 = sum(p.numel() for p in m4.parameters())
    results["Model 4: Recurrent LSTM"] = {
        **m4_metrics, "Physics_Residual_kW": m4_res, "Latency_ms": round(lat_m4, 4), "Trainable_Params": n_params_m4
    }
    
    # -------------------------------------------------------------
    # Model 5: Physics-Informed Neural Network (PINN)
    # -------------------------------------------------------------
    print("[6/6] Training & Evaluating Model 5: Physics-Informed Neural Network (PINN)...")
    m5 = BoilerPINN(in_features=4, hidden_dim=64, w_data=1.0, w_phys=0.20, w_mono=0.05, w_bound=0.02)
    m5.fit_pinn(X_train, y_train, deg_train, pipeline.scaler_X, pipeline.scaler_y, epochs=epochs_nn, lr=0.003, verbose=False)
    t0 = time.time()
    diag = m5.predict_with_diagnostics(X_test, pipeline.scaler_X, pipeline.scaler_y)
    lat_m5 = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_m5 = diag["t_pred_k"]
    m5_metrics = evaluate_model_metrics(y_test_raw, y_pred_m5)
    m5_res = float(np.mean(diag["physics_residual_kw"]))
    n_params_m5 = sum(p.numel() for p in m5.parameters())
    results["Model 5: Physics-Informed Neural Network (PINN)"] = {
        **m5_metrics, "Physics_Residual_kW": m5_res, "Latency_ms": round(lat_m5, 4), "Trainable_Params": n_params_m5
    }
    
    # -------------------------------------------------------------
    # Leaderboard Summary Display
    # -------------------------------------------------------------
    df_results = pd.DataFrame(results).T
    print("\n" + "=" * 75)
    print("BENCHMARK LEADERBOARD SUMMARY")
    print("=" * 75)
    print(df_results.to_string())
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "benchmark_leaderboard.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Benchmark leaderboard written to {out_json}")
    
    # Save trained PINN weights for dashboard and what-if simulation!
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models_saved")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(m5.state_dict(), os.path.join(models_dir, "pinn_trained.pt"))
    print(f"[SAVED] Trained PINN weights saved to {models_dir}/pinn_trained.pt")
    
    return results, df_results


if __name__ == "__main__":
    run_full_benchmark(epochs_nn=200)
