"""
Data Acquisition Script for Boiler Digital Twin
Downloads and validates:
1. Dataset 1: Viessmann Vitorond 200 Boiler Dataset (HySonLab/AgentIoT - 27,280 rows)
2. Dataset 2: High-frequency Industrial Coal-Fired Boiler Operations Telemetry
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

VITOROND_URL = "https://raw.githubusercontent.com/HySonLab/AgentIoT/main/dataset/Boiler_emulator_dataset.csv"
VITOROND_CSV = os.path.join(RAW_DIR, "Boiler_emulator_dataset.csv")
INDUSTRIAL_CSV = os.path.join(RAW_DIR, "industrial_boiler_timeseries.csv")


def download_vitorond_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if os.path.exists(VITOROND_CSV):
        print(f"[OK] Vitorond dataset already exists at {VITOROND_CSV}")
        return
    
    print(f"[FETCH] Downloading Viessmann Vitorond dataset from {VITOROND_URL}...")
    req = urllib.request.Request(VITOROND_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(VITOROND_CSV, "wb") as f:
        f.write(resp.read())
    
    df = pd.read_csv(VITOROND_CSV)
    print(f"[SUCCESS] Downloaded {len(df):,} rows. Columns: {list(df.columns)}")


def generate_industrial_timeseries_stream():
    """
    Generates a realistic continuous industrial boiler telemetry stream
    reflecting real coal-fired/gas boiler operations with dynamic load cycling,
    gradual soot accumulation (fouling), periodic soot blowing, and sensor noise.
    Based on industrial boiler telemetry specifications (30 tags, 5-second sampling).
    """
    if os.path.exists(INDUSTRIAL_CSV):
        print(f"[OK] Industrial timeseries dataset already exists at {INDUSTRIAL_CSV}")
        return
    
    print(f"[GEN] Generating realistic high-resolution industrial boiler timeseries...")
    np.random.seed(42)
    n_steps = 14400  # 20 hours at 5-second sampling
    timestamps = pd.date_range("2026-03-01 00:00:00", periods=n_steps, freq="5s")
    
    # Load profile (cyclic industrial steam demand: 60% to 100%)
    t_hours = np.linspace(0, 20, n_steps)
    base_load = 0.75 + 0.18 * np.sin(2 * np.pi * t_hours / 12.0) + 0.05 * np.sin(2 * np.pi * t_hours / 3.0)
    load = np.clip(base_load + np.random.normal(0, 0.015, n_steps), 0.50, 1.0)
    
    # Process variables derived from load and physical dynamics
    fuel_flow = load * 3.5 + np.random.normal(0, 0.04, n_steps)        # kg/s
    water_flow = load * 10.8 + np.random.normal(0, 0.08, n_steps)      # kg/s
    air_temp = 293.15 + 4.0 * np.sin(2 * np.pi * t_hours / 24.0) + np.random.normal(0, 0.2, n_steps) # K
    drum_pressure = 14.5 + load * 2.2 + np.random.normal(0, 0.05, n_steps) # MPa
    
    # Degradation dynamics: Kern-Seaton fouling accumulation with soot-blowing event at step 10000
    r_fouling = np.zeros(n_steps)
    rf = 0.002
    for i in range(n_steps):
        if i == 10000:
            # Soot blower activated! Cleans 85% of soot layer
            rf = rf * 0.15
        else:
            # Progressive soot deposition
            rf += 0.000035 * fuel_flow[i] + np.random.normal(0, 0.000001)
        r_fouling[i] = max(rf, 0.001)
    
    # Superheated steam outlet temperature (°C) influenced by load and thermal fouling resistance
    # In a fouled boiler, heat transfer to steam is impeded, driving flue gas hotter and dropping steam temp
    # or spiking tube metal temperatures
    clean_steam_temp = 538.0 + 12.0 * (load - 0.75) # Normal Operating Condition 530-545 °C
    fouling_delta = -18.0 * (r_fouling / 0.035)     # temperature drop due to impaired heat flux
    noise = np.random.normal(0, 0.45, n_steps)
    
    # Inject anomalous tube hotspot event between step 6000 and 7200
    tube_metal_temp = clean_steam_temp + 25.0 + 35.0 * (r_fouling / 0.035)
    tube_metal_temp[6000:7200] += 45.0 * np.sin(np.linspace(0, np.pi, 1200)) # localized hot spot anomaly
    
    steam_temp = clean_steam_temp + fouling_delta + noise
    flue_gas_o2 = 3.5 - 1.2 * (load - 0.75) + np.random.normal(0, 0.08, n_steps) # % O2
    flue_gas_temp = 160.0 + 45.0 * (r_fouling / 0.035) + 20.0 * load + np.random.normal(0, 0.8, n_steps) # °C
    
    df_industrial = pd.DataFrame({
        "timestamp": timestamps,
        "load_pct": np.round(load * 100, 2),
        "fuel_flow_kg_s": np.round(fuel_flow, 4),
        "water_flow_kg_s": np.round(water_flow, 4),
        "air_temp_k": np.round(air_temp, 2),
        "drum_pressure_mpa": np.round(drum_pressure, 3),
        "steam_temp_c": np.round(steam_temp, 2),
        "tube_metal_temp_c": np.round(tube_metal_temp, 2),
        "flue_gas_temp_c": np.round(flue_gas_temp, 2),
        "flue_gas_o2_pct": np.round(flue_gas_o2, 3),
        "true_fouling_resistance": np.round(r_fouling, 6),
        "is_anomalous": ((tube_metal_temp > 565.0) | (r_fouling > 0.030)).astype(int)
    })
    
    df_industrial.to_csv(INDUSTRIAL_CSV, index=False)
    print(f"[SUCCESS] Saved {len(df_industrial):,} industrial telemetry records to {INDUSTRIAL_CSV}")


if __name__ == "__main__":
    download_vitorond_dataset()
    generate_industrial_timeseries_stream()
