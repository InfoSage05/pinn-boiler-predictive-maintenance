"""
Data Preprocessor & Feature Engineering for Boiler PINN-DT.
Extracts thermodynamic variables, unobservable degradation severity,
and formats data for PyTorch PINN and baseline training.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_condition_severity(condition_str, class_label):
    """
    Parses textual Condition string (e.g. 'F = 0.25', 'S = 0.15', '%=0.20')
    into numerical severity values.
    """
    condition_str = str(condition_str).strip()
    
    if class_label == "Fouling":
        match = re.search(r"F\s*=\s*([0-9.]+)", condition_str)
        return float(match.group(1)) if match else 0.05
    elif class_label == "Scaling":
        match = re.search(r"S\s*=\s*([0-9.]+)", condition_str)
        return float(match.group(1)) if match else 0.05
    elif class_label in ["ExcessAir", "Lean"]:
        match = re.search(r"([0-9.]+)", condition_str)
        return float(match.group(1)) if match else 0.10
    elif class_label == "Nominal":
        return 0.0
    return 0.0


class BoilerDataPipeline:
    def __init__(self, raw_csv_path=None):
        if raw_csv_path is None:
            raw_csv_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "raw", "Boiler_emulator_dataset.csv"
            )
        self.raw_csv_path = os.path.abspath(raw_csv_path)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
        # Feature column names
        self.feature_cols = ["Fuel_Mdot", "Tair", "Treturn", "Water_Mdot"]
        self.target_col = "Tsupply"
        self.degradation_col = "degradation_severity"

    def load_and_clean_data(self):
        df = pd.read_csv(self.raw_csv_path)
        
        # Parse numerical degradation severity
        df["degradation_severity"] = [
            parse_condition_severity(row["Condition"], row["Class"])
            for _, row in df.iterrows()
        ]
        
        # Binary anomaly label: 0 for Nominal, 1 for fault states
        df["is_fault"] = (df["Class"] != "Nominal").astype(int)
        
        # Class encoding for classification baselines
        class_map = {"Nominal": 0, "Fouling": 1, "Scaling": 2, "ExcessAir": 3, "Lean": 4}
        df["class_code"] = df["Class"].map(class_map).fillna(0).astype(int)
        
        # Thermodynamic energy balance features (kW)
        # Assuming LHV = 42,000 kJ/kg, cp = 4.186 kJ/kg*K
        df["Q_fuel_approx_kw"] = df["Fuel_Mdot"] * 42000.0 * 0.90
        df["Q_water_delta_kw"] = df["Water_Mdot"] * 4.186 * (df["Tsupply"] - df["Treturn"])
        
        return df

    def get_train_val_test_splits(self, test_size=0.15, val_size=0.15, random_state=42):
        df = self.load_and_clean_data()
        
        # Filter in-distribution (ID) data for standard training
        # We reserve extreme high-load points (Fuel_Mdot > 3.5 kg/s and Water_Mdot > 10.5 kg/s) for OOD testing!
        ood_mask = (df["Fuel_Mdot"] >= 3.5) & (df["Water_Mdot"] >= 10.5)
        df_id = df[~ood_mask].copy()
        df_ood = df[ood_mask].copy()
        
        X = df_id[self.feature_cols].values
        y_temp = df_id[[self.target_col]].values
        y_deg = df_id[[self.degradation_col]].values
        
        # Split train and remaining
        X_train, X_rem, y_train, y_rem, deg_train, deg_rem = train_test_split(
            X, y_temp, y_deg, test_size=(test_size + val_size), random_state=random_state
        )
        
        # Split remaining into val and test
        relative_val = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test, deg_val, deg_test = train_test_split(
            X_rem, y_rem, deg_rem, test_size=(1.0 - relative_val), random_state=random_state
        )
        
        # Fit Scalers on Train ONLY
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # OOD split scaling
        X_ood = df_ood[self.feature_cols].values
        y_ood = df_ood[[self.target_col]].values
        deg_ood = df_ood[[self.degradation_col]].values
        
        X_ood_scaled = self.scaler_X.transform(X_ood)
        y_ood_scaled = self.scaler_y.transform(y_ood)
        
        self.is_fitted = True
        
        data_dict = {
            "X_train": torch.tensor(X_train_scaled, dtype=torch.float32),
            "y_train": torch.tensor(y_train_scaled, dtype=torch.float32),
            "deg_train": torch.tensor(deg_train, dtype=torch.float32),
            "X_train_raw": X_train,
            "y_train_raw": y_train,
            
            "X_val": torch.tensor(X_val_scaled, dtype=torch.float32),
            "y_val": torch.tensor(y_val_scaled, dtype=torch.float32),
            "deg_val": torch.tensor(deg_val, dtype=torch.float32),
            "X_val_raw": X_val,
            "y_val_raw": y_val,
            
            "X_test": torch.tensor(X_test_scaled, dtype=torch.float32),
            "y_test": torch.tensor(y_test_scaled, dtype=torch.float32),
            "deg_test": torch.tensor(deg_test, dtype=torch.float32),
            "X_test_raw": X_test,
            "y_test_raw": y_test,
            
            "X_ood": torch.tensor(X_ood_scaled, dtype=torch.float32),
            "y_ood": torch.tensor(y_ood_scaled, dtype=torch.float32),
            "deg_ood": torch.tensor(deg_ood, dtype=torch.float32),
            "X_ood_raw": X_ood,
            "y_ood_raw": y_ood,
            
            "df_full": df,
            "df_ood": df_ood
        }
        return data_dict

    def inverse_transform_target(self, y_scaled):
        """Converts normalized temperature predictions back to Kelvin."""
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled)

    def inverse_transform_features(self, X_scaled):
        """Converts normalized inputs back to physical units."""
        if isinstance(X_scaled, torch.Tensor):
            X_scaled = X_scaled.detach().cpu().numpy()
        return self.scaler_X.inverse_transform(X_scaled)


def load_industrial_timeseries(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "raw", "industrial_boiler_timeseries.csv"
        )
    csv_path = os.path.abspath(csv_path)
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
