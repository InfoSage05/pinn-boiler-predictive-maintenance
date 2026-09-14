"""
Digital Twin State Synchronizer.
Continuously synchronizes virtual model state with incoming telemetry,
invokes the PyTorch PINN inference engine for unobservable state estimation,
and evaluates real-time physical consistency residuals.
"""

from datetime import datetime
from collections import deque
import numpy as np
import torch
import pandas as pd

from src.digital_twin.state import (
    BoilerPhysicalParameters,
    BoilerOperationalTelemetry,
    BoilerDegradationState,
    DigitalTwinStateSnapshot
)
from src.physics.boiler_thermo import BoilerThermodynamics
from src.physics.fouling_model import FoulingDegradationModel


class DigitalTwinSynchronizer:
    def __init__(
        self,
        pinn_model=None,
        scaler_X=None,
        scaler_y=None,
        history_len=200
    ):
        self.params = BoilerPhysicalParameters()
        self.thermo = BoilerThermodynamics()
        self.fouling_model = FoulingDegradationModel()
        self.pinn_model = pinn_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.history = deque(maxlen=history_len)
        self.current_snapshot: DigitalTwinStateSnapshot = None

    def process_telemetry_frame(self, telemetry_dict: dict) -> DigitalTwinStateSnapshot:
        """
        Processes a single incoming telemetry record, updates virtual twin state,
        estimates unobservable fouling parameters, and evaluates physics residuals.
        """
        # 1. Parse operational telemetry
        ts = telemetry_dict.get("timestamp", datetime.now())
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
            
        fuel_flow = float(telemetry_dict.get("fuel_flow_kg_s", telemetry_dict.get("Fuel_Mdot", 2.5)))
        water_flow = float(telemetry_dict.get("water_flow_kg_s", telemetry_dict.get("Water_Mdot", 7.75)))
        air_temp = float(telemetry_dict.get("air_temp_k", telemetry_dict.get("Tair", 293.15)))
        return_temp = float(telemetry_dict.get("return_water_temp_k", telemetry_dict.get("Treturn", 333.0)))
        supply_temp = float(telemetry_dict.get("supply_water_temp_k", telemetry_dict.get("Tsupply", 345.0)))
        load_pct = float(telemetry_dict.get("load_pct", 75.0))
        
        telemetry = BoilerOperationalTelemetry(
            timestamp=ts,
            fuel_flow_kg_s=fuel_flow,
            water_flow_kg_s=water_flow,
            air_temp_k=air_temp,
            return_water_temp_k=return_temp,
            supply_water_temp_k=supply_temp,
            load_percentage=load_pct
        )
        
        # 2. Estimate latent degradation via PINN or analytical physics
        if self.pinn_model is not None and self.scaler_X is not None and self.scaler_y is not None:
            raw_features = np.array([[fuel_flow, air_temp, return_temp, water_flow]])
            scaled_features = self.scaler_X.transform(raw_features)
            diag = self.pinn_model.predict_with_diagnostics(
                torch.tensor(scaled_features, dtype=torch.float32),
                self.scaler_X,
                self.scaler_y
            )
            rf_est = float(diag["fouling_pred"][0, 0])
            phys_residual = float(diag["physics_residual_kw"][0, 0])
        else:
            # Fallback estimation using 1st Law energy discrepancy
            q_clean = self.thermo.compute_clean_q_water(fuel_mdot=fuel_flow, t_air=air_temp)
            q_absorbed = water_flow * self.thermo.water_cp * (supply_temp - return_temp)
            inv_q_absorbed = 1.0 / max(q_absorbed, 10.0)
            inv_q_clean = 1.0 / q_clean
            rf_est = max((inv_q_absorbed - inv_q_clean) / self.thermo.gamma_foul, 0.0)
            phys_residual = abs(q_clean - q_absorbed)
            
        # 3. Calculate derived degradation attributes
        soot_mm = self.fouling_model.compute_soot_thickness_mm(rf_est)
        u_eff = self.fouling_model.compute_effective_u(self.params.clean_heat_transfer_coeff_kw_m2k, rf_est)
        
        # Tube metal temperature estimation
        heat_flux = (u_eff * (supply_temp - return_temp)) / self.params.effective_heat_area_m2
        tube_temp_est_c = (supply_temp - 273.15) + (heat_flux * 0.8) # approximate tube wall gradient
        fuel_waste_usd = self.fouling_model.compute_hourly_fuel_waste_cost(fuel_flow, rf_est)
        
        degradation = BoilerDegradationState(
            fouling_resistance_rf=rf_est,
            estimated_soot_thickness_mm=soot_mm,
            effective_u_kw_m2k=u_eff,
            tube_metal_temp_est_c=tube_temp_est_c,
            hourly_fuel_waste_rate_usd=fuel_waste_usd
        )
        
        # 4. Compute Health Index & Remaining Safe Operating Window (RSOW)
        rf_ratio = min(rf_est / self.params.critical_fouling_threshold, 1.5)
        temp_margin = max(0.0, (tube_temp_est_c - 520.0) / (self.params.max_safe_tube_metal_temp_c - 520.0))
        residual_penalty = min(phys_residual / 40.0, 1.0)
        
        hi = 1.0 - (0.50 * rf_ratio + 0.30 * temp_margin + 0.20 * residual_penalty)
        hi = float(np.clip(hi, 0.05, 1.0))
        
        # Determine Alarm Level based on ISA-18.2 rules
        if hi >= 0.75 and tube_temp_est_c < 548.0:
            alarm = "NORMAL"
        elif hi >= 0.50:
            alarm = "ADVISORY"
        elif hi >= 0.30:
            alarm = "WARNING"
        else:
            alarm = "CRITICAL"
            
        # Remaining Safe Operating Window (hours)
        if rf_est >= self.params.critical_fouling_threshold:
            rsow = 2.0 # immediate maintenance required
        else:
            burn_rate = 0.00035 # normal fouling accumulation per hour
            rsow = (self.params.critical_fouling_threshold - rf_est) / burn_rate
            rsow = float(np.clip(rsow, 1.0, 120.0))
            
        snapshot = DigitalTwinStateSnapshot(
            snapshot_time=ts,
            physical_params=self.params,
            telemetry=telemetry,
            degradation=degradation,
            health_index=hi,
            physics_residual_kw=phys_residual,
            alarm_level=alarm,
            remaining_safe_operating_window_hours=rsow
        )
        
        self.current_snapshot = snapshot
        self.history.append(snapshot)
        return snapshot

    def get_history_dataframe(self) -> pd.DataFrame:
        """Exports snapshot history as a structured pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        records = [s.to_dict() for s in self.history]
        return pd.DataFrame(records)
