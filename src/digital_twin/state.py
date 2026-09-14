"""
Digital Twin State Representations and Parameter Container.
Follows ISO 23247 Digital Twin framework and OpenFactoryTwin (OFacT) principles:
Maintains physical asset parameters, live operational telemetry,
latent degradation state, and synchronized virtual state snapshots.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class BoilerPhysicalParameters:
    """Static and slowly-varying physical attributes of the boiler asset."""
    asset_id: str = "BOILER-UNIT-01"
    model_name: str = "Viessmann Vitorond 200"
    rated_capacity_kw: float = 388.8
    effective_heat_area_m2: float = 12.5
    clean_heat_transfer_coeff_kw_m2k: float = 1.85
    c_sys_thermal_capacity_kj_k: float = 350.0
    critical_fouling_threshold: float = 0.035
    max_safe_tube_metal_temp_c: float = 560.0
    design_steam_temp_c: float = 540.0


@dataclass
class BoilerOperationalTelemetry:
    """Current streaming sensor observations from the physical boiler."""
    timestamp: datetime = field(default_factory=datetime.now)
    fuel_flow_kg_s: float = 2.5
    water_flow_kg_s: float = 7.75
    air_temp_k: float = 293.15
    return_water_temp_k: float = 333.0
    supply_water_temp_k: float = 345.0
    drum_pressure_mpa: float = 16.2
    flue_gas_o2_pct: float = 3.5
    load_percentage: float = 75.0


@dataclass
class BoilerDegradationState:
    """Virtual latent degradation state estimated by PINN."""
    fouling_resistance_rf: float = 0.005      # m^2*K/kW
    estimated_soot_thickness_mm: float = 0.40 # mm
    waterside_scale_index: float = 0.002
    effective_u_kw_m2k: float = 1.72
    tube_metal_temp_est_c: float = 542.0
    hourly_fuel_waste_rate_usd: float = 8.50


@dataclass
class DigitalTwinStateSnapshot:
    """Complete synchronized digital shadow of the boiler work system."""
    snapshot_time: datetime
    physical_params: BoilerPhysicalParameters
    telemetry: BoilerOperationalTelemetry
    degradation: BoilerDegradationState
    health_index: float = 0.92
    physics_residual_kw: float = 1.2
    alarm_level: str = "NORMAL" # NORMAL, ADVISORY, WARNING, CRITICAL
    remaining_safe_operating_window_hours: float = 48.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.snapshot_time.isoformat(),
            "load_pct": self.telemetry.load_percentage,
            "fuel_flow": self.telemetry.fuel_flow_kg_s,
            "water_flow": self.telemetry.water_flow_kg_s,
            "supply_temp_k": self.telemetry.supply_water_temp_k,
            "fouling_rf": self.degradation.fouling_resistance_rf,
            "soot_thickness_mm": self.degradation.estimated_soot_thickness_mm,
            "effective_u": self.degradation.effective_u_kw_m2k,
            "tube_metal_temp_c": self.degradation.tube_metal_temp_est_c,
            "health_index": self.health_index,
            "physics_residual_kw": self.physics_residual_kw,
            "alarm_level": self.alarm_level,
            "rsow_hours": self.remaining_safe_operating_window_hours
        }
