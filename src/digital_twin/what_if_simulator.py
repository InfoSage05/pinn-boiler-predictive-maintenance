"""
What-If Scenario Simulation Sandbox for Industrial Boiler Digital Twin.
Allows operators and plant engineers to simulate hypothetical operational decisions:
- Increasing load to 95% or 100% under high grid demand
- Derating load to 65% or 75% to alleviate thermal stress
- Delaying or advancing soot-blowing maintenance
Projects future thermal stress, tube metal overheating risk, and cumulative fuel waste.
"""

import numpy as np
import pandas as pd
from src.physics.boiler_thermo import BoilerThermodynamics
from src.physics.fouling_model import FoulingDegradationModel


class WhatIfScenarioSimulator:
    def __init__(self, pinn_model=None, scaler_X=None, scaler_y=None):
        self.thermo = BoilerThermodynamics()
        self.fouling_model = FoulingDegradationModel()
        self.pinn_model = pinn_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def simulate_future_scenario(
        self,
        current_rf: float = 0.020,
        current_t_supply_k: float = 345.0,
        scenario_load_pct: float = 85.0,
        excess_air_ratio: float = 1.15,
        horizon_hours: float = 24.0,
        time_step_hours: float = 0.5,
        perform_soot_blowing_at_hour: float = -1.0
    ) -> pd.DataFrame:
        """
        Simulates forward trajectory under chosen operational decisions.
        """
        steps = int(horizon_hours / time_step_hours) + 1
        time_points = np.linspace(0, horizon_hours, steps)
        
        # Operational variables scaled by load
        load_factor = scenario_load_pct / 100.0
        fuel_flow = load_factor * 3.3
        water_flow = load_factor * 10.2
        t_air = 293.15
        t_return = 333.0
        
        rf_traj = []
        supply_temp_traj = []
        tube_temp_c_traj = []
        health_index_traj = []
        cumulative_fuel_waste_usd = []
        status_traj = []
        
        running_rf = current_rf
        total_waste = 0.0
        
        for t in time_points:
            # Check if soot blowing is scheduled
            if 0 <= perform_soot_blowing_at_hour <= t and (t - time_step_hours) < perform_soot_blowing_at_hour:
                # Soot blower cleans 85% of soot layer
                running_rf = max(running_rf * 0.15, 0.002)
            else:
                # Progressive deposition proportional to fuel flow
                deposition_rate = 0.00030 * (fuel_flow / 2.5)
                running_rf += deposition_rate * time_step_hours
                
            # Physics-predicted supply temperature
            t_supply = self.thermo.compute_steady_state_t_supply(
                fuel_mdot=fuel_flow,
                t_air=t_air,
                t_return=t_return,
                water_mdot=water_flow,
                r_fouling=running_rf
            )
            
            # Tube metal temperature estimation
            u_eff = self.fouling_model.compute_effective_u(1.85, running_rf)
            heat_flux = (u_eff * (t_supply - t_return)) / 12.5
            tube_temp_c = (t_supply - 273.15) + (heat_flux * 0.85)
            
            # Fuel waste calculation
            hourly_waste = self.fouling_model.compute_hourly_fuel_waste_cost(fuel_flow, running_rf)
            total_waste += hourly_waste * time_step_hours
            
            # Health Index
            rf_ratio = min(running_rf / 0.035, 1.5)
            temp_margin = max(0.0, (tube_temp_c - 520.0) / (560.0 - 520.0))
            hi = np.clip(1.0 - (0.55 * rf_ratio + 0.45 * temp_margin), 0.05, 1.0)
            
            if hi >= 0.75:
                status = "NORMAL"
            elif hi >= 0.50:
                status = "ADVISORY"
            elif hi >= 0.30:
                status = "WARNING"
            else:
                status = "CRITICAL"
                
            rf_traj.append(running_rf)
            supply_temp_traj.append(t_supply)
            tube_temp_c_traj.append(tube_temp_c)
            health_index_traj.append(hi)
            cumulative_fuel_waste_usd.append(total_waste)
            status_traj.append(status)
            
        df_sim = pd.DataFrame({
            "hour": time_points,
            "fouling_rf": np.round(rf_traj, 5),
            "supply_temp_k": np.round(supply_temp_traj, 2),
            "tube_temp_c": np.round(tube_temp_c_traj, 2),
            "health_index": np.round(health_index_traj, 3),
            "cumulative_fuel_waste_usd": np.round(cumulative_fuel_waste_usd, 2),
            "status": status_traj
        })
        return df_sim

    def compare_operational_policies(self, current_rf=0.025, horizon_hours=36.0):
        """
        Compares 3 candidate operational policies for plant leadership:
        Policy A: Continue baseline 90% load without intervention
        Policy B: Derate load to 75% immediately to prolong safe window
        Policy C: Execute soot-blowing at Hour 8, then resume 90% load
        """
        df_a = self.simulate_future_scenario(
            current_rf=current_rf, scenario_load_pct=90.0, horizon_hours=horizon_hours
        )
        df_b = self.simulate_future_scenario(
            current_rf=current_rf, scenario_load_pct=75.0, horizon_hours=horizon_hours
        )
        df_c = self.simulate_future_scenario(
            current_rf=current_rf, scenario_load_pct=90.0, horizon_hours=horizon_hours, perform_soot_blowing_at_hour=8.0
        )
        return {"Policy A (No Action)": df_a, "Policy B (Derate 75%)": df_b, "Policy C (Soot Blow H8)": df_c}
