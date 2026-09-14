"""
Fouling & Scaling Degradation Dynamics Model.
Implements the Kern-Seaton asymptotic deposition-removal model,
thermal resistance degradation, tube metal temperature overheating calculations,
and fuel waste economics.
"""

import numpy as np


class FoulingDegradationModel:
    """
    Models physical heat exchanger tube fouling (gas-side soot)
    and scaling (water-side mineral precipitation).
    """
    def __init__(
        self,
        r_clean=0.0005,          # Clean metal & baseline resistance (m^2*K/kW)
        r_foul_crit=0.040,       # Critical fouling resistance trigger for soot blowing
        r_foul_asymptotic=0.065, # Maximum asymptotic fouling resistance
        tau_foul_hours=48.0,     # Time constant for fouling build-up (hours)
        soot_conductivity=0.00008, # Soot thermal conductivity (kW/m*K) ~ 0.08 W/m*K
        fuel_cost_per_kg=0.85    # USD per kg of fuel
    ):
        self.r_clean = r_clean
        self.r_foul_crit = r_foul_crit
        self.r_foul_asymptotic = r_foul_asymptotic
        self.tau_foul_hours = tau_foul_hours
        self.soot_conductivity = soot_conductivity
        self.fuel_cost_per_kg = fuel_cost_per_kg

    def compute_asymptotic_fouling(self, t_hours):
        """
        Kern-Seaton asymptotic fouling curve:
        R_f(t) = R_clean + (R_asymp - R_clean) * (1 - exp(-t / tau))
        """
        return self.r_clean + (self.r_foul_asymptotic - self.r_clean) * (
            1.0 - np.exp(-np.maximum(t_hours, 0.0) / self.tau_foul_hours)
        )

    def compute_soot_thickness_mm(self, r_fouling):
        """
        Estimates soot deposit thickness on fireside tubes (mm):
        delta = R_f * k_soot * 1000
        """
        return np.maximum(r_fouling, 0.0) * self.soot_conductivity * 1000.0

    def compute_effective_u(self, u_clean, r_fouling, r_scaling=0.0):
        """
        Calculates degraded overall heat transfer coefficient U(t) (kW/m^2*K):
        1/U(t) = 1/U_clean + R_f(t) + R_s(t)
        """
        total_r = (1.0 / u_clean) + np.maximum(r_fouling, 0.0) + np.maximum(r_scaling, 0.0)
        return 1.0 / total_r

    def compute_tube_metal_temperature(self, t_water, heat_flux_kw_m2, r_scaling=0.0, h_water=4.5):
        """
        Estimates tube metal wall temperature (°C or K).
        When waterside scale builds up, heat flux cannot dissipate efficiently
        into the water, causing metal tube wall overheating and creep rupture!
        T_wall = T_water + q'' * (1/h_water + R_scaling)
        """
        return t_water + heat_flux_kw_m2 * ((1.0 / h_water) + np.maximum(r_scaling, 0.0))

    def compute_hourly_fuel_waste_cost(self, fuel_mdot_nominal, r_fouling, u_clean=1.85):
        """
        Calculates excess fuel cost ($/hour) caused by thermal inefficiency:
        As U drops, boiler must burn more fuel to deliver identical steam output.
        """
        u_eff = self.compute_effective_u(u_clean, r_fouling)
        efficiency_ratio = u_eff / u_clean # <= 1.0
        # Excess fuel flow required (kg/s)
        excess_fuel_rate = fuel_mdot_nominal * (1.0 / max(efficiency_ratio, 0.50) - 1.0)
        # Cost per hour ($)
        hourly_waste = excess_fuel_rate * 3600.0 * self.fuel_cost_per_kg
        return hourly_waste
