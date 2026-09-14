"""
Thermodynamic Energy Balance Engine for Industrial Boiler.
Implements the 1st Law of Thermodynamics (Conservation of Energy),
heat exchanger effectiveness-NTU / resistance relations, and PyTorch autograd loss functions.
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp


class BoilerThermodynamics:
    """
    Thermodynamic model of the Viessmann Vitorond 200 / Industrial Boiler.
    Governed by the 1st Law of Thermodynamics:
    Sensible water heating: Q_water = m_dot_w * c_p * (T_supply - T_return)
    Heat exchanger network: 1 / Q_water = 1 / Q_clean + gamma * R_fouling
    Transient response: C_sys * dT_supply / dt = Q_transfer - Q_water - Q_casing_loss
    """
    def __init__(
        self,
        c_sys=350.0,             # System thermal capacity (kJ/K)
        water_cp=4.186,          # Specific heat capacity of water (kJ/kg*K)
        q_clean_kw=388.8,        # Baseline nominal thermal transfer capacity (kW)
        gamma_foul=0.00012,      # Resistance scaling factor
        u_loss_kw_per_k=0.025,   # Ambient shell loss coefficient (kW/K)
        t_ambient=293.15         # Ambient temperature (K)
    ):
        self.c_sys = c_sys
        self.water_cp = water_cp
        self.q_clean_kw = q_clean_kw
        self.gamma_foul = gamma_foul
        self.u_loss_kw_per_k = u_loss_kw_per_k
        self.t_ambient = t_ambient

    def compute_clean_q_water(self, fuel_mdot=2.5, t_air=293.15):
        """
        Nominal combustion heat transferred to water (kW).
        Slightly modulated by fuel rate and intake air temperature.
        """
        # Baseline ~388.8 kW with slight positive modulation from fuel and air
        q_mod = self.q_clean_kw * (0.98 + 0.008 * fuel_mdot + 0.0002 * (t_air - 293.15))
        return q_mod

    def compute_steady_state_t_supply(self, fuel_mdot, t_air, t_return, water_mdot, r_fouling=0.0):
        """
        Analytical steady-state supply temperature from thermal resistance:
        1 / Q_eff = 1 / Q_clean + 0.00012 * r_fouling
        Q_eff = m_w * c_p * (T_supply - T_return)
        => T_supply = T_return + Q_eff / (m_w * c_p)
        """
        q_clean = self.compute_clean_q_water(fuel_mdot, t_air)
        inv_q_eff = (1.0 / q_clean) + self.gamma_foul * np.maximum(r_fouling, 0.0)
        q_eff = 1.0 / inv_q_eff
        
        delta_t = q_eff / (water_mdot * self.water_cp)
        return t_return + delta_t

    def evaluate_energy_residual_numpy(self, dt_supply_dt, fuel_mdot, t_air, t_return, water_mdot, t_supply, r_fouling=0.0):
        """
        Evaluates physical transient energy residual in kW:
        Residual = C_sys * dT/dt - (Q_effective - Q_absorbed - Q_loss)
        """
        q_clean = self.compute_clean_q_water(fuel_mdot, t_air)
        inv_q_eff = (1.0 / q_clean) + self.gamma_foul * np.maximum(r_fouling, 0.0)
        q_eff = 1.0 / inv_q_eff
        
        q_absorbed = water_mdot * self.water_cp * (t_supply - t_return)
        q_loss = self.u_loss_kw_per_k * (t_supply - self.t_ambient)
        
        acc = self.c_sys * dt_supply_dt
        return acc - (q_eff - q_absorbed - q_loss)

    def evaluate_energy_residual_torch(
        self,
        dt_supply_dt,
        fuel_mdot,
        t_air,
        t_return,
        water_mdot,
        t_supply,
        r_foul_pred=None
    ):
        """
        Fully differentiable PyTorch tensor energy residual.
        Enforces 1st Law energy conservation during PINN gradient descent.
        """
        if r_foul_pred is None:
            r_foul_pred = torch.zeros_like(t_supply)
            
        q_clean = self.q_clean_kw * (0.98 + 0.008 * fuel_mdot + 0.0002 * (t_air - 293.15))
        inv_q_eff = (1.0 / q_clean) + self.gamma_foul * torch.clamp(r_foul_pred, min=0.0)
        q_eff = 1.0 / torch.clamp(inv_q_eff, min=1e-5)
        
        q_absorbed = water_mdot * self.water_cp * (t_supply - t_return)
        q_loss = self.u_loss_kw_per_k * (t_supply - self.t_ambient)
        
        acc = self.c_sys * dt_supply_dt
        net_imbalance = acc - (q_eff - q_absorbed - q_loss)
        return net_imbalance

    def simulate_transient_trajectory(self, t_span, t0, fuel_mdot, t_air, t_return, water_mdot, r_fouling=0.0):
        """
        Integrates the physical ODE over time using Scipy solve_ivp (RK45).
        Returns time array and temperature trajectory.
        """
        def ode_func(t, y):
            t_out = y[0]
            q_clean = self.compute_clean_q_water(fuel_mdot, t_air)
            inv_q_eff = (1.0 / q_clean) + self.gamma_foul * max(r_fouling, 0.0)
            q_eff = 1.0 / inv_q_eff
            q_absorbed = water_mdot * self.water_cp * (t_out - t_return)
            q_loss = self.u_loss_kw_per_k * (t_out - self.t_ambient)
            return [(q_eff - q_absorbed - q_loss) / self.c_sys]
        
        sol = solve_ivp(ode_func, t_span, [t0], method="RK45", t_eval=np.linspace(t_span[0], t_span[1], 100))
        return sol.t, sol.y[0]
