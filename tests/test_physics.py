"""Unit tests for thermodynamic physics engine."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.boiler_thermo import BoilerThermodynamics
from src.physics.fouling_model import FoulingDegradationModel


def test_steady_state_energy_conservation():
    thermo = BoilerThermodynamics()
    fuel_mdot = 2.5
    water_mdot = 7.75
    t_air = 293.15
    t_return = 333.0
    
    t_supply = thermo.compute_steady_state_t_supply(fuel_mdot, t_air, t_return, water_mdot, r_fouling=0.0)
    
    # Verify supply temperature is physically higher than return temperature
    assert t_supply > t_return, "Supply temperature must be higher than return temperature!"
    assert t_supply < 380.0, "Supply temperature must be within safe water operating range!"
    
    # Verify energy residual at steady state is near zero
    res = thermo.evaluate_energy_residual_numpy(
        dt_supply_dt=0.0,
        fuel_mdot=fuel_mdot,
        t_air=t_air,
        t_return=t_return,
        water_mdot=water_mdot,
        t_supply=t_supply,
        r_fouling=0.0
    )
    assert abs(res) < 2.0, f"Steady state energy residual too large: {res} kW"


def test_fouling_reduces_heat_transfer():
    thermo = BoilerThermodynamics()
    fouling = FoulingDegradationModel()
    
    # Higher fouling resistance must decrease effective heat transfer U
    u_clean = fouling.compute_effective_u(1.85, r_fouling=0.0)
    u_fouled = fouling.compute_effective_u(1.85, r_fouling=0.035)
    assert u_fouled < u_clean, "Fouling must reduce overall heat transfer coefficient!"
    
    # Higher fouling must reduce outlet water temperature for identical firing rate
    t_clean = thermo.compute_steady_state_t_supply(2.5, 293.15, 333.0, 7.75, r_fouling=0.0)
    t_fouled = thermo.compute_steady_state_t_supply(2.5, 293.15, 333.0, 7.75, r_fouling=0.035)
    assert t_fouled < t_clean, "Fouled heat exchange tubes must reduce outlet water temperature!"


def test_soot_thickness_scaling():
    fouling = FoulingDegradationModel()
    thick_zero = fouling.compute_soot_thickness_mm(0.0)
    thick_fouled = fouling.compute_soot_thickness_mm(0.025)
    assert thick_zero == 0.0
    assert thick_fouled > 0.0
