# Thermodynamic Physics Derivation & PINN Formulation

## 1. Transient Energy Balance for Industrial Steam / Hot Water Boilers

Consider a control volume enclosing the boiler heat exchange tubes, combustion chamber, and water circulating loop.
Applying the **First Law of Thermodynamics (Conservation of Energy)** to the unsteady open system:

    dE_cv/dt = Q_combustion - W_cv - Q_fluid - Q_loss    [kW]

In a boiler system, shaft work `W_cv = 0`. The total stored thermal energy in the control volume is:

    E_cv = (m_metal * cp_metal + m_water * cp_water) * T_supply = C_sys * T_supply

where `C_sys` is the effective lumped thermal heat capacity [kJ/K].
Thus, the governing ordinary differential equation (ODE) is:

    C_sys * dT_supply/dt = Q_combustion - Q_water - Q_casing_loss    [kW]

### 1.1 Combustion Heat Release
The heat release rate from burner firing is:

    Q_combustion(t) = m_dot_fuel(t) * LHV * eta_comb(lambda)

where:
- `m_dot_fuel` is fuel firing mass flow rate [kg/s].
- `LHV` is the Lower Heating Value (approx 42,000 kJ/kg).
- `eta_comb(lambda)` is combustion efficiency dependent on the excess air equivalence ratio lambda.

### 1.2 Sensible Water Heat Absorption
    Q_water(t) = m_dot_water(t) * cp * [T_supply(t) - T_return(t)]

### 1.3 Casing and Ambient Losses
    Q_casing_loss(t) = U_loss * A_shell * [T_supply(t) - T_ambient]

---

## 2. Thermal Resistance Network & Fouling Mechanics

Heat transfer from flue gas through the tube wall to water is governed by overall thermal resistance:

    R_total(t) = 1/(U(t)*A) = 1/(h_gas*A_o) + R_fouling(t)/A_o + ln(r_o/r_i)/(2*pi*k_metal*L) + R_scaling(t)/A_i + 1/(h_water*A_i)

Under clean baseline conditions:

    1/U_clean = 1/h_gas + delta_metal/k_metal + 1/h_water

When soot and slag deposit on the external tube surfaces, an additional conduction resistance R_fouling(t) appears:

    1/U(t) = 1/U_clean + R_fouling(t)

### Kern-Seaton Asymptotic Deposition-Removal Model
Fouling kinetics balance soot particle deposition from the flue gas against shear re-entrainment:

    dRf/dt = m_dot_deposition - beta * tau_shear * Rf(t)

Under steady operational firing:

    Rf(t) = R_clean + (R_asymptotic - R_clean) * [1 - exp(-t / tau_foul)]

---

## 3. Pure PyTorch PINN Autograd Formulation

The neural network N_theta maps operational inputs to the predicted supply temperature and latent fouling resistance:

    [T_supply_hat, Rf_hat] = N_theta(m_dot_fuel, T_air, T_return, m_dot_water, t)

### 3.1 Automatic Differentiation Physics Residual
PyTorch's automatic differentiation engine (`torch.autograd.grad`) evaluates the input-output gradient:

    grad_x T_supply_hat = dT_supply_hat / dx

The energy conservation loss enforces zero physical discrepancy:

    L_physics = (1/M) * sum( m_dot_w * cp * (T_supply_hat - T_return) - [ 1 / (1/Q_clean + gamma*Rf_hat) ] )^2

### 3.2 Monotonicity Constraint Prior
Because fouling accumulates monotonically during operating shifts:

    L_monotonicity = (1/M) * sum( ReLU(dT_supply_hat / dm_dot_water) + ReLU(-dRf_hat / dt) )

This penalizes non-physical positive derivatives with respect to water flow rate and non-physical negative degradation trajectories.
