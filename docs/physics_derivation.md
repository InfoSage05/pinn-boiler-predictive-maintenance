# Thermodynamic Physics Derivation & PINN Formulation

## 1. Transient Energy Balance for Industrial Steam / Hot Water Boilers

Consider a control volume enclosing the boiler heat exchange tubes, combustion chamber, and water circulating loop.
Applying the **First Law of Thermodynamics (Conservation of Energy)** to the unsteady open system:

$$\frac{d E_{cv}}{dt} = \dot{Q}_{combustion}(t) - \dot{W}_{cv} - \dot{Q}_{fluid}(t) - \dot{Q}_{loss}(t)$$

In a boiler system, shaft work $\dot{W}_{cv} = 0$. The total stored thermal energy in the control volume is:

$$E_{cv} \approx (m_{metal} c_{p,metal} + m_{water} c_{p,water}) T_{supply}(t) = C_{sys} T_{supply}(t)$$

where $C_{sys}$ is the effective lumped thermal heat capacity ($\text{kJ/K}$).
Thus, the governing ordinary differential equation (ODE) is:

$$C_{sys} \frac{d T_{supply}(t)}{dt} = \dot{Q}_{combustion}(t) - \dot{Q}_{water}(t) - \dot{Q}_{casing\_loss}(t)$$

### 1.1 Combustion Heat Release
The heat release rate from burner firing is:

$$\dot{Q}_{combustion}(t) = \dot{m}_{fuel}(t) \cdot LHV \cdot \eta_{comb}(\lambda)$$

where:
- $\dot{m}_{fuel}$ is fuel firing mass flow rate ($\text{kg/s}$).
- $LHV$ is the Lower Heating Value ($\approx 42{,}000\text{ kJ/kg}$).
- $\eta_{comb}(\lambda)$ is combustion efficiency dependent on the excess air equivalence ratio $\lambda$.

### 1.2 Sensible Water Heat Absorption
$$\dot{Q}_{water}(t) = \dot{m}_{water}(t) \cdot c_p \cdot [T_{supply}(t) - T_{return}(t)]$$

### 1.3 Casing and Ambient Losses
$$\dot{Q}_{casing\_loss}(t) = U_{loss} A_{shell} [T_{supply}(t) - T_{ambient}]$$

---

## 2. Thermal Resistance Network & Fouling Mechanics

Heat transfer from flue gas through the tube wall to water is governed by overall thermal resistance:

$$R_{total}(t) = \frac{1}{U(t) A} = \frac{1}{h_{gas} A_o} + \frac{R_{fouling}(t)}{A_o} + \frac{\ln(r_o/r_i)}{2\pi k_{metal} L} + \frac{R_{scaling}(t)}{A_i} + \frac{1}{h_{water} A_i}$$

Under clean baseline conditions:

$$\frac{1}{U_{clean}} = \frac{1}{h_{gas}} + \frac{\delta_{metal}}{k_{metal}} + \frac{1}{h_{water}}$$

When soot and slag deposit on the external tube surfaces, an additional conduction resistance $R_{fouling}(t)$ appears:

$$\frac{1}{U(t)} = \frac{1}{U_{clean}} + R_{fouling}(t)$$

### Kern-Seaton Asymptotic Deposition-Removal Model
Fouling kinetics balance soot particle deposition from the flue gas against shear re-entrainment:

$$\frac{d R_f}{dt} = \dot{m}_{deposition} - \beta \tau_{shear} R_f(t)$$

Under steady operational firing:

$$R_f(t) = R_{clean} + (R_{asymptotic} - R_{clean}) \cdot \left[1 - \exp\left(-\frac{t}{\tau_{foul}}\right)\right]$$

---

## 3. Pure PyTorch PINN Autograd Formulation

The neural network $\mathcal{N}_\theta$ maps operational inputs to the predicted supply temperature and latent fouling resistance:

$$[\hat{T}_{supply}, \hat{R}_f] = \mathcal{N}_\theta(\dot{m}_{fuel}, T_{air}, T_{return}, \dot{m}_{water}, t)$$

### 3.1 Automatic Differentiation Physics Residual
PyTorch's automatic differentiation engine (`torch.autograd.grad`) evaluates the input-output gradient:

$$\nabla_{\mathbf{x}} \hat{T}_{supply} = \frac{\partial \hat{T}_{supply}}{\partial \mathbf{x}}$$

The energy conservation loss enforces zero physical discrepancy:

$$\mathcal{L}_{physics} = \frac{1}{M} \sum_{j=1}^M \left( \dot{m}_{w} c_p (\hat{T}_{supply} - T_{return}) - \left[\frac{1}{\frac{1}{\dot{Q}_{clean}} + \gamma \hat{R}_f}\right] \right)^2$$

### 3.2 Monotonicity Constraint Prior
Because fouling accumulates monotonically during operating shifts:

$$\mathcal{L}_{monotonicity} = \frac{1}{M} \sum_{j=1}^M \text{ReLU}\left( \frac{\partial \hat{T}_{supply}}{\partial \dot{m}_{water}} \right) + \text{ReLU}\left( - \frac{\partial \hat{R}_f}{\partial t} \right)$$

This penalizes non-physical positive derivatives with respect to water flow rate and non-physical negative degradation trajectories.
