---
title: Boiler PINN Digital Twin
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.37.1
app_file: dashboard/app.py
pinned: false
---

# Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers

> **A Cyber-Physical Work System Design (WSD) Framework for Thermodynamic State Estimation, Degradation Prognosis, and Dynamic Maintenance Scheduling**  
> *Developed for the Work System Design Course, IIT Kharagpur*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Architecture: CPS 5C](https://img.shields.io/badge/Architecture-CPS%205C-success.svg)](#cyber-physical-system-cps-5c-architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Work System Design Philosophy

A common limitation in predictive maintenance projects is treating machine learning as an isolated curve-fitting exercise on tabular data. In a true **Work System Design (WSD)** environment (as taught in Prof. Subhajit's curriculum at IIT Kharagpur), computational algorithms are only one subsystem within a socio-technical configuration of **people**, **processes**, **information**, and **physical assets**.

This repository implements an end-to-end **Physics-Informed Digital Twin (PINN-DT)** for an industrial steam and hot water boiler (Viessmann Vitorond 200 / Industrial Coal-Fired Boiler). The system fuses real-time sensor observations with first-principles thermodynamics (1st Law energy conservation) to:
1. **Estimate unobservable degradation phenomena** (fireside soot fouling resistance Rf and waterside scaling) in real time.
2. **Predict future thermal trajectories** while strictly respecting conservation laws via automatic differentiation (`torch.autograd.grad`).
3. **Mitigate operator alarm fatigue** through ISA-18.2 compliant root-cause explainability rather than opaque black-box thresholds.
4. **Optimize opportunistic maintenance scheduling** by balancing cumulative fuel waste costs against shift-dependent downtime losses and tube creep rupture risks.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               WORK SYSTEM ARCHITECTURE (STEVEN ALTER)                            │
├──────────────────────────────────────┬───────────────────────────────────────────────────────────┤
│          PHYSICAL ASSET              │                     CYBER / DIGITAL TWIN                  │
│                                      │                                                           │
│  ┌────────────────────────────────┐  │   ┌───────────────────────────────────────────────────┐   │
│  │   Industrial Boiler Asset      │  │   │            Digital Twin State Model               │   │
│  │   - Viessmann Vitorond 200     │  │   │   - Synchronized Virtual Telemetry                │   │
│  │   - Waterwall & Tubes          │  │   │   - Thermodynamic 1st Law ODE Engine              │   │
│  │   - Burner & Flue Passages     │  │   │   - Latent Degradation State Estimator            │   │
│  └───────────────┬────────────────┘  │   └─────────────────────────┬─────────────────────────┘   │
│                  │ Sensor Stream     │                             │ Physical Residuals          │
│                  ▼                   │                             ▼                             │
│  ┌────────────────────────────────┐  │   ┌───────────────────────────────────────────────────┐   │
│  │ Data Acquisition Layer         │  │   │          PyTorch PINN Prognostics Engine          │   │
│  │ (T_in, T_out, Water, Fuel, O2) ├──┼──►│  L = L_data + λ_phys L_physics + λ_mono L_mono    │   │
│  └────────────────────────────────┘  │   │  Forward: T_out(t) | Inverse: R_fouling(t)        │   │
│                                      │   └─────────────────────────┬─────────────────────────┘   │
├──────────────────────────────────────┴─────────────────────────────┼─────────────────────────────┤
│                               HUMAN-IN-THE-LOOP / DECISION LAYER   │                             │
│                                                                    ▼                             │
│  ┌────────────────────────────────┐      ┌───────────────────────────────────────────────────┐   │
│  │   Work Order & Scheduling      │      │        Cognitive Decision Support Cockpit         │   │
│  │   - Opportunistic Maintenance  │◄─────┤   - Health Index (HI) & Safe Operating Window     │   │
│  │   - Shift Labor Assignment     │      │   - Root-Cause Isolation (Fouling vs Combustion)  │   │
│  │   - Soot Blowing Work Order    │      │   - "What-If" Scenario Simulator (Load Derating)  │   │
│  └────────────────────────────────┘      └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cyber-Physical System (CPS) 5C Architecture

The project architecture maps directly to Jay Lee's CPS 5C framework:

| Level | Name | Role in this Boiler System |
| :--- | :--- | :--- |
| **C1** | **Connection** | Telemetry ingestion from temperature, pressure, water mass flow, fuel firing rate, and flue gas O2 transmitters. |
| **C2** | **Conversion** | Feature normalization, sensor health verification, and thermodynamic property lookup (enthalpy, density, heat capacity). |
| **C3** | **Cyber** | Virtual Digital Twin state model holding physical geometry, parameters, and the pure PyTorch PINN dual-head engine. |
| **C4** | **Cognition** | Composite Health Index (HI in [0, 1]), Remaining Safe Operating Window (RSOW), and ISA-18.2 root-cause diagnostic cards. |
| **C5** | **Configuration** | Dynamic opportunistic shift scheduler, automated work order generation (crew, LOTO protocol), and What-If scenario sandbox. |

---

## 3. Thermodynamic Physics & PINN Formulation

### 3.1 First-Principles Energy Balance (1st Law of Thermodynamics)
The transient thermal behavior of the boiler working fluid is governed by:

    C_sys * dT_supply/dt = Q_combustion - Q_water - Q_casing_loss    [kW]

Where:
- `Q_combustion = m_dot_fuel * LHV * eta_comb(lambda)` is chemical heat release [kW].
- `Q_water = m_dot_water * cp * (T_supply - T_return)` is sensible heat absorbed by the working fluid [kW].
- `Q_casing_loss = U_loss * A_shell * (T_supply - T_ambient)` is convective/radiative casing loss [kW].

### 3.2 Thermal Resistance & Fouling Mechanics
Fireside soot and slag deposition on heat exchange tubes introduces a conductive fouling resistance Rf(t):

    1/U(t) = 1/U_clean + R_fouling(t)

    1/Q_water = 1/Q_clean + gamma * Rf(t)

As `Rf(t)` increases, heat absorption drops, forcing higher fuel firing, elevating stack temperatures, and increasing tube metal temperatures toward metallurgical creep limits (> 560 C).

### 3.3 Pure PyTorch PINN Loss with Autograd
The neural network outputs both the predicted temperature `T_supply_hat` and the latent fouling resistance `Rf_hat`:

    [T_supply_hat, Rf_hat] = N_theta(m_dot_fuel, T_air, T_return, m_dot_water, t)

Using `torch.autograd.grad`, the model computes input derivatives and minimizes the composite loss:

    L_total = lambda_data*L_data + lambda_phys*L_physics + lambda_mono*L_mono + lambda_bound*L_boundary + lambda_inv*L_inverse

- **Physics Energy Residual Loss**:
  `L_physics = (1/M) * sum( m_dot_water * cp * (T_supply_hat - T_return) - [ 1 / (1/Q_clean + gamma*Rf_hat) ] )^2`
- **Monotonicity Prior**:
  `L_mono = (1/M) * sum( ReLU(dT_supply_hat / dm_dot_water) + ReLU(-dRf_hat / dt) )`

---

## 4. Dual-Dataset Architecture

1. **Dataset 1: Viessmann Vitorond 200 Boiler Dataset (27,280 samples)**:
   - Primary ground-truth benchmark (HySonLab/AgentIoT).
   - Features: `Fuel_Mdot`, `Tair`, `Treturn`, `Tsupply`, `Water_Mdot`, `Condition`, `Class`.
   - Continuous numerical degradation labels: F in [0.01, 0.46] (Fouling), S in [0.01, 0.46] (Scaling), Nominal, Lean, and Excess Air.
   - Used for rigorous quantitative validation of PINN inverse parameter identification.
2. **Dataset 2: Real Industrial Coal-Fired Boiler Operations Telemetry (14,400 samples)**:
   - High-resolution 5-second sampling telemetry capturing superheated steam temperature (`TE_8332A`), drum pressure, flue gas O2, and draft fan currents.
   - Demonstrates Digital Twin state synchronization under realistic industrial noise and disturbance shifts.

---

## 5. Experimental Benchmark Ladder & Stress Tests

### 6-Model Comparative Benchmark
| Model | Type | Train Data | Test RMSE (K) | Test MAE (K) | Test R2 | Energy Residual (kW) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 0** | Physics-Only Analytical (1st Law) | 0 samples | 7.37 K | 5.51 K | -0.611 | **0.000 kW** | 0.0078 ms |
| **Model 1** | Polynomial Ridge Regression | 17,902 | 4.22 K | 3.46 K | 0.472 | 143.62 kW | 0.0008 ms |
| **Model 2** | Random Forest Regressor | 6,000 | 4.56 K | 3.64 K | 0.384 | 142.21 kW | 0.0204 ms |
| **Model 3** | Standard Deep MLP (Data-Only) | 17,902 | 4.23 K | 3.45 K | 0.468 | 134.47 kW | 0.0029 ms |
| **Model 4** | Recurrent LSTM Network | 17,902 | 4.20 K | 3.46 K | 0.476 | 142.66 kW | 0.0097 ms |
| **Model 5** | **Physics-Informed Neural Network (PINN)** | 17,902 | 5.26 K | 4.04 K | 0.181 | **44.46 kW** (**>3x reduction**) | 0.0029 ms |

### Stress Test Experiments
1. **Data Scarcity Ablation**: Evaluating on 1%, 5%, 10%, 25%, 50%, 100% of data. PINN retains high physical consistency even with only 1% training data.
2. **Out-of-Distribution (OOD) Extrapolation**: Testing models on extreme peak loads (m_dot_fuel >= 3.5 kg/s, m_dot_water >= 10.5 kg/s). Pure ML produces unphysical predictions, while PINN remains bounded.
3. **Sensor Noise Robustness**: Contaminating sensor inputs with 0% to 20% Gaussian noise. Physics regularization acts as a denoising regularizer.
4. **Physics Loss Weight Sensitivity**: Sweeping lambda_phys in [0.0, 1.0].

---

## 6. Interactive Control Room Cockpit (Streamlit)

Launch the interactive industrial dashboard:

```bash
streamlit run dashboard/app.py
```

### Cockpit Capabilities:
- **Tab 1: Digital Twin Live Cockpit**: Real-time telemetry, 1st Law Sankey energy balance, live Health Index gauge, and ISA-18.2 alarm banner.
- **Tab 2: Prognostics & RSOW**: Degradation trajectory tracking soot build-up toward the critical Rf = 0.035 threshold, with remaining hours countdown.
- **Tab 3: PINN & Physics Residual Inspector**: Live autograd loss curves, energy conservation residual check, and 6-model leaderboard.
- **Tab 4: 'What-If' Scenario Studio**: Interactive sliders for load (50%–100%) and soot-blowing intervention timing, simulating future tube temperatures and fuel waste.
- **Tab 5: Maintenance Scheduler & Work Orders**: Cost-optimal shift scheduling (off-peak night window selection) and automated industrial Work Order generation with Lockout/Tagout (LOTO) protocols.

---

## 7. Project Directory Layout

```
boiler-pinn-digital-twin/
├── README.md                          <- Project documentation & WSD defense overview
├── requirements.txt                   <- PyTorch, Streamlit, Plotly, Pandas, Scikit-learn
├── configs/
│   └── default_config.yaml           <- Boiler geometry, thermal constants, loss weights
├── data/
│   ├── download_datasets.py          <- Automated downloader & timeseries generator
│   ├── raw/                          <- Downloaded raw datasets
│   └── processed/                    <- Preprocessed train/val/test/OOD splits
├── docs/
│   ├── system_architecture.md        <- WSD framework & CPS 5C breakdown
│   ├── physics_derivation.md         <- First-principles thermodynamics derivation
│   └── maintenance_strategy.md       <- Condition-based & opportunistic scheduling theory
├── uml/
│   ├── use_case_diagram.puml         <- Actor use cases in PlantUML
│   ├── sequence_diagram.puml         <- Ingestion-Twin-PINN-Operator workflow
│   ├── component_cps_diagram.puml    <- CPS 5-layer component architecture
│   └── statechart_diagram.puml       <- Asset health lifecycle state machine
├── src/
│   ├── physics/
│   │   ├── boiler_thermo.py          <- 1st Law energy balance & autograd residual
│   │   ├── fouling_model.py          <- Kern-Seaton degradation & fuel waste economics
│   │   └── preprocessor.py           <- Pipeline, normalization, and OOD split
│   ├── models/
│   │   ├── baselines.py              <- Model 0-4: Physics ODE, Ridge, RF, MLP, LSTM
│   │   └── pinn_model.py             <- Model 5: Pure PyTorch PINN with autograd
│   ├── digital_twin/
│   │   ├── state.py                  <- State representations & parameters
│   │   ├── synchronizer.py           <- Streaming telemetry synchronizer
│   │   └── what_if_simulator.py      <- Scenario sandbox (load derating analysis)
│   └── maintenance/
│       ├── health_index.py           <- Composite Health Index & RSOW
│       ├── anomaly_detector.py       <- ISA-18.2 alarm classifier & root-cause isolation
│       └── scheduler.py              <- Cost-optimal opportunistic shift scheduler
├── experiments/
│   ├── run_all_benchmarks.py         <- Master 6-model comparative evaluation
│   ├── exp_data_scarcity.py          <- 1% to 100% data scarcity study
│   ├── exp_ood_extrapolation.py      <- Out-of-distribution stress test
│   ├── exp_noise_robustness.py       <- Sensor noise robustness test
│   └── exp_physics_ablation.py       <- Lambda physics sensitivity study
├── dashboard/
│   └── app.py                        <- 5-tab Streamlit control room cockpit
└── tests/
    ├── test_physics.py               <- Thermodynamics & 1st Law unit tests
    ├── test_pinn.py                  <- PINN autograd & gradient flow tests
    └── test_digital_twin.py          <- Twin sync, alarm, and scheduler tests
```

---

## 8. Verification & Running Tests

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Run the experimental benchmark suite:

```bash
python experiments/run_all_benchmarks.py
python experiments/exp_data_scarcity.py
python experiments/exp_ood_extrapolation.py
python experiments/exp_noise_robustness.py
python experiments/exp_physics_ablation.py
```

---

## 9. Academic Citations & Acknowledgements

- **Work System Design**: Prof. Subhajit, IIT Kharagpur Course Curriculum.
- **Alter, S. (2013)**. *Work System Theory: Overview of Core Concepts, Extensions, and Challenges for the Future*. Journal of the Association for Information Systems.
- **Lee, J., Bagheri, B., & Kao, H. A. (2015)**. *A Cyber-Physical Systems architecture for Industry 4.0-based manufacturing systems*. Manufacturing Letters.
- **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics.
- **HySonLab / AgentIoT**: Viessmann Vitorond 200 Boiler Dataset (Shohet, Kandil & McArthur).
- **ISA-18.2 / IEC 62682**: *Management of Alarm Systems for the Process Industries*.
