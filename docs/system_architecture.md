# Cyber-Physical Work System Design (WSD) Architecture

## 1. Academic & Theoretical Foundation

This project implements a **Physics-Informed Digital Twin (PINN-DT)** within a formal **Work System Design (WSD)** framework, directly aligned with the curriculum of the Work System Design course at IIT Bhilai.

In classical industrial engineering, predictive models are often developed in an algorithmic vacuum without considering the humans, organizational processes, decision contexts, or operational workflows surrounding the physical asset. In contrast, this project designs the computational intelligence (PyTorch PINN) as an integrated subsystem within a complete socio-technical work system.

---

## 2. The Work System Framework (Steven Alter)

Steven Alter's Work System Framework analyzes operational systems across nine fundamental elements:

```
                  ┌──────────────────────────────────────────────┐
                  │                 ENVIRONMENT                  │
                  │  - Ambient thermal conditions                │
                  │  - Grid power demand & production load       │
                  │  - Environmental emissions regulations       │
                  └──────────────────────┬───────────────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────────────────────┐
│                                   WORK SYSTEM                                   │
│                                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────────────┐  │
│  │     PARTICIPANTS     │  │      PROCESSES       │  │      INFORMATION      │  │
│  │ - Control Operator   │  │ - Telemetry Monit.   │  │ - Raw Sensor Data     │  │
│  │ - Maintenance Tech   │  │ - Anomaly Triage     │  │ - Virtual Twin State  │  │
│  │ - Reliability Eng.   │  │ - RSOW Prognosis     │  │ - Physics Residuals   │  │
│  │ - Plant Ops Manager  │  │ - Shift Scheduling   │  │ - Health Index (HI)   │  │
│  │                      │  │ - Soot Blowing LOTO  │  │ - Work Order Tickets  │  │
│  └──────────────────────┘  └──────────────────────┘  └───────────────────────┘  │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                TECHNOLOGIES                               │  │
│  │  - Industrial Steam Boiler Asset (Viessmann Vitorond 200)                 │  │
│  │  - Process Transmitters (T, P, Flow, Flue Gas O2)                         │  │
│  │  - Time-Series Historian (DuckDB / SQLite / Telemetry Replay)             │  │
│  │  - PyTorch PINN Engine (Forward Prediction + Inverse Degradation Head)    │  │
│  │  - Digital Twin State Synchronizer & What-If Scenario Simulator          │  │
│  │  - Cognitive Operator Decision Cockpit (Streamlit / Plotly SCADA)         │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                             PRODUCTS & SERVICES                           │  │
│  │  - Continuous, safe, energy-efficient high-pressure steam generation      │  │
│  │  - Zero catastrophic unplanned boiler tube rupture downtime               │  │
│  │  - Cost-optimal predictive maintenance work packages                      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                  ┌──────────────────────▼───────────────────────┐
                  │                  CUSTOMERS                   │
                  │  - Downstream chemical / manufacturing units │
                  │  - Steam turbine power generation            │
                  │  - Plant economic stakeholders               │
                  └──────────────────────────────────────────────┘
```

---

## 3. Cyber-Physical System (CPS) 5C Architecture (Jay Lee)

The system adheres to the industry-standard 5C CPS architecture:

### Level 1: Connection (Physical Sensing)
- Temperature Transmitters (Feedwater Return, Steam Supply, Flue Gas).
- Flow Transmitters (Water Mass Flow, Fuel Oil/Gas Firing Rate).
- Pressure Transmitters (Upper Furnace Pressure, Steam Drum Pressure).
- Flue Gas Analyzers (Excess Oxygen % $O_2$, Flue Gas Stack Temp).

### Level 2: Conversion (Data-to-Information)
- Noise filtering and outlier rejection.
- Thermodynamic physical property mapping (enthalpy, density, heat capacity).
- Real-time heat transfer rate calculation: $\dot{Q}_{water} = \dot{m}_w c_p (T_{out} - T_{in})$.

### Level 3: Cyber (Digital Twin & PINN)
- Digital Shadow virtual representation of the boiler asset.
- PyTorch PINN forward model: predicts future supply temperature constrained by 1st Law energy conservation.
- PyTorch PINN inverse model: estimates unobservable latent fouling resistance $R_f(t)$ and heat transfer coefficient $U(t)$.
- Continuous physics residual evaluation: $\Delta \dot{Q} = |\dot{Q}_{in} - \dot{Q}_{out} - \dot{Q}_{loss}|$.

### Level 4: Cognition (Prognostics & Explainability)
- Multi-criteria Health Index ($HI \in [0, 1]$) combining fouling severity, tube metal overheating margin, and energy balance error.
- Prognostic calculation of Remaining Safe Operating Window ($RSOW$).
- ISA-18.2 compliant root-cause diagnosis (Fireside Soot Fouling vs. Waterside Scale vs. Combustion Excess Air).
- Mitigation of operator alarm fatigue through transparent physical explanations.

### Level 5: Configuration (Supervisory Feedback & Execution)
- What-If Scenario Sandbox: enables operators to simulate load derating (e.g. 90% $\rightarrow$ 75%) to extend asset survival.
- Cost-Optimal Opportunistic Scheduler: dynamically identifies the optimal maintenance shift (e.g., night off-peak window) minimizing production loss and fuel waste.
- Automated Work Order Generation with LOTO safety protocols, crew assignments, and required replacement gaskets/nozzles.
