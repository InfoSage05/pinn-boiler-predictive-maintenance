# MASTER PROJECT GENERATION PROMPT
## GA-Based Digital Twin for Boiler Predictive Maintenance | IIT Kharagpur

---

> **HOW TO USE:** Copy everything below the horizontal rule and paste it directly into your Antigravity IDE chat. The agent will scaffold the entire project from scratch.

---

---

# PROJECT SCAFFOLD REQUEST — GENETIC ALGORITHM DIGITAL TWIN FOR BOILER PREDICTIVE MAINTENANCE

## Academic Context
- **Institution:** Indian Institute of Technology, Kharagpur
- **Course:** Work System Design (ME/IE elective)
- **Instructor:** Prof. Subhajit
- **Project Type:** End-to-end research-grade software project with a live interactive dashboard and a 14-slide presentation content pack
- **Framework:** PyTorch (primary ML framework)

---

## Mission

Build a **complete, production-quality, research-grade project** from scratch titled:

> **"Genetic Algorithm-Accelerated Digital Twin for Boiler Predictive Maintenance"**

The project must be fully functional with real runnable Python code, a Streamlit dashboard, auto-generated presentation assets, and all documentation. Do NOT produce placeholder code. Every file must be importable and executable. The entire project must run end-to-end with `python main.py` and `streamlit run dashboard/app.py`.

---

## 1. Project Overview

### Problem Statement
Industrial boilers degrade over time due to fouling, scaling, corrosion, and tube creep. Unplanned failures cost millions in downtime. Traditional preventive maintenance is wasteful; reactive maintenance is catastrophic. The goal is to build a **Digital Twin (DT)** of the boiler that mirrors real-time physical state and uses a **Genetic Algorithm (GA)** to continuously optimize the DT's parameters, enabling accurate **Remaining Useful Life (RUL)** prediction and **optimal maintenance scheduling**.

### Core Innovation
Instead of a fixed physics model, we use a **GA-calibrated Digital Twin**:
1. A physics-based ODE model simulates boiler thermal-hydraulic behavior.
2. The GA evolves a population of parameter sets (fouling resistance, heat transfer coefficients, degradation rates) to minimize the difference between DT outputs and real (or synthetic) sensor data.
3. The calibrated DT then predicts RUL and flags maintenance windows.
4. PyTorch is used for differentiable physics components, surrogate modeling, and gradient-guided GA mutation operators.

---

## 2. Theoretical Foundations to Implement

### 2.1 Digital Twin Physics Model (implement as PyTorch nn.Module)
Implement a first-principles boiler ODE system:

**Energy Balance (Evaporator):**
```
m_w * Cp_w * dT_w/dt = Q_fire - Q_steam - Q_loss
Q_fire = η * m_fuel * LHV
Q_loss = U_eff * A * (T_gas - T_w)
U_eff = 1 / (1/h_fire + R_fouling + t_wall/k_wall + 1/h_water)
```

**Kern-Seaton Fouling Kinetics:**
```
dR_f/dt = φ_d - φ_r
φ_d = K_d * v^(-0.8) * C_f   (deposition rate)
φ_r = K_r * R_f * τ_w        (removal rate)
R_f_asymptote = K_d/(K_r * τ_w)
```

**Larson-Miller Creep Rupture (tube life):**
```
LMP = T * (C + log10(t_r))   [Larson-Miller Parameter]
σ_hoop = P * D / (2 * t)     [Hoop stress]
RUL = 10^((LMP/T) - C)       [Remaining life in hours]
```

**Implement these as differentiable PyTorch modules** so gradients can flow through them.

### 2.2 Genetic Algorithm Engine (implement from scratch in PyTorch/NumPy)
Implement a full GA with the following:

**Chromosome / Individual:**
```python
# Each individual encodes boiler DT parameters
chromosome = {
    'fouling_resistance_Rf': float,      # m²K/W, range [0, 0.002]
    'heat_transfer_coeff_h': float,      # W/m²K, range [1000, 8000]
    'combustion_efficiency_eta': float,  # range [0.75, 0.98]
    'degradation_rate_kd': float,        # range [1e-6, 1e-3]
    'removal_rate_kr': float,            # range [1e-5, 1e-2]
    'wall_thickness_t': float,           # m, range [0.003, 0.015]
}
```

**GA Operators:**
- **Selection:** Tournament selection (k=3) + Elitism (top 10%)
- **Crossover:** Simulated Binary Crossover (SBX) with distribution index η_c = 20
- **Mutation:** Polynomial mutation with p_m = 1/n_genes, η_m = 20
- **Fitness Function:** Weighted RMSE between DT predictions and sensor readings:
```
fitness = 1 / (w1*RMSE_temp + w2*RMSE_pressure + w3*RMSE_efficiency + ε)
```

**GA Hyperparameters (expose as config):**
- Population size: 100
- Generations: 200
- Crossover probability: 0.9
- Mutation probability: 1/n_genes
- Elite fraction: 0.1

**PyTorch Integration:** Use `torch.autograd` to compute gradient of fitness w.r.t. chromosome parameters and use it as a "gradient-guided mutation" operator — a hybrid GA-gradient approach.

### 2.3 Surrogate Model (PyTorch MLP)
Train a surrogate neural network to approximate the expensive physics simulation:
- **Input:** chromosome parameters (6D) + operating conditions (load, ambient temp)
- **Output:** predicted KPIs (steam temp, efficiency, RUL)
- **Architecture:** 4-layer MLP [8 → 64 → 128 → 64 → 3], GELU activations, BatchNorm
- **Training:** Adam optimizer, MSE loss, trained on GA-evaluated data
- Use the surrogate to pre-screen the population before expensive physics evaluation.

---

## 3. Complete File & Directory Structure to Create

```
Project_GA_DT_Boiler/
│
├── README.md                          # Full academic README
├── requirements.txt                   # All dependencies
├── main.py                            # Entry point: runs GA + generates report
├── config.yaml                        # All hyperparameters and config
│
├── models/
│   ├── __init__.py
│   ├── digital_twin.py                # PyTorch boiler DT ODE model
│   ├── ga_engine.py                   # Full Genetic Algorithm implementation
│   ├── surrogate_net.py               # MLP surrogate model (PyTorch)
│   ├── degradation_model.py           # Kern-Seaton fouling + Larson-Miller creep
│   └── rul_predictor.py               # RUL prediction and confidence intervals
│
├── simulation/
│   ├── __init__.py
│   ├── boiler_ode.py                  # scipy ODE integration of physics model
│   ├── sensor_simulator.py            # Synthetic sensor data generator
│   └── data_pipeline.py              # Data loading, normalization, batching
│
├── optimization/
│   ├── __init__.py
│   ├── fitness_functions.py           # All fitness/objective functions
│   ├── operators.py                   # SBX crossover, polynomial mutation
│   └── hybrid_gradient_mutation.py    # PyTorch autograd + GA mutation
│
├── maintenance/
│   ├── __init__.py
│   ├── scheduler.py                   # Opportunistic maintenance scheduling
│   ├── cost_model.py                  # Economic cost-benefit analysis
│   └── work_order.py                  # LOTO / work order generation
│
├── dashboard/
│   ├── app.py                         # Streamlit dashboard (main)
│   ├── components/
│   │   ├── ga_live_plot.py            # Real-time GA convergence plot
│   │   ├── twin_monitor.py            # Digital twin state monitor
│   │   ├── rul_gauge.py               # RUL gauge and confidence band
│   │   └── maintenance_calendar.py    # Maintenance scheduling calendar
│   └── assets/
│       └── style.css                  # Custom CSS for dark-mode dashboard
│
├── scripts/
│   ├── generate_ppt_assets.py         # Auto-generate all presentation graphics
│   ├── run_ablation_study.py          # Ablation: GA vs random search vs gradient-only
│   └── benchmark.py                   # Benchmark GA speed with/without surrogate
│
├── ppt_assets/                        # Generated presentation graphics (PNG, 16:9)
│   ├── 01_title_slide.png
│   ├── 02_problem_statement.png
│   ├── 03_work_system_framework.png   # Steven Alter WSD framework applied
│   ├── 04_cps_5c_architecture.png     # Jay Lee CPS 5C applied to boiler
│   ├── 05_digital_twin_architecture.png
│   ├── 06_ga_algorithm_flowchart.png
│   ├── 07_chromosome_encoding.png
│   ├── 08_fitness_landscape.png
│   ├── 09_fouling_kinetics_plot.png
│   ├── 10_rul_prediction_curves.png
│   ├── 11_ga_convergence_comparison.png
│   ├── 12_maintenance_schedule_gantt.png
│   ├── 13_results_kpi_dashboard.png
│   └── 14_conclusion_future_work.png
│
├── docs/
│   ├── system_architecture.md         # Full system design document
│   ├── ga_theory.md                   # Mathematical derivation of all GA operators
│   ├── physics_model.md               # Boiler physics equations and derivations
│   └── api_reference.md               # Auto-generated API docs
│
├── tests/
│   ├── test_digital_twin.py
│   ├── test_ga_engine.py
│   ├── test_surrogate.py
│   └── test_rul_predictor.py
│
└── PPT_CONTENT_PACK.md                # Complete slide-by-slide content for 14-slide deck
```

---

## 4. Detailed Implementation Requirements

### 4.1 `models/digital_twin.py`
```python
# Must implement:
class BoilerDigitalTwin(nn.Module):
    """
    Differentiable Digital Twin of an industrial fire-tube boiler.
    Parameters are GA-optimized chromosome values.
    Implements energy balance ODE + fouling kinetics as differentiable operations.
    """
    def __init__(self, config: dict): ...
    def forward(self, operating_conditions: torch.Tensor, time_steps: torch.Tensor) -> dict: ...
    def compute_efficiency(self, ...) -> torch.Tensor: ...
    def compute_steam_temp(self, ...) -> torch.Tensor: ...
    def compute_rul(self, ...) -> torch.Tensor: ...
    def update_parameters(self, chromosome: dict): ...
```

### 4.2 `models/ga_engine.py`
```python
class GeneticAlgorithm:
    """
    Full GA with:
    - Tournament selection with elitism
    - SBX crossover  
    - Polynomial mutation
    - Optional gradient-guided mutation (PyTorch autograd)
    - Surrogate-assisted evaluation
    - Convergence tracking and diversity metrics
    """
    def __init__(self, config: GAConfig): ...
    def initialize_population(self) -> List[Individual]: ...
    def evaluate_population(self, population, digital_twin, sensor_data) -> List[float]: ...
    def selection(self, population, fitnesses) -> List[Individual]: ...
    def crossover(self, parent1, parent2) -> Tuple[Individual, Individual]: ...
    def mutate(self, individual) -> Individual: ...
    def gradient_guided_mutation(self, individual, digital_twin) -> Individual: ...
    def run(self, digital_twin, sensor_data, generations=200) -> Tuple[Individual, History]: ...
    def plot_convergence(self, history) -> plt.Figure: ...
```

### 4.3 `dashboard/app.py` — Streamlit Dashboard
The dashboard must have:
1. **Sidebar:** Configuration panel for GA hyperparameters, boiler operating conditions
2. **Tab 1 — Live GA Optimization:** Real-time convergence plot (best/avg/worst fitness), population diversity gauge, current best chromosome values
3. **Tab 2 — Digital Twin Monitor:** Time-series plots of steam temperature, pressure, efficiency vs. DT predictions; fouling resistance accumulation curve
4. **Tab 3 — RUL & Health Index:** Gauge meter for RUL (days), confidence interval band, health index 0–100, degradation trajectory
5. **Tab 4 — Maintenance Scheduler:** Gantt chart of optimal maintenance windows, cost-benefit analysis table, work order summary
6. **Header:** "GA-DT Boiler Predictive Maintenance | IIT Kharagpur | Work System Design"
7. **Theme:** Dark mode with industrial blue/orange color palette

### 4.4 `PPT_CONTENT_PACK.md` — 14 Slides
Generate a complete, detailed content pack with the following slides:

| # | Slide Title | Key Content |
|---|-------------|-------------|
| 1 | Title | Project name, team, IIT Kharagpur, course, date |
| 2 | Problem Statement | Unplanned downtime stats, boiler failure modes, cost of reactive maintenance |
| 3 | Work System Framework (Steven Alter) | WSD 9 elements applied to boiler maintenance work system |
| 4 | CPS 5C Architecture (Jay Lee) | Connection → Conversion → Cyber → Cognition → Configuration applied to DT |
| 5 | Digital Twin Architecture | Multi-layer DT: physical layer → data layer → analytics layer → service layer |
| 6 | Genetic Algorithm — Theory | Chromosome encoding, operators, fitness landscape visualization |
| 7 | GA Operators Deep Dive | SBX crossover math, polynomial mutation, tournament selection, elitism |
| 8 | Physics Model | Boiler ODE, Kern-Seaton fouling, Larson-Miller creep equations |
| 9 | Fouling Kinetics & RUL | Fouling buildup curves, RUL prediction with confidence intervals |
| 10 | GA Convergence Results | Convergence plots, comparison: GA vs PSO vs Random Search |
| 11 | Ablation Study | Effect of population size, crossover rate, surrogate acceleration |
| 12 | Maintenance Scheduling | Gantt chart, opportunistic maintenance economics, NPV analysis |
| 13 | KPI Dashboard Results | RMSE, MAPE, RUL accuracy, fuel savings %, CO2 reduction |
| 14 | Conclusion & Future Work | Summary of contributions, limitations, future: multi-objective GA, NSGA-II |

---

## 5. Key Academic Frameworks to Apply

### 5.1 Work System Design (Steven Alter, 2013)
Apply the 9-element WSD framework:
- **Customers:** Plant operators, maintenance engineers, plant manager
- **Products/Services:** RUL predictions, maintenance alerts, optimized schedules
- **Processes/Activities:** Data acquisition → GA optimization → DT calibration → RUL prediction → scheduling
- **Participants:** IoT sensors, GA engine, DT model, dashboard UI, human operators
- **Information:** Sensor telemetry, DT state, GA population history, maintenance logs
- **Technology:** PyTorch, Genetic Algorithm, ODE solver, Streamlit, SCADA interface
- **Environment:** Industrial boiler plant, IIT Kharagpur lab (simulation)
- **Infrastructure:** NVIDIA GPU (PyTorch), SCADA system, cloud storage
- **Strategies:** Predictive vs. preventive vs. reactive — quantify the shift

### 5.2 CPS 5C Architecture (Jay Lee, 2015)
Map each layer to the project:
- **Connection:** IoT sensors (temperature, pressure, flow, vibration) → MQTT broker
- **Conversion:** Feature extraction, ODE integration, PyTorch inference
- **Cyber:** Digital Twin model, GA optimization engine, surrogate network
- **Cognition:** RUL prediction, health index scoring, anomaly detection
- **Configuration:** Maintenance schedule output, LOTO work orders, operator alerts

---

## 6. requirements.txt (Generate This Exactly)
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
scikit-learn>=1.3.0
deap>=1.4.0
pytest>=7.4.0
black>=23.0.0
```

---

## 7. Synthetic Data Generation (`simulation/sensor_simulator.py`)
Generate realistic synthetic sensor time-series data:
- **Duration:** 8760 hours (1 year of operation)
- **Sampling rate:** 1 hour
- **Channels:** steam_temp (°C), steam_pressure (bar), flue_gas_temp (°C), feed_water_temp (°C), fuel_flow (kg/h), steam_flow (kg/h), vibration_rms (mm/s)
- **Degradation injection:** linearly increasing fouling resistance from 0 to R_f_max over the year
- **Noise:** Gaussian noise (σ = 1% of signal) + occasional sensor spikes (0.1% probability)
- **Anomaly events:** 3 injected anomalies at hours 2000, 5000, 7500

---

## 8. Results to Report (Compute and Display These)
1. **GA Convergence:** Plot best fitness vs. generation; show convergence in <50 generations for well-tuned hyperparameters
2. **DT Accuracy:** RMSE < 2°C for steam temperature, RMSE < 0.1 bar for pressure
3. **RUL Prediction:** Mean Absolute Percentage Error (MAPE) < 5% against ground truth
4. **Maintenance Savings:** Quantify: 30-40% reduction in unplanned downtime, 15-20% fuel cost savings
5. **Ablation Study:** GA vs. Random Search (GA should outperform by >40% in fitness)
6. **Surrogate Speedup:** Show 10x speed improvement with surrogate-assisted GA

---

## 9. Code Quality Requirements
- **All code must be fully runnable** — no `pass` statements or TODOs in core logic
- **Docstrings:** Google-style docstrings on every class and method
- **Type hints:** Full type annotations throughout
- **Logging:** Use Python `logging` module with INFO/DEBUG levels
- **Config-driven:** All hyperparameters must come from `config.yaml`, not hardcoded
- **Reproducibility:** Set `torch.manual_seed(42)`, `np.random.seed(42)` at entry points
- **Tests:** Pytest unit tests for all core modules (>80% coverage target)

---

## 10. Execution Instructions to Generate
Add a `QUICKSTART.md` with:
```bash
# Setup
git clone <repo>
cd Project_GA_DT_Boiler
pip install -r requirements.txt

# Run main GA optimization
python main.py --config config.yaml --generations 200 --population 100

# Launch interactive dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v

# Generate presentation assets
python scripts/generate_ppt_assets.py
```

---

## FINAL INSTRUCTION TO THE AI AGENT

**Generate ALL files listed above in their entirety.** Start with `requirements.txt` and `config.yaml`, then build the models layer by layer (physics → GA → surrogate → RUL), then the dashboard, then scripts, then documentation, and finally the `PPT_CONTENT_PACK.md`. 

Do not summarize or truncate any file. Every Python file must be complete and importable. The dashboard must launch without errors. The GA must run and produce convergence plots. All 14 presentation slides must have fully written content.

**Institution:** IIT Kharagpur | **Course:** Work System Design | **Instructor:** Prof. Subhajit
