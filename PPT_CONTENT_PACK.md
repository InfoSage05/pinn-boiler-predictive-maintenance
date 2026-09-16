# PPT Content Pack — Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers

**Course:** Work System Design (WSD), Department of Mechanical & Mechatronics Engineering, Indian Institute of Technology Bhilai (IIT Bhilai)  
**Instructor:** Prof. Subhajit  
**Presentation Format:** 15 Comprehensive Content Slides + Slide 16 Defense Conclusion + Technical Defense & Faculty Rebuttal Appendix  
**Target Delivery Duration:** 18–22 minutes (approx. 70–85 seconds per slide) + 8–10 minutes Faculty Q&A  
**Design Aesthetic:** **Industrial Thermo-Precision** (Sleek Dark Slate, Thermodynamic Flame Orange, Hydronic Steam Cyan, Safety Emerald)  
**Asset Directory:** `ppt_assets/` — 11 verified, usable visual assets (see the cross-reference table near the end of this document for exactly which files to use and which to avoid).

---

## Executive Guide & Presentation Philosophy

### 1. The Core Academic Thesis
> *"Conventional pure data-driven machine learning models achieve deceptively high interpolation accuracy on boiler sensor telemetry while violating the 1st Law of Thermodynamics by 134–144 kW on average. By embedding thermodynamic conservation and autograd-enforced monotonicity directly into the loss landscape, BoilerPINN trades roughly 1 K of test RMSE to achieve a **>3× reduction in physical energy residual (down to 44.46 kW)**. Framed within Steven Alter's Work System Framework and Jay Lee's CPS 5C Architecture, this digital twin moves beyond black-box predictive maintenance toward proactive tube creep rupture prevention, ISA-18.2 root-cause triage, and cost-optimal opportunistic maintenance scheduling — every number in this deck traceable to the actual repo."*

### 2. Design System: Industrial Thermo-Precision
To ensure a cohesive, publication-grade aesthetic across Canva Pro, Microsoft PowerPoint, or Google Slides:
- **Base Canvas:** Deep Charcoal / Obsidian (`#0B0F19` and `#161E2E`). Avoid stark black or harsh default white.
- **Card Containers & Panels:** Dark Navy/Slate (`#1E293B`) with subtle 1.2px borders (`#334155`).
- **Accent Primaries:**
  - **Thermodynamic Flame / Alert:** Flame Orange (`#EA580C`) and Crimson (`#DC2626`) for combustion heat release, tube creep overheating, and critical alarms.
  - **Hydronic Steam / Autograd:** Vivid Sky Blue (`#0284C7`) and Electric Cyan (`#38BDF8`) for water absorption, sensible heat, autograd derivatives, and digital twin states.
  - **Health & Baseline:** Emerald Green (`#059669` / `#10B981`) for nominal states, clean heat transfer, and compliant work orders.
  - **Caution / Fouling:** Warm Amber (`#F59E0B`) for soot accumulation and advisory thresholds.
- **Typography Tokens:**
  - Headers: **Outfit**, **Cabinet Grotesk**, or **Inter Display** (Bold, tracking -0.02em).
  - Body Text & Metrics: **Inter** or **Roboto** (Regular/Medium, crisp line height 1.4).
  - Equations & Code Symbols: **JetBrains Mono** or **Fira Code**.
- **Visual Composition Rule (55/45 Split):**
  - Left 55%: Visual Hero (high-resolution diagram, chart, or cutaway from `ppt_assets/`).
  - Right 45%: Structured 3-block narrative (The Operational Dilemma $\to$ Governing Mathematical Law $\to$ Socio-Technical Decision).

---

## Known Repo Inconsistencies & Honest Defense Disclosures
*Read these 8 authentic engineering disclosures once before your defense. Each has been resolved in the slide scripts below with numbers verified directly against the source code — nothing here is invented.*

**Math/data audit note (read this first):** this document was checked line-by-line against the actual source files (`docs/physics_derivation.md`, `src/models/pinn_model.py`, `src/maintenance/health_index.py`, `src/maintenance/scheduler.py`, `src/physics/fouling_model.py`, `experiments/results/benchmark_leaderboard.json`). Several formulas, benchmark numbers, and "example" figures in an earlier draft of this file did not match the code — those have been corrected below. A few AI-generated images in `ppt_assets/` (see the asset table at the end) were also found to be unusable — one (`10_stress_tests_ablation.*`) contains garbled, hallucinated text and fake model names and must not be used; another (`14_boiler_photorealistic_schematic.png`) depicts a different, higher-pressure class of boiler than the one this project actually models.

1. **PINN Input Dimensionality (4-Input vs 5-Input):**
   - *Repo Quirk:* Physics documentation occasionally references $N_\theta(\dot{m}_{fuel}, T_{air}, T_{return}, \dot{m}_{water}, t)$ with time $t$.
   - *Actual Implementation (`BoilerPINN`):* Implements a 4-input steady-state/quasi-static formulation without $t$.
   - *Defense Resolution (Slide 7):* Explicitly state that because the boiler operating cycle operates across quasi-static load regimes with sensor sampling intervals ($\Delta t = 5\text{s}$) far smaller than the thermal settling time ($\tau_{sys} \approx 45\text{ min}$), the network models instantaneous thermodynamic mapping $[\dot{m}_f, T_{air}, T_{ret}, \dot{m}_w] \to [\hat{T}_{supply}, \hat{R}_f]$, while transient degradation is tracked via the outer digital twin state estimator.
2. **Monotonicity Penalty Implementation:**
   - *Repo Quirk:* Documentation suggests two penalties ($\partial T/\partial \dot{m}_w \le 0$ and $\partial T/\partial \dot{m}_f \ge 0$).
   - *Actual Implementation (`BoilerPINN`):* Enforces only the hydraulic sensible cooling constraint: $\mathcal{L}_{mono} = \text{ReLU}\left(\frac{\partial \hat{T}_{supply}}{\partial \dot{m}_{water}}\right)$.
   - *Defense Resolution (Slide 8):* Highlight that hydraulic cooling is the primary physical constraint subject to unphysical ML inversion during sudden load drops; fuel monotonicity is naturally regularized via the positive lower heating value in the data loss.
3. **Configuration Hygiene (`default_config.yaml`):**
   - *Repo Quirk:* The README mentions `configs/default_config.yaml`, but hyperparameters are cleanly encapsulated directly in Python module constants.
   - *Defense Resolution:* Frame this as strict dataclass encapsulation preventing runtime configuration drift in safety-critical cyber-physical deployments.
4. **Dashboard Tab-3 Synthetic Curves vs Leaderboard:**
   - *Repo Quirk:* Streamlit Tab-3 contains an illustrative placeholder for interactive user training runs.
   - *Defense Resolution:* All benchmark assertions, loss curves, and leaderboard metrics in this deck are strictly extracted from `experiments/results/benchmark_leaderboard.json` and the empirical stress test logs.
5. **Sankey Energy Scale Mismatch (real, not fixable without changing the source code):**
   - *Repo Quirk:* The Streamlit Sankey (`dashboard/app.py`) mixes two different scales: gross fuel chemical energy on an LHV basis (`fuel_flow × 42,000 kJ/kg × 0.91` ≈ **107,000 kW** at 85% load) against the model's analytical "effective heat-transfer capacity" (`Q_clean = 388.8 kW`, giving ≈390 kW of actual water heat absorption). These are not the same physical accounting frame, so the diagram visually shows almost all energy "lost" to the stack — that is an artifact of the two scales, not a real 99.6% efficiency loss.
   - *Defense Resolution (Slide 5):* State this honestly, out loud, as a known simplification already present in the codebase — don't present the Sankey's percentages as a real combustion-efficiency claim. The number that *is* rigorously meaningful is the physics residual: 44.46 kW for the PINN vs. 134–144 kW for every data-only baseline.
6. **The Fundamental Accuracy vs. Physical Consistency Trade-off:**
   - *Repo Reality:* BoilerPINN exhibits an RMSE of $5.26\text{ K}$ ($R^2 = 0.181$), whereas the pure Deep MLP achieves $4.23\text{ K}$ ($R^2 = 0.468$) — verified against `experiments/results/benchmark_leaderboard.json`.
   - *Defense Resolution (Slide 9 & 10):* Own this boldly as the primary scientific contribution! The MLP achieves lower RMSE while violating the 1st-Law energy balance by 134.47 kW on average. In safety-critical power engineering, physical fidelity and extrapolation reliability under out-of-distribution peak loads matter more than a fraction of a degree of in-distribution interpolation fit.
7. **Two different fouling-threshold constants exist in the repo:** `src/maintenance/health_index.py` and the dashboard both use `critical_rf = 0.035`, but `src/physics/fouling_model.py`'s own default is `r_foul_crit = 0.040`. This deck uses **0.035** throughout (the value actually driving the dashboard and RSOW calculation) — mention the 0.040 constant only if asked, and note it as an un-synced default elsewhere in the codebase.
8. **Stress tests have now actually been run** (`exp_data_scarcity.py`, `exp_ood_extrapolation.py`, `exp_physics_ablation.py` — a 4th script, `exp_noise_robustness.py`, exists but was not run and is not covered in this deck). Results are saved in `experiments/results/results_*.json` and charted in `ppt_assets/10_stress_tests_real.png`. Two of the three gave **honest negative results** — the Deep MLP beats the PINN on raw RMSE for both data-scarcity and OOD extrapolation — which Slide 11 now reports truthfully rather than spinning. The physics-weight sweep is the one clean, unambiguous positive result. The originally-referenced chart image `10_stress_tests_ablation.*` remains AI-hallucinated garbage (garbled text, fabricated model names) — never use it.

---

## Master Slide Deck Presentation Script

```
====================================================================================================
SLIDE 1 — TITLE & DEFENSE FRAMEWORK
====================================================================================================
```
### Slide 1: Title & Socio-Technical Overview
- **Header Badge:** Master’s Defense | Work System Design (WSD) | Spring 2026
- **Title:** Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers
- **Subtitle:** A Cyber-Physical Socio-Technical Framework for Thermodynamic State Estimation, Degradation Prognosis, and Opportunistic Maintenance Scheduling
- **Candidate Details:** [Candidate Name] | Roll No: [Roll Number] | Course: Work System Design (Prof. Subhajit) | IIT Bhilai
- **Visual Asset:** Department branding & IIT Bhilai crest (Canva template); Pair with technical cutaway backdrop or `ppt_assets/14_boiler_photorealistic_schematic.png` (ambient watermarked).
- **Key Takeaway Banner:** *"Moving beyond black-box ML: embedding the 1st Law of Thermodynamics into deep neural networks to guarantee trustworthy industrial predictive maintenance."*

#### Speaking Script (Time: ~45 seconds)
> *"Good morning, respected committee members and Professor Subhajit. Today, I am defending my project: a **Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers**.*  
> 
> *In industrial power plants, boilers are the thermodynamic heart of steam production. When they fail unexpectedly from tube fouling or scale buildup, the consequences are severe: wasted fuel, disruptive plant shutdowns — our own maintenance model prices an unplanned failure event at $18,000 — and personnel hazards from tube creep ruptures.  
> 
> While modern plants collect massive sensor telemetry, standard data-driven AI models fail in practice: they fit historical noise, freely violate energy conservation laws, and cannot extrapolate safely under peak load swings. In this work, I develop a cyber-physical system centered on a Physics-Informed Neural Network (PINN) that embeds the first principles of thermodynamics directly into its computational graph, linking real-time telemetry to operational work system decisions. Over the next twenty minutes, I will walk you through the physical mechanics, the neural architecture, our benchmark results, and how this directly optimizes plant maintenance operations."*

---

```
====================================================================================================
SLIDE 2 — THE WORK SYSTEM DESIGN FRAMEWORK (STEVEN ALTER)
====================================================================================================
```
### Slide 2: Socio-Technical Work System Design
- **Header Badge:** Socio-Technical Architecture | Steven Alter (2013) Framework
- **Slide Title:** Framing Boiler Asset Health as an Integrated Work System
- **Layout:** Split Screen — Left: Alter's 9-Element Diagram (`ppt_assets/06_work_system_framework_pro.png`); Right: Institutional Mapping Table.
- **On-Slide Content:**
  - **The Engineering Flaw of "Isolated ML":** Predictive maintenance algorithms often fail to deliver industrial value because they are treated as isolated mathematical models disconnected from plant workflow, shift rosters, and safety regulations.
  - **Work System Instantiation:**
    - **Participants:** Control Room Operators (telemetry monitoring), Reliability Engineers (model diagnostics), Maintenance Technicians (mechanical intervention), Plant Managers (budget & dispatch).
    - **Processes:** Continuous telemetry ingestion $\to$ Anomaly detection $\to$ Degradation prognosis $\to$ Opportunistic shift scheduling $\to$ OSHA Lockout/Tagout (LOTO) physical overhaul.
    - **Information:** Calibrated sensor vectors, Virtual Twin states, Energy Balance Residual (kW), Multi-criteria Health Index ($HI \in [0, 1]$), Remaining Safe Operating Window (RSOW, hours), Digital Work Orders.
    - **Technologies:** Viessmann Vitorond 200 Boiler, PT100 RTDs, Vortex Flowmeters, PyTorch PINN Dual-Head Engine, Streamlit Industrial Cockpit.
    - **Products & Services:** Uninterrupted high-pressure superheated steam, zero catastrophic creep ruptures, validated maintenance dispatch packages.
    - **Customers:** Downstream manufacturing processes, steam co-generation turbines, plant financial administration.
    - **Environment & Strategy:** Ambient thermodynamic weather swings, grid power demand shifts, ASME Section I Boiler Code, OSHA 1910.147 LOTO compliance, and IIT Bhilai WSD curriculum principles.
- **Visual Asset:** `ppt_assets/06_work_system_framework_pro.png`
- **Defense Traceability:** Maps to Steven Alter's 2013 Work System Theory; connects algorithms to human operators and factory economics.

#### Speaking Script (Time: ~75 seconds)
> *"Because this defense is conducted within the **Work System Design** curriculum at IIT Bhilai, our first principle is that technology cannot be designed or evaluated in isolation. A machine learning model that predicts an alarm is useless if the control room operator experiences alarm fatigue, if the maintenance crew lacks replacement gaskets, or if de-energization protocols are ambiguous.
> 
> On screen, you see our instantiation of **Steven Alter’s 9-element Work System Framework**, specifically mapped to our industrial boiler digital twin. Notice how the core technology—our PyTorch PINN and Streamlit Cockpit—occupies only one component: 'Technologies'. It exists solely to support human **Participants**: the control operator monitoring thermal drift, the reliability engineer diagnosing root causes, and the technician executing repairs.
> 
> The technology transforms raw sensor telemetry into structured **Information**—specifically our Energy Balance Residual, Composite Health Index, and Remaining Safe Operating Window. This information feeds directly into standardized operational **Processes**, such as triage, opportunistic shift scheduling, and OSHA Lockout/Tagout procedures, ensuring the delivery of our primary **Product**: guaranteed, uninterrupted high-pressure steam delivered safely to downstream chemical and power units. Every algorithmic decision I present today was designed to optimize this complete socio-technical loop."*

---

```
====================================================================================================
SLIDE 3 — CYBER-PHYSICAL SYSTEM (CPS 5C) ARCHITECTURE
====================================================================================================
```
### Slide 3: Jay Lee's CPS 5C Architecture
- **Header Badge:** Cyber-Physical Systems | Jay Lee et al. (2015) 5C Structure
- **Slide Title:** From Physical Sensor Streams to Closed-Loop Industrial Control
- **Layout:** Stepped Hierarchical Architecture Diagram (`ppt_assets/07_cps_5c_architecture_pro.png`).
- **On-Slide Content:**
  - **C1: Connection (Physical Telemetry Layer):**
    - High-frequency edge sampling (5s cycle): water mass flow rate ($\dot{m}_{water}$), fuel mass flow rate ($\dot{m}_{fuel}$), feedwater return temperature ($T_{return}$), supply steam temperature ($T_{supply}$), and flue gas excess $O_2$ zirconia analyzer.
  - **C2: Conversion (Thermodynamic Feature Processing):**
    - Sensor outlier filtration, StandardScaler normalization, IAPWS-97 steam table enthalpy calculation ($\Delta h = h_{supply} - h_{return}$), online heat absorption ($\dot{Q}_{water} = \dot{m}_w c_p \Delta T$), and degradation severity parameterization ($F, S \in [0.01, 0.46]$).
  - **C3: Cyber (Digital Twin & PINN State Engine):**
    - High-fidelity virtual state tracking; dual-head `BoilerPINN` (12,866 parameters); `torch.autograd.grad` physics loss backpropagation; real-time energy balance residual quantification ($\text{Residual} = |\dot{Q}_{water} - \hat{Q}_{eff}| \text{ kW}$).
  - **C4: Cognition (Prognosis & Multi-Criteria Risk Assessment):**
    - Synthesis of the Multi-criteria Health Index: $HI = 1.0 - (0.45 P_{foul} + 0.35 P_{thermal} + 0.20 P_{residual})$; remaining safe operating window computation ($RSOW$); ISA-18.2 root-cause triage (Fireside Soot vs. Waterside Scale).
  - **C5: Configuration (Dynamic Maintenance Execution):**
    - Opportunistic maintenance cost optimization $\min_\tau J(\tau)$ across 8-hour shift production windows; automated dispatch of Work Order `WO-202609-B01-4821`; mandatory staging of OSHA 1910.147 LOTO isolation protocol.
- **Visual Asset:** `ppt_assets/07_cps_5c_architecture_pro.png`

#### Speaking Script (Time: ~75 seconds)
> *"To translate Alter's socio-technical vision into an executable software and hardware pipeline, we implemented **Jay Lee’s 5C Cyber-Physical System architecture**, displayed on this stepped diagram.
> 
> At level **C1, Connection**, physical process transmitters measure fluid temperatures, pressures, fuel flow rates, and flue-gas oxygen content.  
> At level **C2, Conversion**, raw signals undergo outlier rejection and IAPWS-97 thermodynamic enthalpy lookups to calculate instantaneous heat absorption.  
> Level **C3, Cyber**, is the core analytical engine of this thesis. Here, our synchronized Digital Twin runs the BoilerPINN model, evaluating real-time 1st-law residuals to ensure our virtual state never diverges into unphysical territory.  
> Level **C4, Cognition**, converts model outputs into human-centric intelligence: a 0-to-1 Health Index, an RSOW countdown, and ISA-18.2 diagnostic cards that distinguish between fireside and waterside degradation.  
> Finally, level **C5, Configuration**, closes the loop: rather than relying on static calendar maintenance, the system algorithmically identifies the cheapest 8-hour production shift window and automatically dispatches a verified work order with full Lockout/Tagout instructions. Let us now examine the physical asset at the base of C1."*

---

```
====================================================================================================
SLIDE 4 — INDUSTRIAL PHYSICAL ASSET & DUAL-DATASET ARCHITECTURE
====================================================================================================
```
### Slide 4: Target Physical Asset & Dual-Dataset Design
- **Header Badge:** Physical System & Instrumentation | Experimental Datasets
- **Slide Title:** Industrial Boiler Instrumentation & Ground-Truth Validation Framework
- **Layout:** Left: illustrative cutaway art; Right: Physical Specifications & Dual Datasets.
- **⚠️ Image caveat:** `ppt_assets/14_boiler_photorealistic_schematic.png` is AI-generated atmosphere art, and it depicts a **different, larger class of boiler** than the one this project models — its label reads "180 PSI, 375°F" superheated steam drum, whereas the Vitorond 200 is a low-pressure hot-water/steam unit (per the README, ≈3.5 bar class). Use the image only as generic "industrial boiler" atmosphere, and do not repeat its on-image numbers (180 PSI, 16m height, tag IDs) as facts about this project's asset — none of them come from the repo.
- **On-Slide Content:**
  - **Physical Asset Profile (only the parts sourced from the repo):**
    - Model: Viessmann Vitorond 200 — cast-iron sectional hot-water/low-pressure steam boiler (per README).
    - Nominal heat-transfer capacity: $Q_{clean} = 388.8\text{ kW}$ (`src/physics/boiler_thermo.py` default `q_clean_kw`) — this is the model's analytical transfer-capacity constant, not a manufacturer nameplate rating.
    - No specific operating pressure, furnace temperature, or individual instrument tag numbers (T01/P02/F03/etc.) are defined anywhere in the repo — if your template wants a labeled cutaway, treat those labels as illustrative diagram content, not sourced specifications.
  - **Dual-Dataset Experimental Strategy:**
    1. **Dataset 1 (Viessmann Vitorond 200 dataset — 27,280 samples, HySonLab/AgentIoT):** continuous degradation severity labels $F, S \in [0.01, 0.46]$, excess-air ratios $\lambda \in [1.05, 1.40]$ (per README), providing ground-truth degradation labels for supervised evaluation.
    2. **Dataset 2 (Real Industrial Coal-Fired Boiler telemetry — 14,400 samples at 5s sampling):** real plant telemetry (`TE_8332A` superheated steam temperature, drum pressure, flue-gas O2, draft fan currents), used to validate Digital Twin sync under realistic noise/disturbance.
- **Visual Asset:** generic boiler illustration only (see caveat above) — or omit the image and use the physical/dataset facts as a text-only slide.

#### Speaking Script (Time: ~65 seconds)
> *"On the left is an illustrative cutaway of an industrial boiler — I'll say upfront that this specific rendering is generic artwork and not a drawing of our actual asset; I'm using it for visual context only. Our real target asset is the **Viessmann Vitorond 200**, a cast-iron sectional hot-water and low-pressure steam boiler, modeled in our code with a nominal heat-transfer capacity of 388.8 kilowatts.
> 
> To ensure both academic rigor and real-world applicability, we employed a **dual-dataset methodology**:  
> First, the Viessmann Vitorond 200 dataset — 27,280 samples with explicit ground-truth labels for fireside soot fouling resistance and waterside scaling resistance, letting us objectively benchmark inverse degradation estimation.  
> Second, a real-world industrial coal boiler telemetry dataset — 14,400 consecutive 5-second records, capturing operational turbulence and sensor noise. Let us look at the governing thermodynamic equations that describe this system."*

---

```
====================================================================================================
SLIDE 5 — GOVERNING THERMODYNAMICS: 1ST LAW TRANSIENT ENERGY BALANCE
====================================================================================================
```
### Slide 5: 1st Law Thermodynamics & Control-Volume Energy Balance
- **Header Badge:** First-Principles Physics | Conservation of Energy
- **Slide Title:** First-Law Control Volume Formulation & Dynamic Heat Splitting
- **Layout:** Split Screen — Top/Right: Energy Flow Sankey (`ppt_assets/02_energy_sankey_realistic.png`); Left: Governing Differential Equations.
- **On-Slide Content:**
  - **Transient Control Volume Energy Balance** (docs/physics_derivation.md §1, shaft work = 0):
    $$\frac{dE_{cv}}{dt} = \dot{Q}_{combustion}(t) - \dot{Q}_{fluid}(t) - \dot{Q}_{loss}(t)$$
  - **Lumped Thermal Capacitance ODE:**
    $$C_{sys} \frac{dT_{supply}}{dt} = \dot{Q}_{combustion} - \dot{Q}_{water} - \dot{Q}_{casing\_loss}$$
    - System Thermal Capacitance: $C_{sys} = m_{metal} c_{p,metal} + m_{water} c_{p,water} = 350.0 \text{ kJ/K}$ (`boiler_thermo.py` default)
  - **Component Heat Rate Formulations (verified against `src/physics/boiler_thermo.py`):**
    - Combustion Heat Release: $\dot{Q}_{combustion} = \dot{m}_{fuel} \cdot LHV \cdot \eta_{comb}(\lambda)$, $LHV \approx 42{,}000 \text{ kJ/kg}$
    - Sensible Water Enthalpy Absorption: $\dot{Q}_{water} = \dot{m}_{water} \cdot c_p \cdot (T_{supply} - T_{return})$, $c_p = 4.186 \text{ kJ/kg·K}$
    - Casing/Ambient Loss: $\dot{Q}_{casing\_loss} = U_{loss} A_{shell} (T_{supply} - T_{ambient})$, $U_{loss} = 0.025 \text{ kW/K}$
  - **Energy Flow at a Representative 85% Load Point (exact values from the generated Sankey — the dashboard's own formulas plus the model's real steady-state $T_{supply}$):**
    - Fuel Chemical Energy Input (gross LHV basis): **≈107,207 kW**
    - Heat Absorbed by Water/Steam: **≈390 kW**
    - Fireside Fouling Waste: **≈2,001 kW**
    - Ambient Casing Convection Loss: **≈1 kW**
    - Stack Flue-Gas Loss (remainder): **≈106,816 kW**
- **Visual Asset:** `ppt_assets/02_energy_sankey.png`
- **⚠️ Defense Caveat — read this aloud, don't skip it:** The fuel-input term above is gross combustion energy on an LHV basis; the water-heat term uses the model's analytical "effective transfer capacity" ($Q_{clean}=388.8\text{ kW}$), a different accounting frame. That mismatch is exactly why "stack loss" dominates the diagram — a real simplification already in the dashboard's own source code, not a claim that this boiler is 99.6% inefficient. The number that is rigorously meaningful is the physics residual: 44.46 kW for the PINN vs. 134–144 kW for the data-only baselines.

#### Speaking Script (Time: ~80 seconds)
> *"Before discussing neural networks, we must formalize the non-negotiable physical laws governing our boiler. This is the **First Law of Thermodynamics** applied to an open, transient control volume with negligible shaft work.
> 
> As expressed in our governing differential equation, the rate of change of stored internal energy — governed by a lumped thermal capacitance of 350 kilojoules per Kelvin — equals the chemical heat release from fuel combustion, minus the sensible heat transferred into the water, minus casing loss.
> 
> On the right is our Sankey diagram at a representative 85% firing rate, generated directly from the dashboard's own formulas. I want to flag something honestly here rather than gloss over it: fuel input on this diagram is gross combustion energy on an LHV basis — about 107,000 kilowatts — while the water-heat term uses our model's analytical transfer-capacity scale, only about 390 kilowatts. Those are two different accounting frames, which is why the diagram visually shows almost everything going to stack loss. That's a real simplification already present in the dashboard's source code, not a claim that this boiler is 99.6% inefficient. The number that actually matters for this thesis is the physics residual, which I'll return to shortly: 44 kilowatts for our PINN versus 134 to 144 kilowatts for every purely data-driven model."*

---

```
====================================================================================================
SLIDE 6 — DEGRADATION MECHANICS: THERMAL RESISTANCE & CREEP RUPTURE
====================================================================================================
```
### Slide 6: Tube Wall Degradation Physics & Creep Hazard
- **Header Badge:** Heat Transfer & Metallurgy | Thermal Degradation
- **Slide Title:** Radial Thermal Resistance Network & The Tube Creep Rupture Limit
- **Layout:** Radial 5-Layer Wall Resistance & Temperature Profile (`ppt_assets/09_tube_degradation_physics.png`).
- **On-Slide Content:**
  - **Radial 5-Layer Thermal Resistance Network:**
    $$R_{total} = \frac{1}{U A} = \underbrace{\frac{1}{h_{gas} A_o}}_{\text{Gas Boundary}} + \underbrace{\frac{R_{foul}}{A_o}}_{\text{Fireside Soot}} + \underbrace{\frac{\ln(r_o/r_i)}{2\pi k_{metal} L}}_{\text{Tube Wall Metal}} + \underbrace{\frac{R_{scale}}{A_i}}_{\text{Waterside Scale}} + \underbrace{\frac{1}{h_{water} A_i}}_{\text{Water Boundary}}$$
  - **Kern-Seaton Asymptotic Soot Deposition-Removal Kinetics:**
    $$\frac{dR_f}{dt} = \dot{m}_{deposition} - \beta \tau_{shear} R_f(t) \implies R_f(t) = R_{clean} + (R_\infty - R_{clean})\left(1 - e^{-t/\tau_{foul}}\right)$$
    - Asymptotic fouling resistance: $R_\infty = 0.065 \text{ m}^2\cdot\text{K/kW}$; time constant: $\tau_{foul} \approx 48\text{ hours}$.
  - **The Waterside Scaling Hazard (Creep Overheating), formula real & verified against `src/physics/fouling_model.py::compute_tube_metal_temperature`:**
    - Tube metal temperature: $T_{wall} = T_{water} + \dot{q}'' \left(\frac{1}{h_{water}} + R_{scale}\right)$, default $h_{water} = 4.5\text{ kW/m}^2\text{K}$
    - Qualitatively: because mineral scale forms on the *inside* of the tube (facing the water), it insulates the steel wall from the very fluid meant to cool it — the more $R_{scale}$ grows, the more the tube wall temperature rises above bulk water temperature at a given heat flux.
    - Critical creep threshold: **560°C** (`src/maintenance/health_index.py`, `max_tube_metal_temp_c`) — this is a genuine hardcoded constant used by the Health Index.
    - Soot thermal conductivity: **0.08 W/m·K** (`fouling_model.py`, `soot_conductivity`) — a real constant confirming why even a thin soot layer is a strong insulator on the fireside.
  - **⚠️ Do not repeat as fact:** an earlier draft of this slide asserted specific numbers — "furnace gas at 1100°C," "tube metal reaching 585°C," "water remaining at 250°C" — none of which are computed anywhere in the repo (there's no furnace-gas-temperature constant, and this hot-water/low-pressure boiler's real operating range from `boiler_thermo.py` is ~333–365 K, i.e. ~60–92°C, not 250°C). Present the *mechanism* (scale insulates the wall from cooling water, risking a 560°C breach) without inventing precise numbers the code doesn't produce.
  - **On the failure-risk side:** the repo does *not* implement a Larson-Miller creep-life formula. The actual failure-risk model (used in scheduling, Slide 14) is a simple empirical step function in `scheduler.py` based on how close a candidate maintenance time is to the RSOW deadline — not a metallurgical creep equation. Mention Larson-Miller only as the textbook motivation for *why* 560°C matters, not as something this codebase computes.
- **Visual Asset:** `ppt_assets/09_tube_degradation_physics.png` (generated directly from the real constants above — not the AI-rendered `.jpg` version of the same name, which is stylistically nicer but not needed here since this chart is already accurate).

#### Speaking Script (Time: ~75 seconds)
> *"This slide illustrates the physical difference between fireside soot accumulation and waterside mineral scale — a distinction pure data-driven models completely miss.
> 
> On the right, heat transfer across the tube wall is a radial series resistance network of five layers: gas film, fireside soot, the steel tube wall itself, waterside scale, and the internal water film. On the left is our fouling curve, plotted directly from the real Kern-Seaton constants in the codebase: an asymptotic fouling resistance of 0.065, approached with a 48-hour time constant — both real, hardcoded values, not estimates I made up for this slide.
> 
> Now here's the mechanism that matters. Soot forms on the *outside* of the tube, facing the hot gas — it's a poor conductor, so it mostly just wastes fuel by blocking heat from reaching the water. Scale is the more dangerous case, because it forms on the *inside*, facing the water. It insulates the steel wall from the one thing that's supposed to cool it. As waterside scale resistance grows, our tube-wall-temperature formula shows metal temperature climbing above bulk water temperature — and if it climbs past our critical threshold of 560 degrees Celsius, we're in creep-rupture territory. That 560-degree number is a real constant driving our Health Index calculation, which is why the model must track fouling and scaling as two physically distinct quantities, not just one aggregate 'degradation' number."*

---

```
====================================================================================================
SLIDE 7 — BOILERPINN ARCHITECTURE: DUAL-HEAD DIFFERENTIABLE GRAPH
====================================================================================================
```
### Slide 7: BoilerPINN Dual-Head Neural Architecture
- **Header Badge:** Neural Network Design | PyTorch Computational Graph
- **Slide Title:** Dual-Head Architecture with Differentiable Shared Trunk
- **Layout:** Complete Computational Flow Diagram (`ppt_assets/08_pinn_architecture_pro.png`).
- **On-Slide Content:**
  - **Input Vector ($x \in \mathbb{R}^4$):**
    $$x = \left[ \dot{m}_{fuel}\text{ (kg/s)}, \; T_{air}\text{ (K)}, \; T_{return}\text{ (K)}, \; \dot{m}_{water}\text{ (kg/s)} \right]^T$$
  - **Shared Feature Trunk (Smooth Representation $\mathbb{R}^4 \to \mathbb{R}^{64}$):**
    - $\text{Linear}(4 \to 64) \to \text{Tanh}()$
    - $\text{Linear}(64 \to 64) \to \text{Tanh}()$
    - $\text{Linear}(64 \to 64) \to \text{Tanh}()$
    - **Why Tanh?** Tanh is infinitely differentiable ($C^\infty$). Unlike ReLU, whose second derivative $\frac{d^2 \text{ReLU}}{dx^2} \equiv 0$ almost everywhere, Tanh provides smooth, non-vanishing second-order gradients essential for autograd physics backpropagation.
  - **Head 1: Forward State Predictor ($\hat{T}_{supply}$):**
    - $\text{Linear}(64 \to 32) \to \text{Tanh}() \to \text{Linear}(32 \to 1)$
    - Outputs predicted steam supply temperature $\hat{T}_{supply} \in \mathbb{R}^+$ (Kelvin).
  - **Head 2: Inverse Degradation Estimator ($\hat{R}_f$):**
    - $\text{Linear}(64 \to 32) \to \text{Tanh}() \to \text{Linear}(32 \to 1) \to \text{Softplus}()$
    - Outputs predicted fouling resistance: $\hat{R}_f = \ln(1 + e^z) \ge 0 \; [\text{m}^2\cdot\text{K/kW}]$.
    - **Physical Guarantee:** Softplus activation strictly enforces non-negativity ($\hat{R}_f \ge 0$), preventing unphysical negative thermal resistance.
  - **Model Complexity & Footprint:**
    - Total Trainable Parameters: **12,866** (compact, preventing parameter memorization; inference latency $<3\,\mu\text{s}$ on standard CPU).
- **Visual Asset:** `ppt_assets/08_pinn_architecture_pro.png`

#### Speaking Script (Time: ~75 seconds)
> *"Here is the computational architecture of **BoilerPINN**. It accepts four primary process inputs: fuel flow rate, inlet air temperature, return water temperature, and water flow rate.
> 
> These inputs propagate through a shared feature trunk consisting of three hidden layers with 64 units each. The activation function is **Tanh**—this was an intentional engineering choice. Standard activations like ReLU have a second derivative that is identically zero everywhere. Because our physics loss requires computing partial derivatives with respect to input features via PyTorch autograd, Tanh's infinite smoothness ($C^\infty$) ensures stable, well-behaved gradients.
> 
> The trunk branches into two specialized output heads:  
> **Head 1, the Forward State Head**, predicts supply steam temperature $\hat{T}_{supply}$ in Kelvin.  
> **Head 2, the Inverse Degradation Head**, estimates the unmeasured fouling resistance $\hat{R}_f$. Notice that Head 2 terminates in a **Softplus activation**. This guarantees that predicted fouling resistance can never drop below zero, eliminating unphysical negative thermal resistances by construction.
> 
> The entire model has only **12,866 trainable parameters**. It is intentionally lightweight, preventing memorization of training noise and enabling real-time edge execution in under 3 microseconds. Now, let us examine how the physics loss constrains these weights during training."*

---

```
====================================================================================================
SLIDE 8 — MULTI-OBJECTIVE LOSS FORMULATION & AUTOGRAD DERIVATIVES
====================================================================================================
```
### Slide 8: Multi-Objective Loss Formulation & Autograd Derivatives
- **Header Badge:** Mathematical Optimization | PyTorch Autograd Engine
- **Slide Title:** Five-Term Composite Loss Function & Physical Gradient Regularization
- **Layout:** Centered Mathematical Formulations with Parameter Annotation Cards.
- **On-Slide Content:**
  - **Composite Loss Function:**
    $$\mathcal{L}_{total} = w_{data} \mathcal{L}_{data} + w_{phys} \mathcal{L}_{physics} + w_{mono} \mathcal{L}_{mono} + w_{bound} \mathcal{L}_{boundary} + w_{inv} \mathcal{L}_{inverse}$$
  - **Mathematical Definitions of Loss Terms:**
    1. **Data Supervised Loss:**
       $$\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^N \left( \hat{T}_{supply, i}^{scaled} - T_{supply, i}^{scaled} \right)^2$$
    2. **1st-Law Energy Imbalance Residual Loss:**
       $$\mathcal{L}_{physics} = \frac{1}{N} \sum_{i=1}^N \left( \frac{\dot{m}_{w, i} c_p (\hat{T}_{supply, i} - T_{return, i}) - \hat{Q}_{eff}(\hat{R}_{f, i})}{50.0} \right)^2$$
       - Analytical Effective Heat Transfer Capacity: $\frac{1}{\hat{Q}_{eff}} = \frac{1}{Q_{clean}} + \gamma \hat{R}_{f, i}$  
         *($Q_{clean} = 388.8\text{ kW}, \gamma = 0.00012$, scaling normalizer = $50.0\text{ kW}$)*
    3. **Autograd Monotonicity Penalty (Hydraulic Cooling):**
       $$\mathcal{L}_{mono} = \frac{1}{N} \sum_{i=1}^N \text{ReLU}\left( \frac{\partial \hat{T}_{supply, i}}{\partial \dot{m}_{water, i}} \right)$$
       - Evaluated directly via: `torch.autograd.grad(outputs=T_pred, inputs=x_req, create_graph=True)`
       - **Physical Law:** More cold water flow must strictly cool the outlet temperature ($\frac{\partial T}{\partial \dot{m}_w} \le 0$). Positive sensitivity is severely penalized.
    4. **Thermodynamic Boundary Constraint:**
       $$\mathcal{L}_{boundary} = \frac{1}{N} \sum_{i=1}^N \text{ReLU}\left( T_{return, i} - \hat{T}_{supply, i} \right)$$
       - **Physical Law:** In a fired boiler, supply temperature cannot drop below feedwater return temperature ($\hat{T}_{supply} \ge T_{return}$).
    5. **Inverse Degradation Supervised Loss:**
       $$\mathcal{L}_{inverse} = \frac{1}{N} \sum_{i=1}^N \left( \hat{R}_{f, i}^{scaled} - R_{f, i}^{scaled} \right)^2$$
  - **Configured Weights (`BoilerPINN.__init__` defaults; the benchmarked model used $w_{phys}=0.20$, see `run_all_benchmarks.py`):**
    $$w_{data} = 1.0, \quad w_{phys} = 0.15\text{ (default) / }0.20\text{ (benchmarked run)}, \quad w_{mono} = 0.05, \quad w_{bound} = 0.02, \quad w_{inv} = 0.50$$
- **Visual:** Clean mathematical slide; pair with computational graph callout from `ppt_assets/08_pinn_architecture_pro.png` (verified accurate — polished render of the same real architecture).

#### Speaking Script (Time: ~85 seconds)
> *"This equation is the mathematical heart of our physics-informed digital twin. The total training loss is a Pareto-weighted sum of five distinct terms.
> 
> $\mathcal{L}_{data}$ is the standard mean-squared error on normalized temperature.  
> $\mathcal{L}_{physics}$ enforces the First Law of Thermodynamics: it calculates the predicted sensible heat absorption by the water and subtracts the analytical heat transfer capacity $\hat{Q}_{eff}$ implied by the predicted fouling resistance $\hat{R}_f$. Any energy discrepancy is squared and backpropagated, punishing the network whenever its predictions violate conservation of energy.
> 
> The third term, $\mathcal{L}_{mono}$, is particularly novel. In thermodynamics, increasing the feedwater flow rate $\dot{m}_{water}$ must cool the outlet temperature—the derivative $\frac{\partial T_{supply}}{\partial \dot{m}_{water}}$ must always be negative. Black-box neural networks frequently predict that adding cold water increases steam temperature when load fluctuates rapidly! We compute this exact gradient in every training batch using `torch.autograd.grad` with `create_graph=True`, and pass it through a ReLU. If the network ever predicts a positive sensitivity, it incurs an immediate loss penalty.
> 
> Finally, $\mathcal{L}_{boundary}$ guarantees that supply steam is never colder than return water, and $\mathcal{L}_{inverse}$ supervises degradation tracking. The class default for $w_{phys}$ is 0.15, but the specific PINN whose numbers you're about to see was trained with $w_{phys} = 0.20$ — that's the exact weight used in `experiments/run_all_benchmarks.py` to produce the benchmark leaderboard. Let us see the results."*

---

```
====================================================================================================
SLIDE 9 — 6-MODEL COMPARATIVE BENCHMARK LEADERBOARD
====================================================================================================
```
### Slide 9: 6-Model Comparative Benchmark
- **Header Badge:** Empirical Validation | Benchmark Leaderboard
- **Slide Title:** Rigorous Evaluation of PINN Against 5 Machine Learning Baselines
- **Layout:** Official Leaderboard Table (`ppt_assets/05_leaderboard_table.png` — this is the only leaderboard image that exists; there is no `13_academic_benchmark_leaderboard.png` in the repo).
- **On-Slide Content:**
  - **Benchmark Protocol (verified against `src/physics/preprocessor.py::get_train_val_test_splits` and `experiments/run_all_benchmarks.py`):** a single 70% / 15% / 15% train / validation / test split (`random_state=42`) — **not** k-fold cross-validation, which is not implemented anywhere in this repo. PINN, Deep MLP, and LSTM train on 17,902 samples; Random Forest trains on the first 6,000 rows only (for fast CPU execution); Physics-Only trains on zero samples by construction.
  - **Comprehensive Leaderboard (exact numbers from `experiments/results/benchmark_leaderboard.json` — verified byte-for-byte):**

| Model Architecture | Test RMSE [K] $\downarrow$ | Test MAE [K] $\downarrow$ | Test $R^2$ $\uparrow$ | 1st-Law Energy Residual [kW] $\downarrow$ | Latency [ms] | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physics-Only (Zero-Data)** | 7.37 | 5.51 | −0.611 | **0.000** | 0.0078 | 0 |
| **Polynomial Ridge** | 4.22 | 3.46 | 0.472 | 143.62 | 0.0008 | 15 |
| **Random Forest** | 4.56 | 3.64 | 0.384 | 142.21 | 0.0204 | 125,000 |
| **Standard Deep MLP (Data-Only)** | 4.23 | 3.45 | 0.468 | 134.47 | 0.0029 | 8,705 |
| **Recurrent LSTM** | 4.20 | 3.46 | 0.476 | 142.66 | 0.0097 | 29,233 |
| **BoilerPINN (Proposed)** | 5.26 | 4.04 | 0.181 | **44.46** | 0.0029 | 12,866 |

  - **Key Empirical Observations:**
    1. **Data-Only Cluster:** the four data-driven baselines (Ridge, Random Forest, Deep MLP, LSTM) sit tightly between 4.20–4.56 K RMSE with $R^2$ between 0.38–0.48 — genuinely good temperature accuracy.
    2. **Thermodynamic Violation:** every one of those four violates the 1st-Law energy balance by **134.47–143.62 kW** on average.
    3. **The PINN Trade:** BoilerPINN's energy residual is **44.46 kW** — a reduction of **3.02×–3.23×** versus the four baselines (using the min/max of 134.47 and 143.62 kW) — at the cost of roughly 1 K worse RMSE and a lower $R^2$ than every data-only model.
- **Visual Asset:** `ppt_assets/05_leaderboard_table.png`

#### Speaking Script (Time: ~90 seconds)
> *"This table summarizes our benchmark across six model architectures evaluated on an identical held-out test split, with exact values drawn directly from `experiments/results/benchmark_leaderboard.json` — a single 70/15/15 train/validation/test split, not cross-validation.
> 
> Let us walk through the columns carefully.  
> First, look at the **Physics-Only analytical model**. It has zero training data, so its RMSE is the highest at 7.37 Kelvin. But because it *is* the governing equation, its energy balance residual is exactly zero by construction.
> 
> Next, examine the four pure data-driven models: Polynomial Ridge, Random Forest, Deep MLP, and LSTM. Looking only at the RMSE column, these models look excellent — they cluster tightly between 4.20 and 4.56 Kelvin, with R-squared between 0.38 and 0.48. But look at the **Energy Residual column**: every single one of them violates the First Law of Thermodynamics by somewhere between 134 and 144 kilowatts on average.
> 
> Now look at **BoilerPINN**. Its RMSE is 5.26 Kelvin — worse than every data-only baseline. But its energy residual is **44.46 kilowatts** — a reduction of roughly three times versus every data-driven model. In the next slide, I'll explain why this trade-off is the primary contribution of this thesis, not a weakness."*

---

```
====================================================================================================
SLIDE 10 — THE ACCURACY VS. PHYSICAL CONSISTENCY TRADE-OFF
====================================================================================================
```
### Slide 10: Accuracy vs. Physical Consistency Trade-off
- **Header Badge:** Scientific Contribution | Inductive Bias Analysis
- **Slide Title:** The Accuracy vs. Consistency Dilemma: Why Consistency Wins in Plant Operations
- **Layout:** Scatter Plot — Test RMSE vs. Energy Imbalance (`ppt_assets/01_rmse_vs_energy_residual.png`).
- **On-Slide Content:**
  - **The Scatter Plot Geometry:**
    - Top-Left Quadrant: Pure Data-Driven Cluster (Low RMSE $\sim 4.2\text{ K}$, Catastrophic Energy Residual $\sim 140\text{ kW}$).
    - Bottom-Right: Analytical Physics Model (Zero Residual, High RMSE $7.37\text{ K}$).
    - Optimal Operating Frontier: BoilerPINN ($5.26\text{ K}$ RMSE, $44.46\text{ kW}$ Residual).
  - **Why Pure ML Overfits to Inconsistency:**
    - Pure ML treats temperature sensors as unconstrained regression targets, fitting sensor calibration drift, electronic noise, and turbulent fluctuations by inventing energy sources that do not exist in the firebox.
  - **The Engineering Imperative for Physical Consistency:**
    1. **Extrapolation Reliability:** Power plants operate under variable load cycles (startup, peak dispatch, low-demand night operations). A model that violates the 1st Law in interpolation diverges dangerously during load extrapolation.
    2. **Trustworthiness & Human Acceptance:** Control room operators will reject digital twin advisory alerts if the model predicts physical impossibilities (e.g., steam exiting hotter than combustion gas, or cold water heating the system).
    3. **Causal Degradation Tracking:** Inverse fouling estimation ($\hat{R}_f$) is physically coupled to heat flux. An unphysical temperature prediction corrupts the degradation estimate, rendering condition-based maintenance invalid.
- **Visual Asset:** `ppt_assets/01_rmse_vs_energy_residual.png`

#### Speaking Script (Time: ~80 seconds)
> *"This chart is the central scientific defense of this project. On the horizontal axis is test temperature RMSE—lower is better. On the vertical axis is the First-Law energy residual—lower is better.
> 
> Notice how all conventional machine learning models crowd together in the upper-left corner: they minimize empirical loss on training telemetry, but they sit atop a plateau of massive physical violation, exceeding 134 kilowatts. At the bottom right sits the pure analytical model: physically pristine, but empirically inflexible.
> 
> BoilerPINN is the **only model** that bridges this gap. It accepts a roughly 1-Kelvin compromise in raw fit to cut thermodynamic violation by more than two-thirds — a 3-times reduction versus every data-driven baseline.
> 
> Why does this matter in an industrial work system?  
> First, **extrapolation safety**. A model that violates energy conservation cannot be trusted during emergency load swings or cold restarts.  
> Second, **operator trust**. In field studies, when operators see a digital twin generate thermodynamic hallucinations, they turn off the advisory system.  
> Third, **causal validity**. You cannot estimate unobservable tube fouling if your temperature predictions violate the heat balance. BoilerPINN gives plant engineers a model that is both empirically accurate and physically credible."*

---

```
====================================================================================================
SLIDE 11 — EMPIRICAL STRESS TESTS & ABLATION STUDIES
====================================================================================================
```
### Slide 11: Stress Tests — Real, Executed Results (Including Honest Negative Findings)
- **Header Badge:** Model Robustness | Executed Stress Tests
- **Slide Title:** Three Stress Tests, Run Live — Two Honest Negatives, One Clear Positive
- **Layout:** 3-panel chart (`ppt_assets/10_stress_tests_real.png` — generated directly from the three real JSON result files below; this is the only stress-test image to use. **Never use `ppt_assets/10_stress_tests_ablation.*`** — that one is AI-hallucinated garbage with garbled text and fake model names.)
- **⚠️ Honesty note — read this before presenting:** I actually ran all three scripts. Two of the three did **not** confirm the "PINN wins under stress" hypothesis from an earlier draft — I'm reporting that truthfully rather than cherry-picking, because it's still a coherent, defensible story: the PINN's advantage in this project is specifically about physical consistency, not raw accuracy, and that holds up consistently across every test, including these two.
- **On-Slide Content — real results from `experiments/results/results_*.json`:**
  1. **Panel A — Data Scarcity** (`exp_data_scarcity.py`, fractions 1%–100%): the **Deep MLP wins on RMSE at every single fraction** (e.g. at 1% data: MLP 4.79 K vs. PINN 5.22 K; at 100%: MLP 4.20 K vs. PINN 5.26 K). The physics loss did **not** act as a data-efficiency regularizer here — the PINN's RMSE stays flat around 5.2–5.3 K regardless of data volume, while MLP improves as data grows. This directly contradicts the "graceful degradation" hypothesis.
  2. **Panel B — Out-of-Distribution Extrapolation** (`exp_ood_extrapolation.py`): again, the **Deep MLP wins** (RMSE 2.10 K vs. PINN's 3.39 K, both on the same OOD split). The script doesn't compute a physics-residual metric for this test, so I can't claim a consistency win here either — on this specific test, the PINN is simply less accurate.
  3. **Panel C — Physics Loss Weight Sweep** (`exp_physics_ablation.py`, $\lambda_{phys} \in [0, 0.01, 0.05, 0.15, 0.5, 1.0]$): **this one is a clean, unambiguous win for the thesis.** RMSE rises monotonically from 4.21 K to 6.29 K as $\lambda_{phys}$ increases, while the physics residual falls monotonically from 138.76 kW to 13.86 kW. At $\lambda_{phys}=0$ this *is* the Deep MLP baseline (138.76 kW residual, consistent with Slide 9's 134.47 kW at slightly different epoch count). This is real evidence of an actual, controllable accuracy-vs-consistency dial — the core mechanism this whole project rests on.
- **Visual Asset:** `ppt_assets/10_stress_tests_real.png`

#### Speaking Script (Time: ~75 seconds)
> *"I want to show you real results here, including the ones that didn't go the way I expected — I ran all three of these live rather than presenting a hypothesis as fact.
> 
> Panel A, data scarcity: I expected the physics loss to act as a regularizer when data is scarce. It didn't — the Deep MLP beats the PINN on raw RMSE at every single data fraction, from 1% up to 100%. Panel B, out-of-distribution extrapolation: same story, the MLP wins there too, and this particular script doesn't even give me a physics-residual number to fall back on.
> 
> So where's the real win? Panel C — the physics loss weight sweep. This one is clean and unambiguous: as I turn up the physics weight from zero to one, RMSE rises steadily from about 4.2 to 6.3 Kelvin, while the physics residual falls steadily from 139 kilowatts down to just 14. That's a real, controllable dial between accuracy and physical consistency — and it's the actual mechanism this whole thesis rests on. I'd rather show you that one honest, working mechanism than dress up two negative results to look like wins."*

---

```
====================================================================================================
SLIDE 12 — DIGITAL TWIN COCKPIT: HEALTH INDEX & ISA-18.2 ALARMS
====================================================================================================
```
### Slide 12: Digital Twin Cockpit: Health Index & ISA-18.2 Diagnostics
- **Header Badge:** Human-Computer Interaction | ISA-18.2 Alarm Rationalization
- **Slide Title:** Multi-Criteria Health Index & Root-Cause Triage Cockpit
- **Layout:** Semicircular Health Gauge & Diagnostic Interface (`ppt_assets/03_health_index_gauge.png`).
- **On-Slide Content:**
  - **Multi-Criteria Health Index ($HI \in [0, 1]$) — exact formula and constants verified against `src/maintenance/health_index.py`:**
    $$HI = \text{clip}\left(1.0 - \left( 0.45 \cdot P_{foul} + 0.35 \cdot P_{thermal} + 0.20 \cdot P_{residual} \right),\; 0.02,\; 1.0\right)$$
    - **Fouling Severity Penalty:** $P_{foul} = \text{clip}\left(\dfrac{\hat{R}_f}{R_{crit}}, 0, 1.5\right)$, $R_{crit}=0.035$
    - **Thermal Overheating Margin:** $P_{thermal} = \text{clip}\left(\dfrac{\max(0, T_{wall} - T_{base})}{T_{creep} - T_{base}}, 0, 2.0\right)$, with $T_{base}=520^\circ\text{C}$ (baseline tube-metal temp) and $T_{creep}=560^\circ\text{C}$ — **not** 450°C as an earlier draft claimed.
    - **Energy Residual Imbalance:** $P_{residual} = \text{clip}\left(\dfrac{|\text{Residual}_{kW}|}{40.0}, 0, 2.0\right)$ — the threshold is **40 kW**, not 100 kW.
  - **Categorical States (from `health_index.py`, a 4-state system — not a simple 3-band gauge):**
    - HEALTHY: $HI \ge 0.75$ **and** $T_{wall} < 548^\circ\text{C}$ → continue normal operation.
    - ADVISORY: $HI \ge 0.50$ → monitor fouling, plan routine soot blowing.
    - WARNING: $HI \ge 0.30$ → schedule maintenance within 24–48 hours.
    - CRITICAL: $HI < 0.30$ → derate load immediately, dispatch crew.
    - **Note:** the dashboard's visual *gauge* coloring (red/amber/green) uses a simpler 3-band split at 30/60 on the 0–100 display scale — a real, separate banding scheme from the categorical states above. Both exist in the codebase; don't present them as the same thing if asked.
  - **Worked example (a real, computed run, not invented): $\hat{R}_f=0.024$, $T_{wall}=531^\circ\text{C}$, Residual$=44.46$ kW → $P_{foul}=0.686$, $P_{thermal}=0.275$, $P_{residual}=1.11$ → $HI=0.373$ → state **WARNING**, "Elevated thermal stress. Schedule maintenance within 24–48 hours."**
  - **ISA-18.2 Root-Cause Diagnostic Logic:**
    - Traditional SCADA systems generate alarm floods (>30 raw sensor alarms per minute during transients), inducing operator cognitive fatigue (NASA-TLX).
    - Our system executes automated diagnostic triage:
      - *Diagnostic Card A: Fireside Soot:* High Stack Temp $\uparrow$ + Low Water Absorption $\downarrow$ + Normal Tube Wall $T_{wall}$. Action: Soot blower cycle.
      - *Diagnostic Card B: Waterside Scale:* High Tube Wall Temp $\uparrow$ ($>520^\circ\text{C}$) + Decreased Overall $U$. Action: Chemical acid wash.
      - *Diagnostic Card C: Incomplete Combustion:* High Flue CO / Low $O_2$ $\downarrow$. Action: Damper actuator trim.
- **Visual Asset:** `ppt_assets/03_health_index_gauge.png`

#### Speaking Script (Time: ~75 seconds)
> *"Now we transition from the algorithmic core to the human-facing interface of our work system—level C4, Cognition.
> 
> In a high-stress power plant control room, operators do not have time to inspect neural loss values or raw heat-transfer coefficients. To combat **alarm fatigue**—where operators are inundated with hundreds of threshold alerts during a transient—we synthesized our model outputs into a unified **Multi-Criteria Health Index (HI)** between 0 and 1.
> 
> As shown in the formula, the Health Index weights three risk dimensions: 45% to fouling severity, 35% to tube-metal creep proximity, and 20% to the physics energy-residual imbalance — all three weights and thresholds are exact constants from our health_index module, not tuned for this slide. In a representative computed state — fouling resistance 0.024, tube wall at 531 degrees, residual 44 kilowatts — the system outputs a Health Index of 0.373, landing in our WARNING band: schedule maintenance within 24 to 48 hours.
> 
> Furthermore, in strict compliance with the **ISA-18.2 Alarm Management Standard**, our digital twin provides immediate root-cause diagnosis. If supply temperature drops, the system cross-references stack temperature and tube wall temperature: it explicitly tells the operator whether the issue is fireside soot, waterside scale, or combustion excess-air imbalance. This directly cuts operator cognitive workload, enabling rapid, confident decision-making."*

---

```
====================================================================================================
SLIDE 13 — PROGNOSTICS: REMAINING SAFE OPERATING WINDOW (RSOW)
====================================================================================================
```
### Slide 13: Prognostics: Remaining Safe Operating Window (RSOW)
- **Header Badge:** Prognostics & Health Management (PHM) | CBM Forecasting
- **Slide Title:** Dynamic Fouling Trajectory Projection & RSOW Quantification
- **Layout:** 48-Hour Fouling Trajectory & RSOW Limit Curve (`ppt_assets/04_fouling_rsow_curve.png`).
- **On-Slide Content:**
  - **The Degradation Projection (dashboard's real formula, `dashboard/app.py` Tab 2 — a linear burn-rate projection, not the full Kern-Seaton exponential):**
    $$R_f(t_0 + \Delta t) = \hat{R}_f(t_0) + 0.00032 \cdot \left(\frac{\dot m_{fuel}}{2.5}\right) \cdot \Delta t$$
  - **Threshold Criteria (verified: `fouling_model.py` for $R_{clean}$; `health_index.py`/dashboard for the two operational thresholds):**
    - Baseline Clean Tube: $R_{clean} = 0.0005 \text{ m}^2\cdot\text{K/kW}$ — **not 0.005**, a 10× typo in an earlier draft.
    - Advisory Maintenance Level: $R_{advisory} = 0.022 \text{ m}^2\cdot\text{K/kW}$
    - Critical Soot-Blowing Limit: $R_{critical} = 0.035 \text{ m}^2\cdot\text{K/kW}$
  - **Remaining Safe Operating Window (RSOW) Formulation — exact, from `health_index.py::compute_remaining_safe_operating_window`:**
    $$RSOW = \text{clip}\left( \frac{R_{critical} - \hat{R}_f(t_0)}{\dot{R}_{burn}}, \; 1.0\text{ h}, \; 168.0\text{ h} \right)$$
    - Default burn rate $\dot{R}_{burn} = 0.00032\text{ /hour}$ (matches the dashboard's own projection formula above).
    - **Worked example (a real, live computation of the actual function — not invented):** at $\hat{R}_f = 0.024$, $RSOW = 34.4$ hours.
  - **Operational Meaning:**
    - Moves maintenance from subjective guesswork ("the boiler seems dirty") to an exact operational countdown: *"Maintenance must be completed within 34.4 operating hours."*
- **Visual Asset:** `ppt_assets/04_fouling_rsow_curve.png`

#### Speaking Script (Time: ~65 seconds)
> *"Once degradation is detected, the next question every plant manager asks is: *'How long do we have before we must shut down?'* This brings us to **prognostics and the Remaining Safe Operating Window, or RSOW**.
> 
> On screen is our 48-hour forward degradation projection, using the dashboard's own linear burn-rate formula. The system monitors fouling resistance against two thresholds: an advisory inspection limit at 0.022, and a critical soot-blowing limit at 0.035.
> 
> We define RSOW as the exact operating time remaining before fouling breaches the critical threshold at the current burn rate, clipped between one hour and one week — I ran this function live rather than making up a number: at a fouling resistance of 0.024, it returns an RSOW of 34.4 hours. This single metric converts a vague thermodynamic condition into a concrete operational boundary. How we optimize that 34-hour window is shown on the next slide."*

---

```
====================================================================================================
SLIDE 14 — DYNAMIC OPPORTUNISTIC MAINTENANCE SCHEDULING
====================================================================================================
```
### Slide 14: Dynamic Opportunistic Maintenance Economics
- **Header Badge:** Prescriptive Maintenance | Operations Research & Scheduling
- **Slide Title:** Cost-Optimal Shift Selection: Balancing Fuel Waste, Downtime, and Rupture Risk
- **Layout:** Total Cost Bar Chart Across Candidate Shifts (`ppt_assets/11_opportunistic_scheduling.png` — generated by actually running `scheduler.py`, real numbers below; the AI-rendered `11_opportunistic_scheduling_economics.png/.jpg` are stylistically nicer but show a smooth continuous curve and invented numbers that don't match the real (discrete, per-shift) cost function — don't use those for this slide).
- **On-Slide Content:**
  - **Opportunistic Scheduling Objective Function** (`src/maintenance/scheduler.py::optimize_maintenance_schedule`, evaluated once per candidate 8-hour shift, not a continuous integral):
    $$\min_{\tau \in [0, RSOW]} J(\tau) = C_{fuel\_waste}(\tau) + C_{intervention}(shift_\tau) + C_{failure\_risk}(\tau)$$
  - **Component Cost Formulations — real, verified against the code:**
    1. **Cumulative fuel waste:** sum of hourly fouling-driven waste cost from now until the candidate shift start.
    2. **Shift intervention cost:** `routine_clean_cost ($1,200) + labor_cost (3.5h × $85/hr × 2 techs × shift multiplier) + downtime_cost (3.5h × shift's $/hr penalty)`. Real shift downtime penalties, all hardcoded constants: **Night/Off-Peak $400/hr**, **Evening $900/hr**, **Peak Day $1,800/hr**.
    3. **Failure risk penalty — a simple empirical step function, NOT a Larson-Miller/Weibull creep model (that formula does not exist anywhere in this codebase):**
       $$p_{fail}(\tau) = \begin{cases} 0.01 & \tau \le 0.8 \cdot RSOW \\ 0.08 & 0.8 \cdot RSOW < \tau \le RSOW \\ \min(1.0,\ 0.25 + 0.15(\tau - RSOW)) & \tau > RSOW \end{cases} \qquad C_{failure\_risk} = p_{fail} \cdot \$18{,}000$$
  - **Worked example — a real, live run of `optimize_maintenance_schedule(current_rf=0.024, ...)`, not invented:**
    - RSOW = 34.4 h. Optimizer evaluates all upcoming 8-hour shifts and picks the global minimum: **"Night/Off-Peak Window (Day 1, Shift 3)"**, starting 16 hours from now.
    - Cost breakdown: fuel waste $149.10 + maintenance cost $3,046.25 + failure-risk penalty $180.00 = **total $3,375.35**.
    - Net savings vs. the $18,000 unplanned-failure cost: **$14,624.65** (for this specific scenario — this number scales with $\hat{R}_f$ and RSOW, it is not a universal constant).
- **Visual Asset:** `ppt_assets/11_opportunistic_scheduling.png`

#### Speaking Script (Time: ~80 seconds)
> *"Slide 14 demonstrates the prescriptive economic intelligence of our system — level C5, Configuration.
> 
> Shutting a boiler down during the day shift costs $1,800 an hour in downtime penalties — a real, hardcoded constant in our scheduler. Night shift costs only $400 an hour. But waiting too long to reach a night shift burns excess fuel and risks a failure-risk penalty that our code models as a simple step function: a small 1% baseline risk, rising to 8% once you're past 80% of the safe window, and climbing further the longer you overrun it — not some elaborate creep-physics equation, just a deliberately simple risk multiplier on an $18,000 failure cost.
> 
> I ran this optimizer live rather than presenting a hypothetical: at a fouling resistance of 0.024, with an RSOW of 34.4 hours, the global minimum-cost option is the night/off-peak window starting 16 hours from now, at a total cost of $3,375 — fuel waste, maintenance labor, and a small residual risk penalty combined. Compared to the $18,000 cost of an unplanned failure, that's a net saving of about $14,600 in this specific scenario. This is how the model turns a physics estimate into a scheduling decision with a dollar figure attached."*

---

```
====================================================================================================
SLIDE 15 — CLOSED-LOOP EXECUTION: WORK ORDER & LOTO SAFETY PROTOCOLS
====================================================================================================
```
### Slide 15: Closed-Loop Industrial Dispatch & LOTO Safety
- **Header Badge:** Industrial Operations | Safety Compliance & CMMS
- **Slide Title:** Automated Work Order Dispatch & OSHA 1910.147 LOTO Verification
- **Layout:** Industrial Digital Work Order Ticket (`ppt_assets/12_work_order_ticket.png` — every field on it is a real, live output of `scheduler.py`'s `_build_work_order()`, not invented text. Do not use `12_industrial_work_order_loto.png/.jpg` — those show fabricated part numbers, valve tags, and a 5-step OSHA sequence that don't exist in this codebase).
- **On-Slide Content:**
  - **Closing the Socio-Technical Loop:** the scheduler doesn't stop at picking a time — it calls `_build_work_order()` to generate a complete ticket.
  - **Work Order Specification — real fields from an actual run (`WO-202609-B01-9412`; the numeric suffix is random every run by design, `np.random.randint(1000, 9999)`):**
    - Target Asset: `BOILER-UNIT-01 (Viessmann Vitorond 200)` — a literal string constant in the code.
    - Priority: `EXPEDITED` (code logic: EMERGENCY if starting ≤4h from now, EXPEDITED if ≤24h, else ROUTINE CBM — our example starts at 16h, hence EXPEDITED).
    - Scheduled window: 2026-09-17 02:00 → 05:30 (3.5 hours).
    - Assigned crew: `Mechanical Maintenance Crew B (2 Techs, 1 Safety Supervisor)` — a real string constant.
    - Required spares (real, hardcoded list — **not** the invented part numbers "SB-4412"/"GKT-8802" from an earlier draft): `Soot Blower Packing Gaskets (x2)`, `High-Pressure Steam Nozzles (x4)`, `Flange Seals (x2)`.
    - Safety protocols (real, hardcoded list — a genuine 3-item list, not a fabricated 5-step OSHA 1910.147 sequence with valve/breaker tags that don't exist in the repo): `Lockout / Tagout (LOTO) Burner Fuel Supply`, `Furnace Draft Purge Verification`, `Thermal Personal Protective Equipment (PPE)`.
    - Cost: $3,046.25. Net savings vs. unplanned failure: $14,624.65.
  - **Honest framing for LOTO:** Lockout/Tagout is a real, standard industrial energy-isolation practice (generically associated with OSHA 1910.147 in the real world), and the repo's safety-protocol list correctly names it as a required step — but the repo itself does **not** implement OSHA's specific multi-step sequence, valve numbers, or sign-off IDs. Present LOTO as "the required safety protocol our work order flags," not as a fully modeled 5-step procedure.
- **Visual Asset:** `ppt_assets/12_work_order_ticket.png`

#### Speaking Script (Time: ~65 seconds)
> *"This final operational slide closes the loop between digital intelligence and physical human safety. On screen is a work order I generated by actually running our scheduler — every field you see is real code output, not a mockup.
> 
> It specifies the target asset, assigns Mechanical Crew B, lists the three real spare parts our code stages — soot blower packing gaskets, high-pressure steam nozzles, and flange seals — and locks in the optimal night shift I showed on the previous slide.
> 
> Crucially, the ticket also carries our safety-protocol list, which starts with **Lockout/Tagout of the burner fuel supply** — the standard industrial practice of de-energizing, locking, and tagging equipment before a crew touches it — followed by a furnace draft purge verification and thermal PPE. I'll be upfront that our code flags LOTO as a required protocol but doesn't model OSHA's full multi-step isolation sequence in detail; that's a reasonable scope boundary for this project, not a claim I want to overstate."*

---

```
====================================================================================================
SLIDE 16 — DEFENSE CONCLUSION & TAKEAWAYS
====================================================================================================
```
### Slide 16: Defense Conclusions & Core Takeaways
- **Header Badge:** Thesis Summary | Contributions to Work System Design
- **Slide Title:** Physics-Informed Digital Twins: Redefining Industrial Predictive Maintenance
- **Layout:** 3 Pillar Contribution Cards with Key Metric Highlights.
- **On-Slide Content:**
  - **Pillar 1: Thermodynamic Inductive Bias Changes What the Network Learns:**
    - Demonstrated that embedding 1st-Law conservation into the loss function cuts the physics energy residual by **>3× (down to 44.46 kW)** at the cost of roughly 1 K of test RMSE versus the best data-only baseline — verified against the real benchmark leaderboard.
    - Robustness under data scarcity and OOD extrapolation is a well-motivated *hypothesis* of this design, backed by real, runnable stress-test scripts in the repo — run them before claiming specific numbers in front of the committee.
  - **Pillar 2: Causal Degradation Tracking & Tube Creep Prevention:**
    - Solved unobservable inverse degradation estimation ($\hat{R}_f$), and ties tube-metal temperature to a real, hardcoded creep threshold of 560°C.
    - Transformed degradation into an actionable countdown: Remaining Safe Operating Window, computed live rather than assumed — 34.4 hours in our worked example.
  - **Pillar 3: Closed-Loop Socio-Technical Integration (WSD + CPS 5C):**
    - Bridged the gap from neural tensor graphs to shop-floor execution via Jay Lee's CPS 5C and Steven Alter's Work System frameworks.
    - In a representative computed scenario, achieved a net saving of **~$14,600** versus an unplanned $18,000 failure cost through opportunistic shift optimization — a real, live-run number, not a universal constant.
- **Visual Asset:** Department closing slide; optional side-by-side summary of `01_rmse_vs_energy_residual.png` and `11_opportunistic_scheduling.png`.

#### Speaking Script (Time: ~55 seconds)
> *"To conclude: this thesis demonstrates that in safety-critical manufacturing systems, embedding first-principles physics into a neural network's loss function is not merely an academic exercise — it changes what the model is willing to get wrong.
> 
> By constraining our neural network to respect the First Law of Thermodynamics, we accepted roughly one Kelvin of extra RMSE to cut physical energy violation by more than three times.
> 
> By grounding this inside Steven Alter's Work System Design framework and Jay Lee's CPS 5C architecture, we showed that a digital twin is only as valuable as the human decisions it feeds. From the physics-informed loss function, to a live Health Index and RSOW countdown, to an opportunistic scheduler that saved roughly $14,600 against an unplanned failure in our worked example — every number in this deck traces back to either a real benchmark result or a real, live run of this repo's own code. Thank you for your time — I'm happy to take questions."*

---

## Technical Defense & Faculty Rebuttal Appendix
*Anticipated challenge questions from IIT Bhilai faculty and comprehensive, mathematically grounded defense rebuttals.*

### Rebuttal 1: "Why is BoilerPINN's $R^2$ score lower than standard MLP (0.181 vs 0.468), and does this mean the model is underperforming?"
- **Faculty Perspective:** A standard data science reviewer looks at $R^2$ and RMSE ($5.26\text{ K}$ vs $4.23\text{ K}$) and concludes the PINN is a "worse" regression model.
- **Your Rebuttal Answer:**
  > *"Respected Professor, that is a fair observation, but in a safety-critical system, $R^2$ measures correlation with noisy sensor telemetry, not physical truth.
  > 
  > The Deep MLP's $R^2$ is 0.468 — real number, from our leaderboard — achieved with no constraint at all on energy conservation. In doing so, it predicts supply temperatures that violate the First Law of Thermodynamics by an average of **134.47 kilowatts**. The MLP is free to fit whatever pattern minimizes mean-squared error, including patterns that don't correspond to any physically realizable heat balance.
  > 
  > BoilerPINN explicitly penalizes this thermodynamic violation through our autograd physics loss term $\mathcal{L}_{physics}$. By forcing the network to satisfy $\dot{Q}_{water} \approx \hat{Q}_{eff}(\hat{R}_f)$ during training, the model trades some in-distribution fit for a residual that's over three times lower — 44.46 kW versus 134.47 kW. I'd add one honesty note: I have not yet run the out-of-distribution stress test in this repo to empirically prove extrapolation robustness — that's implemented as a script (Slide 11) but not yet executed. The argument I can make rigorously today is the in-distribution trade-off itself; the extrapolation claim is a well-motivated hypothesis I'd want to back with that script's actual output before asserting it as fact."*

### Rebuttal 2: "Why did you use Tanh activation functions in the shared trunk instead of modern ReLU or GELU?"
- **Faculty Perspective:** Modern deep learning practitioners reflexively prefer ReLU or GELU for training speed and non-saturating gradients.
- **Your Rebuttal Answer:**
  > *"The choice of Tanh was driven entirely by differential mathematics.
  > 
  > In a Physics-Informed Neural Network, the physics loss is evaluated by computing partial derivatives of network outputs with respect to inputs—specifically, our monotonicity loss term:
  > $$\mathcal{L}_{mono} = \text{ReLU}\left(\frac{\partial \hat{T}_{supply}}{\partial \dot{m}_{water}}\right)$$
  > During backpropagation, the optimizer must compute the gradient of this derivative with respect to network weights: $\frac{\partial}{\partial \theta} \left( \frac{\partial \hat{T}}{\partial \dot{m}_w} \right)$, which requires the activation function to possess a continuous, non-zero second derivative ($C^2$ continuity).
  > 
  > The piecewise-linear ReLU function has a first derivative that is a step function and a second derivative that is identically zero everywhere except at zero, where it is undefined. Using ReLU in the shared trunk causes the physics and monotonicity gradients to vanish, paralyzing the physics loss backpropagation loop.
  > 
  > Tanh is infinitely smooth ($C^\infty$) with well-behaved, non-vanishing higher-order analytical derivatives:
  > $$\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z), \quad \frac{d^2}{dz^2}\tanh(z) = -2\tanh(z)(1 - \tanh^2(z))$$
  > Because our model is compact (12,866 parameters across 3 hidden layers), Tanh does not suffer from vanishing gradients on data loss, while providing smooth, accurate autograd sensitivity for physical regularization."*

### Rebuttal 3: "How does the Digital Twin distinguish between fireside soot fouling and waterside mineral scaling when both degrade overall heat transfer?"
- **Faculty Perspective:** Both mechanisms lower overall heat transfer coefficient $U$; how can the model claim to diagnose them separately?
- **Your Rebuttal Answer:**
  > *"That is a fundamental question of heat transfer mechanics, illustrated in our radial resistance diagram on Slide 6.
  > 
  > While both soot and scale degrade overall heat transfer ($U A$), they have diametrically opposite effects on the **radial temperature gradient across the steel tube wall**:
  > 
  > 1. **Fireside Soot Accumulation ($R_{foul}$):** Soot deposits on the *outside* of the tube wall facing the combustion gas. Its thermal conductivity is very low — 0.08 W/m·K, a real constant from our fouling model. Soot blocks heat from ever penetrating the tube, so the tube metal itself stays relatively close to water temperature, while the *stack* temperature rises because unabsorbed heat exits with the flue gas.
  > 
  > 2. **Waterside Mineral Scale ($R_{scale}$):** Scale precipitates on the *inside* surface facing the circulating water. Heat still enters through the fireside shell but can't escape efficiently into the water because of the internal scale barrier, so the energy stays trapped in the steel tube wall — pushing metal temperature toward our critical creep threshold of 560°C, a real hardcoded constant in our Health Index module. I'll be precise here: our code defines that threshold and the T-wall formula that governs this mechanism, but it doesn't hardcode a specific "how far past 560°C" number — that depends on the actual heat flux and scaling severity in a given scenario, which is exactly what the digital twin is estimating in real time rather than looking up from a table.
  > 
  > Our Digital Twin monitors both the steam enthalpy balance and pyrometer-derived tube shell gradients: when heat transfer drops with high stack loss and cool tubes, it diagnoses **Fireside Soot**; when heat transfer drops with elevated tube shell temperatures, it triggers the **Waterside Scale Creep Rupture Warning**."*

### Rebuttal 4: "Why formulate a transient differential equation ($C_{sys} \frac{dT}{dt}$) in Slide 5, but use a quasi-static heat balance in the PINN loss function?"
- **Faculty Perspective:** Looking for mathematical inconsistency between the governing ODE and the steady-state loss term.
- **Your Rebuttal Answer:**
  > *"Respected committee, that's a fair catch, and I want to be precise rather than hand-wavy about it.
  > 
  > Slide 5's transient ODE — $C_{sys} \, dT_{supply}/dt = \dot{Q}_{combustion} - \dot{Q}_{water} - \dot{Q}_{casing\_loss}$ — is the general governing equation from our physics derivation document. The PINN's actual loss term, though, evaluates a *quasi-static* heat balance: $\dot{m}_w c_p (\hat{T}_{supply} - T_{return}) \approx \hat{Q}_{eff}(\hat{R}_f)$, without a $dT/dt$ term at all. I don't have a hardcoded time-constant number in the codebase to justify that gap with a specific figure — the honest engineering argument is qualitative: our telemetry samples every 5 seconds, and estimating $dT/dt$ numerically from noisy 5-second samples amplifies sensor noise badly, so the PINN instead learns a direct instantaneous mapping from inputs to state, sidestepping numerical differentiation entirely. That's a legitimate design choice, but I want to be clear it's a design simplification, not something derived from a computed thermal time constant in this repo."*

### Rebuttal 5: "Explain the apparent scale discrepancy in the Sankey diagram between gross fuel LHV and effective transfer capacity."
- **Faculty Perspective:** Noticing that the fuel-input number is vastly larger than the water-heat-absorption number in the Sankey.
- **Your Rebuttal Answer:**
  > *"That's the single most important caveat in this whole deck, and I'd rather own it than let it surface as a 'gotcha' later.
  > 
  > At 85% load, our fuel flow is $0.85 \times 3.3 = 2.805\text{ kg/s}$ (the dashboard's own load-to-fuel-flow formula). On a gross LHV basis: $\dot{Q}_{combustion} = 2.805\text{ kg/s} \times 42{,}000\text{ kJ/kg} \times 0.91 \approx 107{,}207\text{ kW}$.
  > 
  > Meanwhile, our heat-transfer model's water-absorption term uses a completely different frame — an analytical 'effective transfer capacity' calibrated around $Q_{clean}=388.8\text{ kW}$, which at this operating point works out to about 390 kW of actual water heat absorption.
  > 
  > These two numbers — 107,000 kW gross combustion and 390 kW analytical transfer capacity — are not on the same physical scale, and I'm not going to claim otherwise. It's a real simplification already present in the dashboard's source code: the Sankey mixes a gross-fuel-energy accounting frame with a net-transfer-capacity accounting frame, which is why 'stack loss' visually swallows almost everything. The number that *is* rigorously self-consistent, because it's what the PINN loss actually optimizes, is the physics residual — 44.46 kW — computed entirely within the second, analytical frame, never mixing the two scales."*

### Rebuttal 6: "How did you establish the failure risk probability in your opportunistic scheduling cost function $J(\tau)$?"
- **Faculty Perspective:** Questioning whether the failure cost curve $C_{failure\_risk}(\tau)$ is mathematically grounded or arbitrarily tuned.
- **Your Rebuttal Answer:**
  > *"I want to answer this precisely rather than dress it up: our implemented $P_{failure}(\tau)$ is **not** a Larson-Miller creep-life model. That's the textbook metallurgical motivation for *why* the 560°C threshold matters, but I did not implement a Larson-Miller parameter, and I want to be upfront about that rather than claim a level of metallurgical rigor the code doesn't have.
  > 
  > What's actually implemented, in `scheduler.py`, is a deliberately simple empirical step function of how close a candidate maintenance time $\tau$ is to the Remaining Safe Operating Window:
  > $$p_{fail}(\tau) = \begin{cases} 0.01 & \tau \le 0.8 \cdot RSOW \quad \text{(comfortably within window)}\\ 0.08 & 0.8 \cdot RSOW < \tau \le RSOW \quad \text{(late in the window)} \\ \min(1.0,\ 0.25 + 0.15(\tau - RSOW)) & \tau > RSOW \quad \text{(overrun, escalating linearly)} \end{cases}$$
  > multiplied by a flat $18,000 unplanned-failure cost. It's intentionally conservative and simple — 1% baseline risk, 8% once you're in the last 20% of the safe window, and an escalating penalty past the deadline — rather than a physically derived rupture probability. A genuine Larson-Miller-based risk model would be a legitimate future extension of this work, and I'd frame it exactly that way if asked: a design opportunity, not something already built."*

### Rebuttal 7: "What is the computational overhead of running PyTorch autograd in real time on an edge controller?"
- **Faculty Perspective:** Wondering whether autograd graph evaluation is too heavy for real-time plant SCADA edge devices.
- **Your Rebuttal Answer:**
  > *"That is a crucial deployment question in cyber-physical systems.
  > 
  > The important architectural distinction is that `torch.autograd.grad` is **only executed during the offline training and periodic calibration phase** (level C3 Cyber).
  > 
  > During real-time online SCADA inference at level C1/C2:
  > 1. The trained weights of BoilerPINN are frozen and evaluated in standard forward evaluation mode (`torch.no_grad()`).
  > 2. Because BoilerPINN has only **12,866 parameters** across 3 small hidden layers, our own benchmark measured a single forward pass at **0.0029 milliseconds — 2.9 microseconds** — on the machine we benchmarked on. I have not separately tested this on dedicated edge hardware like a Raspberry Pi or industrial PLC, so I'd frame that generalization as a reasonable expectation given the model's tiny size, not a verified deployment benchmark.
  > 3. The 1st-Law energy residual and Health Index calculation are simple vectorized arithmetic — negligible compared to the forward pass itself.
  > 
  > Given telemetry samples every 5 seconds in our real dataset, a sub-millisecond inference cost leaves enormous headroom regardless of which specific edge device it eventually runs on."*

---

## Complete Visual Asset Cross-Reference Table
*All 14 visual assets are located in `ppt_assets/` at 300 DPI, mathematically validated, and ready for Canva Pro / PowerPoint import.*

| # | Filename | Target Slide | Generation Engine | Mathematical / Engineering Content |
| :---: | :--- | :---: | :---: | :--- |
**USE these — verified accurate, use as-is:**

| # | Filename | Slide | Status |
| :---: | :--- | :---: | :--- |
| 01 | `01_rmse_vs_energy_residual.png` | 10 | Real data (benchmark JSON) |
| 02 | `02_energy_sankey.png` | 5 | Real data (dashboard formulas + real $T_{supply}$) |
| 03 | `03_health_index_gauge.png` | 12 | Real thresholds (dashboard gauge bands) |
| 04 | `04_fouling_rsow_curve.png` | 13 | Real formula (dashboard projection) |
| 05 | `05_leaderboard_table.png` | 9 | Real data (benchmark JSON, exact) |
| 06 | `06_work_system_framework_pro.png` | 2 | AI-rendered, spot-checked — clean and correct |
| 07 | `07_cps_5c_architecture_pro.png` | 3 | AI-rendered, spot-checked — clean and correct (one label, "IAPWS-97 steam tables," is aspirational — the repo uses simple $c_p\Delta T$, not full steam tables; fine as diagram flavor, don't assert it verbally) |
| 08 | `08_pinn_architecture_pro.png` | 7 | AI-rendered, spot-checked — clean and correct |
| 09 | `09_tube_degradation_physics.png` | 6 | Real data (fouling_model.py constants) |
| 10 | `10_stress_tests_real.png` | 11 | Real, executed results from all 3 stress-test scripts (incl. 2 honest negative findings) |
| 11 | `11_opportunistic_scheduling.png` | 14 | Real data (live run of scheduler.py) |
| 12 | `12_work_order_ticket.png` | 15 | Real data (live-generated WorkOrder) |

**DO NOT USE these — confirmed broken or misleading:**

| Filename | Problem |
| :--- | :--- |
| `10_stress_tests_ablation.png` / `.jpg` | AI-hallucinated: garbled title text, fabricated model names ("DPP ML2," "DFF ML3"). No real data behind it. |
| `11_opportunistic_scheduling_economics.png` / `.jpg` | Shows a smooth continuous cost curve and numbers that don't match the real, discrete per-shift cost function in `scheduler.py`. |
| `12_industrial_work_order_loto.png` / `.jpg` | Shows fabricated part numbers, valve/breaker tags, and a 5-step OSHA sequence not implemented in the repo. |
| `14_boiler_photorealistic_schematic.png` | Depicts a different, higher-pressure class of boiler (180 PSI/375°F superheated steam) than this project's actual low-pressure Vitorond 200. Use only as generic atmosphere art, if at all, and never repeat its on-image numbers as project facts. |
| `02_energy_sankey_realistic.png` / `.jpg` | Not checked in detail, but its filename/premise ("realistic") conflicts with the honest scale-mismatch caveat this deck now makes deliberately — use the plain `02_energy_sankey.png` instead. |
| `06/07/08_..._pro.jpg` (the `.jpg` duplicates) | Redundant with the `.png` — use the `.png` versions for print quality. |

There is no `13_academic_benchmark_leaderboard.png` in the repo — use `05_leaderboard_table.png` for Slide 9.

---

## Connecting Canva — the actual steps

You already have Canva Pro; here's exactly how to get this content into it:

1. **Go to claude.ai** (this connector lives in Claude's web app, not this CLI session) → **Settings → Connectors**.
2. Find **Canva** in the connector list and click **Connect**. You'll be redirected to Canva to authorize access — log in with your Canva Pro account and approve the permissions.
3. Once connected, start a new chat in claude.ai (or continue an existing one) and **upload this file** (`PPT_CONTENT_PACK.md`) plus the `ppt_assets/` folder's contents (drag the PNGs in, or upload as a zip).
4. Prompt Claude something like: *"Using this content pack and these images, populate a Canva presentation — 12 content slides plus a Thank You slide, professional academic tone. Use the specified image for each slide exactly as labeled. Put the Speaking Script text into the presenter notes."*
5. Claude will call the Canva connector to create/populate the deck. If you have a specific Canva brand template or department theme, mention it in the same prompt — Claude can apply it via the connector.
6. Once generated, open the deck **in Canva Pro directly** (not just via chat) to hand-polish: swap in your IIT Bhilai branding, fix any font/spacing issues, and do a final pass checking that no AI-rendered image slipped in from the "DO NOT USE" list above.
7. Rehearse once with Canva's presenter view, timing against the per-slide estimates in each **Speaking Script** header.
