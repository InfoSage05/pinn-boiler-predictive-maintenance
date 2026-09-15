# PPT Content Pack — Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers

**Course:** Work System Design, IIT Bhilai | **Format:** 14 slides + appendix | **Tone:** Professional academic

Paste this whole document into Claude.ai (with the Canva connector active) with the instruction:
> "Populate my [department] template with this outline. 14 content slides, professional academic tone. Use the 5 chart images in ppt_assets/ where referenced. Keep equations as text/LaTeX, not re-drawn."

Upload the 5 files in `ppt_assets/` alongside this doc when you do that.

---

## ⚠️ Known repo inconsistencies to resolve before the defense

These surfaced while extracting content. Decide how to handle each — I've suggested a default in brackets.

1. **PINN input dimensionality:** `docs/physics_derivation.md` writes the PINN as `N_θ(ṁ_fuel, T_air, T_return, ṁ_water, t)` — 5 inputs including time. The actual `BoilerPINN` code uses `in_features=4` and never passes time. **[Default: present the 4-input equation on the architecture slide, drop `t` from the formula, mention it's a steady-state/quasi-static formulation.]**
2. **Monotonicity loss term:** Docs describe two monotonicity penalties (`∂T/∂ṁ_water` and `-∂Rf/∂t`); code only implements the first. **[Default: present only the implemented term on the loss-function slide.]**
3. **`configs/default_config.yaml`** is referenced in the README's directory layout but does not exist in the repo — physics constants are hardcoded in `.py` files instead. **[Default: don't mention this in the deck; it's a repo hygiene item, not a defense topic. Fix it in the repo separately if you want the README accurate.]**
4. **Dashboard Tab 3 loss curves are synthetic** (explicitly commented "for illustration" in the code) — not real training history. **[Default: do NOT use these in the PPT. Slide 10 below uses only the real JSON leaderboard.]**
5. **Sankey energy-flow scale mismatch:** the dashboard's own Sankey mixes gross fuel LHV energy (~107,000 kW) with the model's analytical "effective transfer capacity" (~390 kW), making stack loss look implausibly large (~99%). This is a real simplification in the source code, not something introduced here. **[Default: caption the Sankey slide as "illustrative energy-flow structure, not calibrated to real boiler efficiency" — see speaker notes on Slide 6.]**
6. **PINN accuracy trade-off:** PINN has *worse* raw RMSE (5.26 K) and R² (0.181) than all 4 data-only baselines, but a **>3x lower physics residual** (44.46 kW vs ~134-144 kW). Own this directly in the talk — it's the central scientific finding, not a weakness to hide.

---

## Slide-by-slide outline

### Slide 1 — Title
**Content:**
- Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers
- Subtitle: A Cyber-Physical Work System Design Framework for Thermodynamic State Estimation, Degradation Prognosis, and Dynamic Maintenance Scheduling
- IIT Bhilai — Work System Design Course | [Your name] | [Date]
**Visual:** IIT Bhilai cover branding (hand-polish in Canva)
**Speaker notes:** Frame the project in one sentence before diving in: "This is not just a machine-learning curve-fit — it's a socio-technical work system that predicts when a boiler needs maintenance before it fails, using a neural network constrained by the laws of physics."

---

### Slide 2 — The Problem & Why It's a Work System, Not Just an ML Model
**Content:**
- Industrial boiler (Viessmann Vitorond 200 / coal-fired boiler) degrades invisibly: fireside soot fouling + waterside scaling
- Consequence chain: fouling → efficiency loss → fuel waste → tube overheating → risk of catastrophic rupture
- Framed via **Steven Alter's Work System Framework** (9 elements): Participants (Control Operator, Maintenance Tech, Reliability Engineer, Plant Ops Manager) + Processes + Information + Technology, situated in an Environment, producing Products/Services for Customers
**Visual:** Simple Alter 9-element diagram (build in Canva from template — not auto-generated here)
**Speaker notes:** Emphasize this is a course in Work System Design — the technical contribution (PINN) is one Technology element serving a human decision process, not an end in itself.

---

### Slide 3 — Four Project Goals
**Content (from README Executive Summary):**
1. Estimate unobservable degradation (fireside fouling resistance Rf, waterside scaling) in real time
2. Predict future thermal trajectories that respect conservation laws (via `torch.autograd.grad`)
3. Mitigate operator alarm fatigue via ISA-18.2-compliant root-cause explainability
4. Optimize opportunistic maintenance scheduling, balancing fuel-waste cost vs. shift-dependent downtime loss vs. tube creep-rupture risk
**Speaker notes:** These four goals map directly to the four demo tabs later in the talk (Cockpit → Prognostics → Physics Inspector → Scheduler) — tell the audience to watch for that mapping.

---

### Slide 4 — CPS 5C Architecture (Jay Lee, 2015)
**Content — table:**
| Level | Name | Role in this project |
|---|---|---|
| C1 | Connection | Telemetry: temperature, pressure, water mass flow, fuel firing rate, flue gas O2 |
| C2 | Conversion | Feature normalization, sensor health checks, thermodynamic property lookup |
| C3 | Cyber | Digital Twin state model + PyTorch PINN dual-head engine |
| C4 | Cognition | Health Index (HI ∈ [0,1]), **RSOW** (Remaining Safe Operating Window), ISA-18.2 root-cause diagnostics |
| C5 | Configuration | Opportunistic shift scheduler, automated work-order generation (crew, LOTO), What-If sandbox |
**Speaker notes:** This is the backbone of the whole system — every later slide is one of these five levels made concrete.

---

### Slide 5 — Governing Physics: 1st Law Energy Balance
**Content (exact, from docs/physics_derivation.md §1):**
- Transient control-volume energy balance (no shaft work):
  `dE_cv/dt = Q_combustion - Q_fluid - Q_loss`
- Governing ODE: `C_sys · dT_supply/dt = Q_combustion - Q_water - Q_casing_loss`
- `Q_combustion(t) = ṁ_fuel(t) · LHV · η_comb(λ)`, LHV ≈ 42,000 kJ/kg
- `Q_water(t) = ṁ_water(t) · cp · [T_supply(t) - T_return(t)]`
- `Q_casing_loss(t) = U_loss · A_shell · [T_supply(t) - T_ambient]`
**Speaker notes:** Point out that every one of these terms is measured or measurable — this equation is not a black box, it's classical thermodynamics, and it's exactly what gets baked into the neural network's loss function two slides from now.

---

### Slide 6 — Energy Flow Visualization
**Visual:** `ppt_assets/02_energy_sankey.png`
**Content:** Sankey diagram of the 1st Law breakdown at a representative 85% load operating point — fuel chemical energy → heat absorbed by water/steam, ambient casing loss, fireside fouling waste, stack flue-gas loss.
**Speaker notes — important caveat, say this out loud, don't skip it:** "This diagram reuses the exact flow formulas from our Streamlit dashboard. One honest limitation: the fuel-energy term uses gross LHV combustion energy while the water-heat term uses our model's analytical 'effective transfer capacity,' which are calibrated on different scales — so the stack-loss share here is illustrative of *structure*, not a calibrated efficiency number. We show it to communicate where the 1st Law terms conceptually go, not as a combustion-efficiency claim."

---

### Slide 7 — Fouling & Thermal Resistance Network
**Content (docs/physics_derivation.md §2):**
- Total thermal resistance: `R_total(t) = 1/(U(t)·A) = 1/(h_gas·A_o) + R_fouling(t)/A_o + ln(r_o/r_i)/(2πk_metal·L) + R_scaling(t)/A_i + 1/(h_water·A_i)`
- Kern-Seaton asymptotic deposition-removal model: `dRf/dt = ṁ_deposition - β·τ_shear·Rf(t)`
- Steady-state solution: `Rf(t) = R_clean + (R_asymptotic - R_clean)·[1 - exp(-t/τ_foul)]`
**Speaker notes:** This is the physical mechanism the "inverse head" of the PINN is trying to identify — fouling resistance is never measured directly, only inferred from its effect on heat transfer.

---

### Slide 8 — PINN Architecture
**Content:**
- `BoilerPINN`: shared trunk `Linear(4→64) → Tanh → Linear(64→64) → Tanh → Linear(64→64) → Tanh` (Tanh chosen for smooth 2nd-order autograd derivatives)
- Inputs (4): `ṁ_fuel, T_air, T_return, ṁ_water`
- Head 1 (forward/state): `Linear(64→32) → Tanh → Linear(32→1)` → `T_supply_hat`
- Head 2 (inverse/degradation): `Linear(64→32) → Tanh → Linear(32→1) → Softplus` → `Rf_hat` (Softplus enforces Rf ≥ 0)
- 12,866 trainable parameters
**Visual:** Simple dual-head network diagram (build in Canva)
**Speaker notes:** Flag the input-dimensionality note from the errata list only if asked — the equation on Slide 5 has a `t` term that isn't literally an input here; this is a steady-state formulation.

---

### Slide 9 — Composite Physics-Informed Loss Function
**Content (exact, from pinn_model.py):**
`L_total = w_data·L_data + w_phys·L_physics + w_mono·L_mono + w_bound·L_boundary + w_inverse·L_inverse`

- `L_data` — MSE on scaled temperature prediction
- `L_physics = mean[(Q_absorbed_water - Q_eff_predicted) / 50]²` — energy-imbalance residual, autograd-computed
- `L_mono = mean[ReLU(∂T_supply_hat/∂ṁ_water)]` — penalizes the unphysical case where more water flow raises outlet temperature
- `L_boundary = mean[ReLU(T_return - T_supply_hat)]` — enforces `T_supply ≥ T_return`
- `L_inverse` — supervised MSE against true fouling/scaling labels
- Default weights: `w_data=1.0, w_phys=0.15-0.20, w_mono=0.05, w_bound=0.02, w_inverse=0.50`
**Speaker notes:** This is the single slide that makes the project a *PINN* rather than an ordinary neural net — every term beyond `L_data` bakes a physical law directly into gradient descent via `torch.autograd.grad`.

---

### Slide 10 — 6-Model Comparative Benchmark
**Visual:** `ppt_assets/05_leaderboard_table.png`
**Content (real numbers, experiments/results/benchmark_leaderboard.json):**
| Model | RMSE (K) | R² | Physics Residual (kW) |
|---|---|---|---|
| Physics-Only (zero data) | 7.37 | -0.611 | 0.000 |
| Polynomial Ridge | 4.22 | 0.472 | 143.62 |
| Random Forest | 4.56 | 0.384 | 142.21 |
| Deep MLP (data-only) | 4.23 | 0.468 | 134.47 |
| LSTM | 4.20 | 0.476 | 142.66 |
| **PINN** | **5.26** | **0.181** | **44.46** |
**Speaker notes:** Say the honest thing directly: "PINN's raw RMSE and R² are worse than every purely data-driven baseline — we're not hiding that. But look at the physics-residual column: every data-only model violates energy conservation by 130-145 kW on average. The PINN cuts that to 44 kW, a >3x reduction. That's the actual contribution — a model that's a little less accurate on temperature but consistent with the laws of physics it operates under, which matters when you use it to extrapolate beyond the training distribution."

---

### Slide 11 — Accuracy vs. Physical Consistency Trade-off
**Visual:** `ppt_assets/01_rmse_vs_energy_residual.png`
**Content:** Scatter plot, RMSE (x) vs. Energy-Balance Residual (y), all 6 models, PINN highlighted
**Speaker notes:** This is the visual argument for the whole thesis in one chart — data-only models cluster in the top-left (accurate but physically inconsistent); PINN sits alone in the bottom-middle (accepts a modest accuracy cost for a large physical-consistency gain). Physics-Only sits at the true zero-residual point but with the worst RMSE, showing the two extremes PINN is trading off between.

---

### Slide 12 — Digital Twin Cockpit: Health Index & Live Monitoring
**Visual:** `ppt_assets/03_health_index_gauge.png`
**Content:**
- Multi-criteria Health Index (HI ∈ [0,1]): combines fouling severity, tube-metal overheating margin, and energy-balance error
- Color bands: 0-30 critical (red), 30-60 warning (amber), 60-100 healthy (green)
- ISA-18.2-compliant root-cause alarm cards: Fireside Soot Fouling vs. Waterside Scale vs. Combustion Excess Air — designed against operator alarm fatigue
**Speaker notes:** This is C4 (Cognition) made visible — the gauge is the one number an operator glances at; the root-cause card is what they read next instead of a wall of raw alarms.

---

### Slide 13 — Prognostics: Fouling Trajectory & RSOW
**Visual:** `ppt_assets/04_fouling_rsow_curve.png`
**Content:**
- 48-hour fouling projection with two thresholds: advisory inspection limit (Rf=0.022) and critical soot-blowing threshold (Rf=0.035)
- **RSOW (Remaining Safe Operating Window)** — the countdown to the critical threshold, computed as `RSOW = (Rf_critical - Rf_current) / burn_rate`, clipped to [1, 168] hours
- Cost-optimization framing (docs/maintenance_strategy.md): schedule intervention at time τ* that minimizes `J(τ) = C_fuel_waste(τ) + C_intervention(shift_τ) + C_failure_risk(τ)`, where failure risk derives from the **Larson-Miller parameter** for tube-metal creep
**Speaker notes:** RSOW is this project's core prognostic concept — it converts "the boiler is degrading" (a vague statement) into "you have 41 hours before mandatory intervention" (an actionable number a shift scheduler can use).

---

### Slide 14 — Maintenance Scheduling, Work Orders & LOTO
**Content:**
- Maintenance paradigm evolution: Run-to-Failure → Time-Based → **Condition-Based (this project)** — continuous PINN-driven monitoring replacing fixed 30-day inspection cycles
- Off-peak/night-shift scheduling reduces production disruption by up to 75% (per shift-dependent cost model)
- Automated work-order generation: asset ID, priority, scheduled window, crew assignment, required spares (gaskets, nozzles), and **LOTO (Lockout/Tagout)** safety checklist
- Human-factors design: NASA-TLX-informed cognitive ergonomics — single root-cause diagnostic cards, predictive-horizon countdown instead of reactive scrambling, automated work-package assembly
**Speaker notes:** Close the loop back to Slide 2 — this is the "Process" and "Participant" elements of the work system acting on what the "Technology" (PINN + Digital Twin) computed. Define LOTO briefly for the audience: standard industrial energy-isolation safety procedure (de-energize, lock, tag) before maintenance crews touch equipment — the repo assumes this knowledge, so state it explicitly.

---

## Appendix slide (optional, for Q&A backup)
- Dataset details: Viessmann Vitorond 200 (27,280 samples, HySonLab/AgentIoT) + Real Industrial Coal-Fired Boiler Telemetry (14,400 samples, 5-second sampling)
- Stress-test suite: data-scarcity ablation (1-100%), OOD extrapolation, sensor-noise robustness (0-20% Gaussian), physics-loss weight sensitivity sweep
- Citations: Alter (2013) Work System Theory; Lee, Bagheri & Kao (2015) CPS 5C; Raissi, Perdikaris & Karniadakis (2019) PINNs; ISA-18.2/IEC 62682 alarm management standard

---

## Chart assets reference

| File | Used on slide | Source of truth |
|---|---|---|
| `ppt_assets/01_rmse_vs_energy_residual.png` | 11 | `experiments/results/benchmark_leaderboard.json` |
| `ppt_assets/02_energy_sankey.png` | 6 | `dashboard/app.py` Sankey formulas + `src/physics/boiler_thermo.py` |
| `ppt_assets/03_health_index_gauge.png` | 12 | `dashboard/app.py` Tab 1 gauge spec |
| `ppt_assets/04_fouling_rsow_curve.png` | 13 | `dashboard/app.py` Tab 2 fouling projection formula |
| `ppt_assets/05_leaderboard_table.png` | 10 | `experiments/results/benchmark_leaderboard.json` (real, not the dashboard's mock fallback table) |
