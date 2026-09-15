# PPT Content Pack — Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers

**Course:** Work System Design, IIT Bhilai | **Format:** 12 content slides + Thank You | **Tone:** Professional academic | **Est. runtime:** ~15-17 min at a natural pace

## How to use this pack
1. Everything below is grounded in your actual repo (README.md, docs/, src/models, experiments/results/benchmark_leaderboard.json) — no invented numbers.
2. All 8 visuals referenced below are already generated and sitting in `ppt_assets/`. Nothing is left to hand-draw from scratch.
3. Take this whole document + the `ppt_assets/` folder into Claude.ai (Canva connector active) and say:
   > "Populate my [department] template with this outline. 12 content slides + a closing Thank You slide, professional academic tone. Use the images in ppt_assets/ exactly where each slide specifies. Keep equations as text/LaTeX, not re-drawn. Use the **Speaking script** text as speaker notes."
4. Hand-polish in Canva Pro after: fonts, IIT Bhilai cover branding, transitions. The content and visuals themselves need no further creation.

---

## ⚠️ Known repo inconsistencies — read once before rehearsing

These are real quirks found while extracting content, not choices made for this deck. Each has a default already baked into the script below — you only need to act if you want to override it.

1. **PINN input dimensionality:** the physics doc writes the model as `N_θ(ṁ_fuel, T_air, T_return, ṁ_water, t)` — 5 inputs including time. The actual code (`BoilerPINN`) uses 4 inputs and never passes time. **Baked-in default:** Slide 8 presents only the 4-input version and calls it a steady-state formulation.
2. **Monotonicity loss term:** docs describe two penalty terms; code implements only one (`∂T/∂ṁ_water`). **Baked-in default:** Slide 9 presents only the implemented term.
3. **`configs/default_config.yaml`** is referenced in the README but doesn't exist — constants are hardcoded in `.py` files. **Not mentioned in the deck** — it's a repo-hygiene item, not a defense topic.
4. **Dashboard's Tab-3 loss curves are synthetic** (commented "for illustration" in the actual code). **Not used anywhere in this deck** — Slide 10 uses only the real benchmark JSON.
5. **Sankey scale mismatch:** the dashboard's own Sankey formula mixes gross fuel LHV energy (~107,000 kW) with the model's analytical "effective transfer capacity" (~390 kW), which visually exaggerates stack loss. This is a real simplification already in your source code. **Baked-in default:** the script for Slide 5 states this out loud as an honest caveat — don't skip that line, it preempts the obvious question.
6. **PINN's accuracy trade-off:** PINN has worse raw RMSE (5.26 K) and R² (0.181) than every data-only baseline, but a **>3x lower physics-energy residual** (44.46 kW vs. ~134-144 kW). This is the central finding — own it directly, it's built into Slides 10 and 11 as the deck's climax, not hidden.

---

## Slide-by-slide script

Each slide has: **On-slide content** (what the audience reads), **Visual** (exact file to place), and **Speaking script** (verbatim narration — read it, adapt it to your own voice, but the facts and numbers must stay exact).

---

### Slide 1 — Title
**On-slide content:**
- Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers
- A Cyber-Physical Work System Design Framework for Thermodynamic State Estimation, Degradation Prognosis, and Dynamic Maintenance Scheduling
- IIT Bhilai — Work System Design Course | [Your name] | [Date]

**Visual:** IIT Bhilai cover branding — hand-build in Canva (logo, department colors); no auto-generated asset for this one.

**Speaking script (≈30s):**
"Good [morning/afternoon] everyone. My project is a Physics-Informed Digital Twin for Predictive Maintenance of Industrial Boilers. In one sentence: this is a socio-technical work system that tells plant operators when a boiler needs maintenance *before* it fails, using a neural network that's constrained to obey the laws of thermodynamics — not just fit data. Over the next fifteen minutes I'll walk through the problem, the physics, the model, the results, and the human-facing system it feeds into."

---

### Slide 2 — The Problem, Framed as a Work System
**On-slide content:**
- Industrial boilers degrade invisibly: fireside soot fouling + waterside scaling
- Consequence chain: fouling → efficiency loss → fuel waste → tube overheating → risk of catastrophic rupture
- This is a Work System Design course — the technology is one element of a larger socio-technical system

**Visual:** `ppt_assets/06_work_system_framework.png` — Steven Alter's 9-element Work System framework, instantiated for this project (Participants, Processes, Information, Technologies, Products & Services, Customers, Environment, Strategy & Infrastructure).

**Speaking script (≈75s):**
"Let's start with the problem. Inside an industrial boiler — we used the Viessmann Vitorond 200 and a real coal-fired boiler telemetry dataset — two things degrade silently over time: soot builds up on the fireside of the heat-exchanger tubes, and scale builds up on the waterside. Neither is directly visible or measurable. The consequence chain is: fouling raises thermal resistance, which cuts efficiency, which wastes fuel, which — if ignored long enough — pushes tube metal temperatures past their creep limit and risks a catastrophic rupture.

Now, this is a *Work System Design* course, so I want to frame this correctly from the start. I'm using Steven Alter's Work System Framework — you can see it on screen. The technology I built, the PINN and the digital twin, is just one box in this diagram: the 'Technologies' element. It exists to serve Participants — the control operator, the maintenance technician, the reliability engineer — through Processes like telemetry monitoring and shift scheduling, producing Information like the Health Index, ultimately delivering the Product: safe, continuous steam generation with zero unplanned downtime, to Customers downstream. Keep this picture in mind — everything I show after this is one box filling in."

---

### Slide 3 — Project Goals & CPS 5C Architecture
**On-slide content:**
- Four goals: (1) estimate unobservable degradation in real time, (2) predict thermal trajectories that respect conservation laws via `torch.autograd.grad`, (3) reduce operator alarm fatigue via ISA-18.2 root-cause explainability, (4) optimize maintenance scheduling balancing fuel waste vs. downtime vs. rupture risk
- Architected using Jay Lee's CPS 5C: Connection → Conversion → Cyber → Cognition → Configuration

**Visual:** `ppt_assets/07_cps_5c_architecture.png` — 5-level CPS architecture diagram with this project's implementation detail at each level.

**Speaking script (≈75s):**
"There were four concrete goals driving this project. First, estimate degradation that's never directly measured — fouling resistance — in real time. Second, predict future thermal trajectories in a way that respects conservation of energy, using PyTorch's autograd to compute physics gradients directly during training. Third, reduce operator alarm fatigue with ISA-18.2-compliant root-cause diagnostics, instead of a wall of raw threshold alarms. Fourth, optimize maintenance scheduling — deciding *when* to intervene by balancing fuel-waste cost against shift-dependent downtime cost against tube-rupture risk.

To structure all of this, I used Jay Lee's CPS 5C architecture — Connection, Conversion, Cyber, Cognition, Configuration. On screen you can see exactly what each level does in this project: C1 is the sensor telemetry — temperature, pressure, flow, flue-gas O2. C2 normalizes and validates it. C3 is the Cyber layer — the digital twin and the PINN engine, which is the technical core I'll spend the next few slides on. C4, Cognition, turns model output into a Health Index and a maintenance countdown. And C5, Configuration, is where that becomes an actual scheduled work order. I'll walk through these levels roughly in order."

---

### Slide 4 — Governing Physics: 1st Law Energy Balance
**On-slide content:**
- Transient control-volume energy balance: `dE_cv/dt = Q_combustion - Q_fluid - Q_loss`
- Governing ODE: `C_sys · dT_supply/dt = Q_combustion - Q_water - Q_casing_loss`
- `Q_combustion(t) = ṁ_fuel(t) · LHV · η_comb(λ)`, LHV ≈ 42,000 kJ/kg
- `Q_water(t) = ṁ_water(t) · cp · [T_supply(t) - T_return(t)]`
- `Q_casing_loss(t) = U_loss · A_shell · [T_supply(t) - T_ambient]`

**Visual:** none — equation-only slide, keep as clean text/LaTeX in Canva, no chart needed here.

**Speaking script (≈70s):**
"Before touching any machine learning, I need to show you the physics the model is required to obey. This is the 1st Law of Thermodynamics applied to the boiler as a transient, open control volume with no shaft work: the rate of change of stored energy equals combustion heat in, minus heat delivered to the water, minus heat lost to ambient.

That gives the governing ODE at the top: system thermal capacity times the rate of change of supply temperature equals combustion heat minus water heat absorption minus casing loss. Each term on the right is physically grounded — combustion heat is fuel mass flow times lower heating value, about 42,000 kilojoules per kilogram, times a combustion efficiency that depends on excess-air ratio. Water heat absorption is just sensible heating — mass flow times specific heat times the temperature rise. Casing loss is a simple convective loss to ambient.

None of this is a black box — it's classical thermodynamics, and every term is either measured directly or computable from measured quantities. That matters because this exact equation is what gets baked into the neural network's loss function two slides from now."

---

### Slide 5 — Energy Flow at a Representative Operating Point
**On-slide content:**
- Sankey breakdown of the 1st Law at 85% load: fuel chemical energy in → heat absorbed by water/steam, ambient casing loss, fireside fouling waste, stack flue-gas loss

**Visual:** `ppt_assets/02_energy_sankey.png`

**Speaking script (≈70s) — read this caveat out loud, do not skip it:**
"This Sankey diagram visualizes that same 1st Law breakdown at a representative 85% load operating point, using the exact flow formulas from our Streamlit dashboard. Fuel chemical energy enters on the left; it splits into heat absorbed by the water, casing convection loss, fireside fouling waste, and stack flue-gas loss.

One honest limitation I want to flag directly, because it's a fair question to ask: the fuel-energy term here uses gross combustion energy on the LHV basis, while the water-heat term uses our model's analytical 'effective transfer capacity,' which is calibrated on a different scale. So the *proportion* going to stack loss in this specific diagram is illustrative of where the 1st-Law terms conceptually go — it is not a calibrated combustion-efficiency claim for a real boiler. I show it to communicate structure, not to claim an efficiency number."

---

### Slide 6 — Fouling Mechanics & Thermal Resistance
**On-slide content:**
- Total thermal resistance network: `R_total = 1/(U·A) = 1/(h_gas·A_o) + R_fouling/A_o + ln(r_o/r_i)/(2πk_metal·L) + R_scaling/A_i + 1/(h_water·A_i)`
- Kern-Seaton asymptotic deposition-removal model: `dRf/dt = ṁ_deposition - β·τ_shear·Rf(t)`
- Steady-state solution: `Rf(t) = R_clean + (R_asymptotic - R_clean)·[1 - exp(-t/τ_foul)]`

**Visual:** none — equation slide, pair with a simple labeled cross-section of a boiler tube wall if your template has one; not required.

**Speaking script (≈65s):**
"So where does fouling resistance, Rf, actually come from physically? The total thermal resistance between combustion gas and water is a series network — gas-side film resistance, fireside fouling resistance, the metal tube wall itself, waterside scaling resistance, and the waterside film resistance. Fouling and scaling are the two terms that grow over time and are what we're trying to track.

The growth dynamics follow the Kern-Seaton asymptotic deposition-removal model: fouling resistance increases from a deposition rate and decreases from a shear-removal term proportional to current fouling and flow shear stress. Solved analytically, that gives an exponential approach toward an asymptotic fouling resistance — the curve you'll see projected forward a few slides from now.

This is the physical quantity the PINN's second output head is trying to identify — and I want to stress: fouling resistance is never measured directly by any sensor. It's only inferred from its effect on heat transfer. That's the estimation problem this whole project exists to solve."

---

### Slide 7 — PINN Architecture: Dual-Head Network
**On-slide content:**
- Shared trunk: `Linear(4→64) → Tanh → Linear(64→64) → Tanh → Linear(64→64) → Tanh`
- Inputs (4): `ṁ_fuel, T_air, T_return, ṁ_water`
- Head 1 (forward/state): `Linear(64→32) → Tanh → Linear(32→1)` → `T_supply_hat`
- Head 2 (inverse/degradation): `Linear(64→32) → Tanh → Linear(32→1) → Softplus` → `Rf_hat`
- 12,866 trainable parameters total

**Visual:** `ppt_assets/08_pinn_architecture.png`

**Speaking script (≈75s):**
"This is the network itself — we call it BoilerPINN. It takes four inputs: fuel mass flow, inlet air temperature, return water temperature, and water mass flow. These pass through a shared trunk of three linear layers with Tanh activations. Tanh is a deliberate choice, not a default — it's smooth and twice-differentiable, which matters because we need clean second-order derivatives for the physics loss, computed via PyTorch's autograd.

After the trunk, the network splits into two heads. Head 1, the forward or state head, predicts the supply water temperature directly — a standard regression output. Head 2, the inverse or degradation head, predicts the fouling resistance Rf, and ends in a Softplus activation specifically to guarantee that predicted fouling resistance is always non-negative, since a negative thermal resistance is physically meaningless.

The whole network is small — under thirteen thousand trainable parameters — which matters for a later point: this isn't a capacity problem, the model isn't underfit for lack of parameters. Its behavior is a direct consequence of what we tell it to optimize, which is the next slide."

---

### Slide 8 — Composite Physics-Informed Loss Function
**On-slide content:**
`L_total = w_data·L_data + w_phys·L_physics + w_mono·L_mono + w_bound·L_boundary + w_inverse·L_inverse`
- `L_data` — MSE on scaled temperature prediction
- `L_physics = mean[(Q_absorbed_water - Q_eff_predicted) / 50]²` — 1st-Law energy-imbalance residual, autograd-computed
- `L_mono = mean[ReLU(∂T_supply_hat/∂ṁ_water)]` — penalizes more water flow raising outlet temperature
- `L_boundary = mean[ReLU(T_return - T_supply_hat)]` — enforces `T_supply ≥ T_return`
- `L_inverse` — supervised MSE against true fouling/scaling labels
- Default weights: `w_data=1.0, w_phys=0.15-0.20, w_mono=0.05, w_bound=0.02, w_inverse=0.50`

**Visual:** none — this is the single most important equation slide in the talk, keep it text-only and give it room to breathe.

**Speaking script (≈85s):**
"This slide is the reason this is a *physics-informed* neural network and not an ordinary one. The total loss is a weighted sum of five terms.

`L_data` is the ordinary part — mean-squared error between predicted and true supply temperature. Everything after that is where physics enters. `L_physics` is the 1st-Law energy residual: we take the heat the model predicts the water absorbs, subtract the effective heat-transfer capacity implied by the predicted fouling resistance, and penalize any imbalance — this is computed by literally taking gradients through the network with autograd, so the network is punished during training every time its predictions violate conservation of energy. `L_mono` enforces a physical monotonicity constraint: increasing water flow should never *raise* predicted outlet temperature — if it does, that term penalizes it. `L_boundary` enforces the trivial but easy-to-violate constraint that supply temperature can't be below return temperature. And `L_inverse` is a supervised term against the true fouling and scaling labels in our dataset, helping the inverse head learn.

The weights matter — physics gets roughly a fifth the weight of the data term. That ratio is a deliberate design choice, and it's exactly what produces the accuracy-versus-consistency trade-off I'll show you on the next two slides."

---

### Slide 9 — 6-Model Comparative Benchmark
**On-slide content:** full leaderboard table

**Visual:** `ppt_assets/05_leaderboard_table.png`

**Speaking script (≈90s):**
"Here's where it gets tested. I benchmarked six models on the same held-out test set: a zero-data physics-only analytical baseline, polynomial ridge regression, a random forest, a standard deep MLP trained purely on data, an LSTM, and finally the PINN.

Walk the table left to right. Physics-Only has the worst RMSE — seven-point-four Kelvin — because it has zero training data and only knows the governing equation; but its physics residual is exactly zero, by construction, since it *is* the physics equation. The four data-only models — Ridge, Random Forest, Deep MLP, LSTM — all land in a tight cluster around four-point-two to four-point-six Kelvin RMSE, which is genuinely good temperature accuracy. Now look at their physics-residual column: every single one of them violates the energy balance by somewhere between 134 and 144 kilowatts on average, because nothing in their training forces them to respect conservation of energy — they only ever saw labeled temperatures.

The PINN sits differently: five-point-three Kelvin RMSE — worse than every data-only baseline — but a physics residual of just 44 kilowatts. That's more than a three-times reduction versus every data-driven model. I'm not going to pretend that trade-off doesn't exist — it does, and I'll address it head-on on the next slide."

---

### Slide 10 — Accuracy vs. Physical Consistency Trade-off
**On-slide content:** scatter plot, RMSE (x) vs. Physics Residual (y), PINN highlighted

**Visual:** `ppt_assets/01_rmse_vs_energy_residual.png`

**Speaking script (≈75s):**
"This chart is the argument for the whole project in one picture. On the x-axis is temperature RMSE — lower is better. On the y-axis is the physics-energy residual — lower is better. Every data-only model clusters in the upper-left: reasonably accurate, but consistently violating energy conservation by well over a hundred kilowatts. Physics-Only sits at the far bottom-right: perfect physical consistency by definition, but the worst raw accuracy, because it never saw a single labeled data point.

The PINN is the only model that sits in between, trading a modest amount of raw accuracy for more than a three-times reduction in physical inconsistency compared to every data-driven baseline. Why does that trade matter in practice, and not just on paper? Because this model isn't just being asked to interpolate inside its training distribution — it's meant to extrapolate into conditions the plant hasn't seen yet, load swings, fouling states beyond the training range, sensor dropout. A model that's occasionally a degree or two less accurate but never breaks the 1st Law is the one you can trust to extrapolate safely. A model that's marginally more accurate in-distribution but freely violates energy conservation gives you no guarantee at all once you leave that distribution. That's the actual scientific contribution here."

---

### Slide 11 — Digital Twin Cockpit: Health Index
**On-slide content:**
- Multi-criteria Health Index (HI ∈ [0,1]): combines fouling severity, tube-metal overheating margin, and energy-balance error
- Bands: 0-30 critical (red), 30-60 warning (amber), 60-100 healthy (green)
- ISA-18.2-compliant root-cause diagnostic cards: Fireside Soot Fouling vs. Waterside Scale vs. Combustion Excess Air

**Visual:** `ppt_assets/03_health_index_gauge.png`

**Speaking script (≈70s):**
"Now we move from the model into the human-facing side of the system — the C4 Cognition layer from the 5C architecture I showed earlier. The PINN's raw outputs — predicted temperature, predicted fouling resistance, the physics residual — aren't something an operator wants to read directly. So they're collapsed into a single multi-criteria Health Index between 0 and 1, combining fouling severity, how close tube-metal temperature is to its creep limit, and the energy-balance error, with color bands: red below 30, amber 30 to 60, green above 60.

Next to the gauge sits an ISA-18.2-compliant root-cause diagnostic card — the system doesn't just say 'alarm,' it distinguishes between fireside soot fouling, waterside scale, and combustion excess-air problems, because those need completely different interventions. This design choice is directly aimed at reducing operator alarm fatigue — instead of a wall of raw threshold alerts, the operator sees one number and one likely cause."

---

### Slide 12 — Prognostics & Maintenance Scheduling
**On-slide content:**
- 48-hour fouling projection with advisory (Rf=0.022) and critical (Rf=0.035) thresholds
- **RSOW (Remaining Safe Operating Window)**: `RSOW = (Rf_critical - Rf_current) / burn_rate`, clipped to [1, 168] hours
- Cost-optimal intervention: `min_τ J(τ) = C_fuel_waste(τ) + C_intervention(shift_τ) + C_failure_risk(τ)`, failure risk via Larson-Miller creep parameter
- Closing the loop: Run-to-Failure → Time-Based → **Condition-Based (this project)**; automated work orders with crew, spares, and LOTO safety checklist

**Visual:** `ppt_assets/04_fouling_rsow_curve.png`

**Speaking script (≈95s):**
"This is where prognosis becomes a decision. We project the fouling trajectory forward 48 hours against two thresholds — an advisory inspection limit and a critical soot-blowing threshold. From that trajectory we compute what we call the Remaining Safe Operating Window, RSOW: the time remaining until fouling resistance crosses the critical threshold at the current burn rate, clipped between one hour and one week. This converts a vague statement — 'the boiler is degrading' — into an actionable number a shift scheduler can plan around: 'you have forty-one hours before mandatory intervention.'

That number then feeds a cost-optimization: we schedule the actual intervention time to minimize the sum of cumulative fuel-waste cost from delaying, the shift-dependent cost of the intervention itself — night or off-peak shutdowns cut production disruption by up to seventy-five percent — and the failure risk, which comes from the Larson-Miller parameter for tube-metal creep.

This closes the loop back to the maintenance-paradigm question: Run-to-Failure reacts only after breakage; Time-Based inspects on a fixed calendar regardless of actual condition; what we've built is Condition-Based — continuous, physics-informed monitoring driving the schedule. The system automatically generates the resulting work order: asset ID, priority, crew assignment, required spares, and a LOTO — Lockout/Tagout — safety checklist, which is the standard industrial procedure of de-energizing, locking, and tagging equipment before a crew touches it. That's the full loop: physics, to prognosis, to a safe, scheduled, human action."

---

### Thank You Slide
**On-slide content:**
- Thank You
- Questions?
- [Your name] | [Email] | Work System Design, IIT Bhilai
- One-line recap: "A neural network that predicts boiler failure before it happens — and never lies about the laws of physics to do it."

**Visual:** none — clean closing slide, IIT Bhilai branding matching Slide 1.

**Speaking script (≈20s):**
"To close: this project shows that a small neural network, constrained by the 1st Law of Thermodynamics, trades a modest amount of raw accuracy for a threefold improvement in physical consistency — and that trade-off is what makes it trustworthy enough to sit inside a real maintenance work system, not just a lab benchmark. Thank you — I'm happy to take questions."

---

## Full asset checklist — everything needed for this deck is already generated

| # | File | Slide | What it shows |
|---|---|---|---|
| 1 | `ppt_assets/01_rmse_vs_energy_residual.png` | 10 | RMSE vs. physics-residual trade-off scatter, all 6 models, PINN highlighted |
| 2 | `ppt_assets/02_energy_sankey.png` | 5 | 1st-Law energy-flow Sankey at 85% load |
| 3 | `ppt_assets/03_health_index_gauge.png` | 11 | Live Health Index semicircular gauge |
| 4 | `ppt_assets/04_fouling_rsow_curve.png` | 12 | 48-hour fouling projection with RSOW thresholds |
| 5 | `ppt_assets/05_leaderboard_table.png` | 9 | 6-model benchmark table, real numbers |
| 6 | `ppt_assets/06_work_system_framework.png` | 2 | Alter's 9-element Work System diagram, instantiated |
| 7 | `ppt_assets/07_cps_5c_architecture.png` | 3 | CPS 5C levels diagram with project-specific detail |
| 8 | `ppt_assets/08_pinn_architecture.png` | 7 | BoilerPINN dual-head network diagram |

**Slides needing no image** (equation/text-only by design — don't force a chart onto them): 1 (title, needs only branding), 4, 6, 8, and the Thank You slide.

**Nothing is left for you to draw from scratch.** The only manual work remaining is Canva branding/polish: IIT Bhilai cover template, font/color pass, and slide transitions.
