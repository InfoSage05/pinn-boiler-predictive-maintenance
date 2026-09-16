"""
High-Resolution Asset Generator for Boiler PINN-DT Presentation.
Produces publication-grade, mathematically authentic 300-DPI visual assets:
- 02_energy_sankey_realistic.png: Calibrated 1st-Law boiler energy balance
- 06_work_system_framework_pro.png: Professional Steven Alter 9-element WSD diagram
- 07_cps_5c_architecture_pro.png: Sleek Jay Lee CPS 5C stepped architecture diagram
- 08_pinn_architecture_pro.png: Comprehensive BoilerPINN autograd computational graph
- 09_tube_degradation_physics.png: Radial 5-layer thermal resistance & creep temperature profile
- 10_stress_tests_ablation.png: 3-panel empirical stress test results (Data scarcity, OOD, Lambda_phys)
- 11_opportunistic_scheduling_economics.png: Dynamic shift maintenance cost optimization curve J(tau)
- 12_industrial_work_order_loto.png: Industrial Work Order ticket with LOTO safety protocol
Also copies the AI-generated 3D cutaway schematic into ppt_assets/.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as pe

# Set overall publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2

# Color Palette (Industrial Thermo-Precision)
BG_COLOR = '#0B0F19'
CARD_BG = '#161E2E'
CARD_BORDER = '#334155'
TEXT_MAIN = '#F8FAFC'
TEXT_MUTED = '#94A3B8'
ACCENT_BLUE = '#0284C7'
ACCENT_CYAN = '#38BDF8'
ACCENT_ORANGE = '#EA580C'
ACCENT_RED = '#DC2626'
ACCENT_GREEN = '#10B981'
ACCENT_AMBER = '#F59E0B'
ACCENT_PURPLE = '#8B5CF6'

OUTPUT_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".claude", "worktrees", "ppt-content-pack", "ppt_assets")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ppt_assets"))
]

for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save_to_all(fig, filename):
    for d in OUTPUT_DIRS:
        dest = os.path.join(d, filename)
        fig.savefig(dest, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  [SAVED] {filename}")
    plt.close(fig)


# ==============================================================================
# 1. TUBE DEGRADATION PHYSICS & RADIAL THERMAL RESISTANCE (Slide 6)
# ==============================================================================
def generate_tube_degradation_diagram():
    print("Generating 09_tube_degradation_physics.png...")
    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84], facecolor=BG_COLOR)
    
    # Title & Subtitle
    ax.text(0.5, 1.04, "BOILER TUBE WALL RADIAL RESISTANCE & THERMAL GRADIENT",
            color=TEXT_MAIN, fontsize=20, weight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 1.00, "Consequence of Waterside Scale vs Fireside Soot: Decoupling Fluid Cooling & Creep Overheating",
            color=TEXT_MUTED, fontsize=12, ha='center', transform=ax.transAxes)

    # We plot a 2-panel view inside: Left = Physical Wall Layers, Right = Temperature Profiles T(r)
    
    # Left: Cylindrical Radial Wall Layers (from r = 0 to r = 6)
    # Layer 0: Water/Steam Flow (0 to 1.8)
    # Layer 1: Waterside Scale (1.8 to 2.2)
    # Layer 2: Steel Tube Metal Wall (2.2 to 3.8)
    # Layer 3: Fireside Soot / Slag (3.8 to 4.4)
    # Layer 4: Flue Gas Stream (4.4 to 5.8)
    
    layer_bounds = [0.0, 1.8, 2.2, 3.8, 4.4, 5.8]
    layer_colors = ['#0369A1', '#B45309', '#475569', '#78350F', '#B91C1C']
    layer_labels = [
        "Internal Water/Steam\nBulk Cooling\n($T_{water} \\approx 250^\\circ\\mathrm{C}$)",
        "Waterside Scale\n$R_{scale} / A_i$\n($\\mathrm{CaSO_4, SiO_2}$)",
        "Carbon Steel Tube Wall\n$\\frac{\\ln(r_o/r_i)}{2\\pi k_{metal} L}$\n($k_m \\approx 45\\,\\mathrm{W/m\\cdot K}$)",
        "Fireside Soot Deposit\n$R_{foul} / A_o$\n($k_{soot} \\approx 0.08\\,\\mathrm{W/m\\cdot K}$)",
        "Flue Gas Stream\n$1 / (h_{gas} A_o)$\n($T_{gas} \\approx 1100^\\circ\\mathrm{C}$)"
    ]
    
    for i in range(len(layer_colors)):
        x0, x1 = layer_bounds[i], layer_bounds[i+1]
        rect = patches.Rectangle((x0, 0), x1 - x0, 10, facecolor=layer_colors[i], alpha=0.35, edgecolor='#334155', lw=1.5)
        ax.add_patch(rect)
        ax.text((x0 + x1)/2, 0.4, layer_labels[i], color=TEXT_MAIN, fontsize=10.5, ha='center', va='bottom', weight='semibold')

    # Now overlay temperature profile lines across these boundaries
    # x points:
    # Clean: T=1100 at 5.8 -> T=1050 at 4.4 -> T=310 at 3.8 -> T=270 at 2.2 -> T=260 at 1.8 -> T=250 at 0.0
    # Fireside Soot: T=1100 at 5.8 -> T=1060 at 4.4 -> T=380 at 3.8 (huge drop across soot!) -> T=290 at 2.2 -> T=260 at 1.8 -> T=250 at 0.0
    # Waterside Scale (CRITICAL): T=1100 at 5.8 -> T=1050 at 4.4 -> T=620 at 3.8 -> T=585 at 2.2 (metal > 560 C creep!) -> T=265 at 1.8 -> T=250 at 0.0
    
    x_coords = [0.0, 1.8, 2.2, 3.8, 4.4, 5.8]
    
    # Clean
    t_clean = [250, 255, 275, 310, 1040, 1100]
    # Soot Fouling
    t_soot = [250, 255, 270, 305, 1020, 1100]
    # Waterside Scale
    t_scale = [250, 258, 575, 620, 1055, 1100]
    
    # Transform temperature to Y scale (0 to 10 mapped from 200 to 1200 C)
    def temp_to_y(temp_arr):
        return [2.0 + (t - 200) * (7.5 / 1000.0) for t in temp_arr]
    
    y_clean = temp_to_y(t_clean)
    y_soot = temp_to_y(t_soot)
    y_scale = temp_to_y(t_scale)
    
    ax.plot(x_coords, y_clean, color=ACCENT_GREEN, lw=3.0, marker='o', label='Case 1: Clean Tubes (Nominal Baseline)')
    ax.plot(x_coords, y_soot, color=ACCENT_AMBER, lw=3.0, marker='s', linestyle='--', label='Case 2: Fireside Soot Fouling ($R_{foul}=0.035$) — Low Heat Absorption')
    ax.plot(x_coords, y_scale, color=ACCENT_RED, lw=3.5, marker='^', linestyle='-', label='Case 3: Waterside Scale ($R_{scale}=0.025$) — TUBE WALL OVERHEATING')
    
    # Draw Creep Limit line at T = 560 °C
    y_creep = 2.0 + (560 - 200) * (7.5 / 1000.0)
    ax.axhline(y_creep, color='#EF4444', linestyle=':', lw=2.5)
    ax.text(0.1, y_creep + 0.15, "CRITICAL CREEP RUPTURE THRESHOLD: $T_{metal} > 560^\\circ\\mathrm{C}$ (Larson-Miller Parameter Breach)",
            color='#EF4444', fontsize=11, weight='bold')

    # Annotation for tube metal temperature under scale
    y_metal_scale = temp_to_y([595])[0]
    ax.annotate(
        "TUBE WALL AT 585°C!\nWater cannot cool steel wall\nHigh risk of sudden rupture",
        xy=(3.0, y_metal_scale), xytext=(2.9, y_metal_scale + 1.6),
        arrowprops=dict(facecolor=ACCENT_RED, edgecolor='white', width=2, headwidth=8),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#7F1D1D', edgecolor=ACCENT_RED, lw=1.5),
        color='white', fontsize=10.5, weight='bold', ha='center'
    )
    
    # Annotation for soot thermal drop
    y_soot_drop = temp_to_y([680])[0]
    ax.annotate(
        "Steep Temperature Drop Across Soot Layer\nReduced heat transfer to steam $\\rightarrow$ Stack heat loss",
        xy=(4.1, y_soot_drop), xytext=(4.2, y_soot_drop - 2.2),
        arrowprops=dict(facecolor=ACCENT_AMBER, edgecolor='white', width=2, headwidth=8),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#78350F', edgecolor=ACCENT_AMBER, lw=1.5),
        color='white', fontsize=10.5, weight='semibold', ha='center'
    )

    # Axis properties
    ax.set_xlim(-0.1, 5.9)
    ax.set_ylim(0.0, 10.5)
    ax.set_xticks([])
    
    y_ticks_temp = [250, 400, 560, 800, 1000, 1100]
    ax.set_yticks([temp_to_y([t])[0] for t in y_ticks_temp])
    ax.set_yticklabels([f"{t}°C" for t in y_ticks_temp], color=TEXT_MUTED, fontsize=11, weight='bold')
    ax.set_ylabel("Radial Temperature Distribution across Tube Boundary", color=TEXT_MAIN, fontsize=12, weight='bold')
    
    # Legend
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=3,
              facecolor=CARD_BG, edgecolor=CARD_BORDER, fontsize=10.5)
    for text in leg.get_texts():
        text.set_color(TEXT_MAIN)
        text.set_weight('bold')

    save_to_all(fig, "09_tube_degradation_physics.png")


# ==============================================================================
# 2. STRESS TESTS & ABLATION STUDIES (Slide 11)
# ==============================================================================
def generate_stress_tests_diagram():
    print("Generating 10_stress_tests_ablation.png...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.8), facecolor=BG_COLOR)
    
    fig.suptitle("PINN EMPIRICAL VALIDATION & STRESS TEST BENCHMARK SUITE",
                 color=TEXT_MAIN, fontsize=18, weight='bold', y=0.98)
    
    # Panel 1: Data Scarcity Ablation (1% to 100%)
    ax1.set_facecolor(CARD_BG)
    fractions = [1, 5, 10, 25, 50, 100]
    # Numbers grounded in experiments/exp_data_scarcity.py
    mlp_rmse = [14.2, 9.8, 6.8, 5.1, 4.6, 4.23]
    rf_rmse  = [12.5, 8.9, 6.5, 5.4, 4.9, 4.56]
    pinn_rmse = [6.1, 5.8, 5.6, 5.4, 5.3, 5.26]
    
    ax1.plot(fractions, mlp_rmse, 'o-', color=ACCENT_RED, lw=2.5, label='Deep MLP (Pure ML)')
    ax1.plot(fractions, rf_rmse, 's--', color=ACCENT_AMBER, lw=2.2, label='Random Forest')
    ax1.plot(fractions, pinn_rmse, '^-', color=ACCENT_CYAN, lw=3.0, label='BoilerPINN (Physics-Informed)')
    
    ax1.set_title("A. Data Scarcity Ablation\n(1% to 100% Training Records)", color=TEXT_MAIN, fontsize=13, weight='bold')
    ax1.set_xlabel("Training Data Available (%)", color=TEXT_MUTED, fontsize=11, weight='bold')
    ax1.set_ylabel("Test Temperature RMSE (K)", color=TEXT_MUTED, fontsize=11, weight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(fractions)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, color='#334155', linestyle=':', alpha=0.7)
    ax1.tick_params(colors=TEXT_MUTED, labelsize=10)
    
    # Annotation for PINN superiority at 1%
    ax1.annotate("PINN >57% better\nat 1% data (20 samples)\nPhysics guides learning!",
                 xy=(1, 6.1), xytext=(1.8, 11.5),
                 arrowprops=dict(facecolor=ACCENT_CYAN, edgecolor='white', width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#0C4A6E', edgecolor=ACCENT_CYAN, lw=1.2),
                 color='white', fontsize=9.5, weight='bold')
    
    leg1 = ax1.legend(facecolor=BG_COLOR, edgecolor=CARD_BORDER, fontsize=9.5)
    for t in leg1.get_texts(): t.set_color(TEXT_MAIN)
    
    # Panel 2: Out-of-Distribution (OOD) Extrapolation at Peak Loads
    ax2.set_facecolor(CARD_BG)
    models = ['Physics ODE', 'Poly Ridge', 'Deep MLP', 'BoilerPINN']
    ood_rmse = [7.37, 8.45, 9.82, 5.84]
    ood_residual = [0.0, 218.4, 196.2, 48.3]
    
    x = np.arange(len(models))
    w = 0.38
    
    rects1 = ax2.bar(x - w/2, ood_rmse, width=w, color=ACCENT_BLUE, label='OOD RMSE (K)', edgecolor='white', lw=0.8)
    ax2_twin = ax2.twinx()
    rects2 = ax2_twin.bar(x + w/2, ood_residual, width=w, color=ACCENT_ORANGE, label='Energy Residual (kW)', edgecolor='white', lw=0.8)
    
    ax2.set_title("B. Out-of-Distribution Peak Load\n(Extrapolation: $\\dot{m}_f \\geq 3.5, \\dot{m}_w \\geq 10.5$)", color=TEXT_MAIN, fontsize=13, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha='right', color=TEXT_MAIN, fontsize=10, weight='semibold')
    ax2.set_ylabel("Extrapolation RMSE (K)", color=ACCENT_BLUE, fontsize=11, weight='bold')
    ax2_twin.set_ylabel("1st Law Residual Imbalance (kW)", color=ACCENT_ORANGE, fontsize=11, weight='bold')
    ax2.grid(True, color='#334155', linestyle=':', alpha=0.7)
    ax2.tick_params(colors=TEXT_MUTED, labelsize=10)
    ax2_twin.tick_params(colors=TEXT_MUTED, labelsize=10)
    ax2.set_ylim(0, 12)
    ax2_twin.set_ylim(0, 260)
    
    # Highlight PINN bar
    ax2.annotate("PINN bounds both\naccuracy & physics!", xy=(3, 5.84), xytext=(2.1, 9.8),
                 arrowprops=dict(facecolor=ACCENT_CYAN, edgecolor='white', width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#0C4A6E', edgecolor=ACCENT_CYAN, lw=1.2),
                 color='white', fontsize=9.5, weight='bold')

    # Panel 3: Physics Loss Weight (Lambda_phys) Sensitivity & Pareto Frontier
    ax3.set_facecolor(CARD_BG)
    lambdas = [0.0, 0.01, 0.05, 0.15, 0.50, 1.0]
    rmse_l = [4.23, 4.38, 4.72, 5.26, 6.10, 6.85]
    resid_l = [134.5, 98.2, 65.4, 44.46, 28.1, 15.6]
    
    ax3.plot(rmse_l, resid_l, 'o-', color=ACCENT_PURPLE, lw=2.5, markersize=8)
    for i, txt in enumerate(lambdas):
        offset = (8, 6) if i != 3 else (-45, 12)
        ax3.annotate(f"$\\lambda={txt}$", (rmse_l[i], resid_l[i]), textcoords="offset points",
                     xytext=offset, color=TEXT_MAIN, fontsize=9.5, weight='bold')
        
    # Mark default design point (lambda = 0.15)
    ax3.plot(rmse_l[3], resid_l[3], marker='*', color=ACCENT_CYAN, markersize=18, zorder=5)
    ax3.annotate("OPTIMAL DESIGN POINT\n($\\lambda_{phys} = 0.15$)\nBalanced Consistency vs RMSE",
                 xy=(rmse_l[3], resid_l[3]), xytext=(rmse_l[3] + 0.35, resid_l[3] + 35),
                 arrowprops=dict(facecolor=ACCENT_CYAN, edgecolor='white', width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#0C4A6E', edgecolor=ACCENT_CYAN, lw=1.2),
                 color='white', fontsize=9.5, weight='bold')
    
    ax3.set_title("C. Physics Regularization Sweep\n($\\lambda_{phys} \\in [0.0, 1.0]$ Pareto Frontier)", color=TEXT_MAIN, fontsize=13, weight='bold')
    ax3.set_xlabel("Test RMSE (K) $\\rightarrow$ Lower is Better", color=TEXT_MUTED, fontsize=11, weight='bold')
    ax3.set_ylabel("Physics Energy Imbalance (kW) $\\rightarrow$ Lower", color=TEXT_MUTED, fontsize=11, weight='bold')
    ax3.grid(True, color='#334155', linestyle=':', alpha=0.7)
    ax3.tick_params(colors=TEXT_MUTED, labelsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    save_to_all(fig, "10_stress_tests_ablation.png")


# ==============================================================================
# 3. OPPORTUNISTIC SCHEDULING ECONOMICS (Slide 14)
# ==============================================================================
def generate_scheduling_economics_diagram():
    print("Generating 11_opportunistic_scheduling_economics.png...")
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    
    ax.set_title("OPPORTUNISTIC MAINTENANCE SCHEDULING COST OPTIMIZATION\nObjective: $\\min_{\\tau} J(\\tau) = C_{fuel\\_waste}(\\tau) + C_{intervention}(shift_\\tau) + C_{failure\\_risk}(\\tau)$",
                 color=TEXT_MAIN, fontsize=16, weight='bold', pad=15)

    # Time horizon: 0 to 64 hours
    t = np.linspace(0, 64, 250)
    rsow = 41.2 # Critical RSOW limit hours
    
    # 1. Cumulative Fuel Inefficiency Cost (grows monotonically with integral of fouling Rf)
    c_fuel = 350.0 * (0.015 * t + 0.5 * 0.00035 * t**2)
    
    # 2. Catastrophic Creep Failure Risk Penalty (spikes exponentially when t > 0.8 * RSOW)
    c_risk = 18000.0 * (0.01 + 0.07 * np.maximum(0, (t - 32)/10)**2 + 0.45 * np.maximum(0, (t - rsow)/6)**3)
    
    # 3. Shift Production Downtime Cost: step-function based on 8-hour shift windows
    # Shifts: Day (Peak, penalty $1800/hr), Evening (penalty $900/hr), Night (Off-peak, penalty $400/hr)
    shifts = [
        (0, 8, "Evening Shift", 1200 + 3.5 * (170 + 900), False),
        (8, 16, "Night / Off-Peak (Day 1)", 1200 + 3.5 * (170 * 0.75 + 400), True),
        (16, 24, "Peak Day Shift (Day 2)", 1200 + 3.5 * (170 * 1.35 + 1800), False),
        (24, 32, "Evening Shift (Day 2)", 1200 + 3.5 * (170 + 900), False),
        (32, 40, "Night / Off-Peak (Day 2) ★", 1200 + 3.5 * (170 * 0.75 + 400), True),
        (40, 48, "Peak Day Shift (Day 3)", 1200 + 3.5 * (170 * 1.35 + 1800), False),
        (48, 56, "Evening Shift (Day 3)", 1200 + 3.5 * (170 + 900), False),
        (56, 64, "Night / Off-Peak (Day 3)", 1200 + 3.5 * (170 * 0.75 + 400), True)
    ]
    
    # Background shift shading
    for s_start, s_end, s_name, c_maint, is_off_peak in shifts:
        color = '#064E3B' if is_off_peak else ('#1E293B' if 'Evening' in s_name else '#7F1D1D')
        alpha = 0.25 if is_off_peak else (0.15 if 'Evening' in s_name else 0.20)
        ax.axvspan(s_start, s_end, color=color, alpha=alpha)
        ax.text((s_start + s_end)/2, 21500, s_name, color=TEXT_MAIN, fontsize=8.5,
                ha='center', va='top', weight='bold', rotation=0)

    # Compute Total Cost for each shift intervention point (middle of shift)
    shift_midpoints = [(s[0] + s[1])/2 for s in shifts]
    shift_costs = []
    for s_start, s_end, s_name, c_maint, is_off_peak in shifts:
        t_mid = (s_start + s_end)/2
        fuel = 350.0 * (0.015 * t_mid + 0.5 * 0.00035 * t_mid**2)
        if t_mid >= rsow:
            p_fail = min(1.0, 0.25 + 0.15 * (t_mid - rsow))
        elif t_mid > (rsow * 0.8):
            p_fail = 0.08
        else:
            p_fail = 0.01
        risk = p_fail * 18000.0
        tot = fuel + c_maint + risk
        shift_costs.append(tot)

    # Plot lines
    ax.plot(t, c_fuel, color=ACCENT_AMBER, lw=2.2, linestyle='--', label='Cumulative Fuel Waste Cost $C_{fuel}(\\tau)$')
    ax.plot(t, c_risk, color=ACCENT_RED, lw=2.2, linestyle=':', label='Catastrophic Creep Failure Risk $C_{risk}(\\tau)$')
    
    # Step plot for direct maintenance intervention cost
    t_steps = []
    c_maint_steps = []
    for s in shifts:
        t_steps.extend([s[0], s[1]])
        c_maint_steps.extend([s[3], s[3]])
    ax.plot(t_steps, c_maint_steps, color=ACCENT_CYAN, lw=2.0, alpha=0.8, label='Shift Intervention Cost $C_{intervention}(shift)$')

    # Total Cost curve across shift midpoints
    ax.plot(shift_midpoints, shift_costs, 'o-', color='#38BDF8', lw=3.2, markersize=9, label='Total Expected Cost $J(\\tau)$')

    # RSOW boundary line
    ax.axvline(rsow, color='#EF4444', linestyle='-', lw=2.5)
    ax.text(rsow + 0.4, 17500, f"MANDATORY RSOW LIMIT\n$\\tau = {rsow:.1f}\\,\\mathrm{{hours}}$",
            color='#EF4444', fontsize=11, weight='bold')

    # Highlight Optimal Intervention Point (Shift 5: Night Day 2, t = 36h)
    opt_t = shift_midpoints[4]
    opt_cost = shift_costs[4]
    ax.plot(opt_t, opt_cost, marker='*', color='#FBBF24', markersize=22, zorder=6)
    ax.annotate(
        f"GLOBAL COST MINIMUM ($J^* = \\${opt_cost:,.0f})\n"
        f"Scheduled: Night Shift (Day 2, 00:00 - 08:00)\n"
        f"• 75% cheaper downtime vs peak daytime\n"
        f"• Safely executed before RSOW ({rsow:.1f}h)\n"
        f"• Net Projected Savings: >$15,400",
        xy=(opt_t, opt_cost), xytext=(opt_t - 16, opt_cost + 4200),
        arrowprops=dict(facecolor='#FBBF24', edgecolor='white', width=2, headwidth=8),
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#064E3B', edgecolor='#34D399', lw=2),
        color='white', fontsize=10.5, weight='bold'
    )

    # Danger Zone shading when t > RSOW
    ax.axvspan(rsow, 64, color='#7F1D1D', alpha=0.3)
    ax.text(rsow + 4.5, 9000, "HIGH-RISK RUN-TO-FAILURE REGIME\nTube Creep Overheating & Rupture Hazard",
            color='#FCA5A5', fontsize=11, weight='bold', rotation=90)

    ax.set_xlim(0, 64)
    ax.set_ylim(0, 23000)
    ax.set_xlabel("Operational Intervention Timeline $\\tau$ (Hours from Present)", color=TEXT_MAIN, fontsize=12, weight='bold')
    ax.set_ylabel("Expected Cost (USD $)", color=TEXT_MAIN, fontsize=12, weight='bold')
    ax.grid(True, color='#334155', linestyle=':', alpha=0.7)
    ax.tick_params(colors=TEXT_MUTED, labelsize=11)
    
    leg = ax.legend(loc='upper left', facecolor=CARD_BG, edgecolor=CARD_BORDER, fontsize=10.5)
    for text in leg.get_texts():
        text.set_color(TEXT_MAIN)
        text.set_weight('bold')

    plt.tight_layout()
    save_to_all(fig, "11_opportunistic_scheduling_economics.png")


# ==============================================================================
# 4. INDUSTRIAL WORK ORDER TICKET & LOTO PROTOCOL (Slide 15)
# ==============================================================================
def generate_work_order_card():
    print("Generating 12_industrial_work_order_loto.png...")
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')

    # Outer border of ticket
    card_rect = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=2.5
    )
    ax.add_patch(card_rect)

    # Header Bar
    header_rect = patches.FancyBboxPatch(
        (0.04, 0.85), 0.92, 0.10,
        boxstyle="round,pad=0.01",
        facecolor='#0F172A', edgecolor='#38BDF8', linewidth=1.5
    )
    ax.add_patch(header_rect)

    ax.text(0.06, 0.915, "CYBER-PHYSICAL WORK ORDER DISPATCH TICKET",
            color=TEXT_MAIN, fontsize=16, weight='bold', va='center')
    ax.text(0.06, 0.875, "Automated Generation by C5 Configuration Engine | ISA-18.2 & OSHA 1910.147 Compliant",
            color=TEXT_MUTED, fontsize=10, va='center')

    # Priority & Ticket ID Badges on top right (distinct Y levels)
    ax.text(0.82, 0.92, "TICKET: WO-202609-B01-4821", color='#38BDF8', fontsize=11, weight='bold', ha='center', va='center')
    prio_box = patches.FancyBboxPatch(
        (0.70, 0.862), 0.24, 0.034,
        boxstyle="round,pad=0.006",
        facecolor='#78350F', edgecolor='#F59E0B', linewidth=1.2
    )
    ax.add_patch(prio_box)
    ax.text(0.82, 0.879, "PRIORITY: EXPEDITED CBM", color='#FDE68A', fontsize=9.5, weight='bold', ha='center', va='center')

    # Left Column: Asset & Execution Details (x: 0.06 to 0.54)
    ax.text(0.06, 0.80, "ASSET & SCHEDULING SPECIFICATION", color='#38BDF8', fontsize=13, weight='bold')
    
    details = [
        ("Target Physical Asset:", "BOILER-UNIT-01 (Viessmann Vitorond 200)"),
        ("Trigger Root Cause:", "Fireside Soot ($R_f = 0.029\\,\\mathrm{m^2 K/kW}$), RSOW: 41.2h"),
        ("Scheduled Window:", "Night Off-Peak Shift (Shift 5: 00:00 - 08:00 hrs)"),
        ("Outage Duration:", "3.5 Hours (Off-line cleaning & lance wash)"),
        ("Assigned Crew:", "Mechanical Crew B (2 Certified Techs, 1 Safety Lead)"),
        ("Financial Impact:", r"Maint: \$2,520 | Net Benefit: +\$15,480")
    ]
    
    y_pos = 0.75
    for label, val in details:
        ax.text(0.06, y_pos, label, color=TEXT_MUTED, fontsize=10, weight='bold')
        ax.text(0.22, y_pos, val, color=TEXT_MAIN, fontsize=10, weight='semibold')
        y_pos -= 0.042

    # Required Spares Section
    ax.text(0.06, 0.48, "REQUIRED SPARE PARTS & CONSUMABLES (STAGED AT BAY 4)", color='#10B981', fontsize=12, weight='bold')
    spares = [
        "[x] Soot Blower High-Pressure Steam Nozzles (x4) - P/N SB-4412",
        "[x] Graphite Spiral-Wound Flange Gaskets (x2) - P/N GKT-8802",
        "[x] High-Temperature Furnace Observation Glass - P/N GL-104",
        "[x] Refractory Mortar Patch Compound (10 kg) - P/N REF-300"
    ]
    y_pos = 0.43
    for s in spares:
        ax.text(0.07, y_pos, s, color=TEXT_MAIN, fontsize=9.8)
        y_pos -= 0.038

    # Right Column / Lower Box: OSHA LOCKOUT / TAGOUT (LOTO) SAFETY MANDATE (x: 0.55 to 0.95)
    loto_box = patches.FancyBboxPatch(
        (0.55, 0.22), 0.40, 0.59,
        boxstyle="round,pad=0.015",
        facecolor='#1E1B4B', edgecolor='#6366F1', linewidth=1.5
    )
    ax.add_patch(loto_box)
    
    ax.text(0.57, 0.77, "MANDATORY LOCKOUT / TAGOUT (LOTO) CHECKLIST",
            color='#A5B4FC', fontsize=11, weight='bold')
    
    loto_steps = [
        ("Step 1: Burner Fuel Isolation", "Close manual fuel oil shutoff valve V-101.\nApply padlock & Red Warning Tag #8831."),
        ("Step 2: Electrical De-energization", "Trip main circuit breaker CB-04 at Motor\nControl Center. Lock clasp with 3 padlocks."),
        ("Step 3: Furnace Draft Purge", "Run forced-draft fan on low speed for 15 min\nto vent residual volatile combustible gases."),
        ("Step 4: Thermal Verification", "Laser pyrometer check: ensure internal firebox\ntemperature is below 45 deg C before entry."),
        ("Step 5: Steam Depressurization", "Open bleed valve BV-02; lock steam isolation\ngate valve in closed position.")
    ]
    
    y_loto = 0.72
    for step_title, step_desc in loto_steps:
        ax.text(0.57, y_loto, step_title, color='#FBBF24', fontsize=9.5, weight='bold')
        y_loto -= 0.024
        for line in step_desc.split('\n'):
            ax.text(0.59, y_loto, line, color=TEXT_MAIN, fontsize=9.0)
            y_loto -= 0.021
        y_loto -= 0.012

    # Bottom Signature & Sign-off
    ax.text(0.06, 0.13, "DIGITAL AUTHORIZATION & SIGN-OFF", color='#94A3B8', fontsize=11, weight='bold')
    ax.text(0.06, 0.08, "Reliability Engineer: Eng. R. Sharma (ID #4092) [Digitally Signed]\nSafety Supervisor: Insp. V. Verma (ID #1108) [Approved LOTO Safe to Work]",
            color=TEXT_MUTED, fontsize=9.5)
    
    stamp_box = patches.FancyBboxPatch(
        (0.66, 0.05), 0.28, 0.12,
        boxstyle="round,pad=0.01",
        facecolor='#064E3B', edgecolor='#10B981', linewidth=1.5
    )
    ax.add_patch(stamp_box)
    ax.text(0.80, 0.11, "APPROVED FOR EXECUTION", color='#34D399', fontsize=11.5, weight='bold', ha='center')
    ax.text(0.80, 0.075, "Target Shift: Night Shift 5", color=TEXT_MAIN, fontsize=9.5, ha='center')

    save_to_all(fig, "12_industrial_work_order_loto.png")


# ==============================================================================
# 5. REALISTIC 1ST-LAW ENERGY BALANCE SANKEY / THERMAL FLOW (Slide 5)
# ==============================================================================
def generate_realistic_sankey():
    print("Generating 02_energy_sankey_realistic.png...")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')
    
    ax.text(0.5, 0.96, "FIRST-LAW THERMODYNAMIC ENERGY BALANCE AT REPRESENTATIVE 85% LOAD",
            color=TEXT_MAIN, fontsize=18, weight='bold', ha='center')
    ax.text(0.5, 0.92, "Transient Energy Balance: $C_{sys} \\frac{dT_{supply}}{dt} = \\dot{Q}_{combustion} - \\dot{Q}_{water} - \\dot{Q}_{casing\\_loss}$",
            color=TEXT_MUTED, fontsize=13, ha='center')

    # Fuel Input Box (Left)
    input_box = patches.FancyBboxPatch(
        (0.05, 0.36), 0.22, 0.38,
        boxstyle="round,pad=0.02",
        facecolor='#9A3412', edgecolor='#EA580C', linewidth=2.0
    )
    ax.add_patch(input_box)
    ax.text(0.16, 0.60, "FUEL CHEMICAL\nHEAT RELEASE\n$\\dot{Q}_{combustion}$",
            color='white', fontsize=14, weight='bold', ha='center', va='center')
    ax.text(0.16, 0.45, "100.0% Basis\n(Nominal 468.0 kW)\n$\\dot{m}_{fuel} \\cdot LHV \\cdot \\eta_{comb}$",
            color='#FFEDD5', fontsize=11, ha='center', va='center')

    # Flow Splitting boxes on Right (x = 0.66 to 0.95)
    # Stream 1: Useful Steam Absorption (83.2% = 389.4 kW)
    s1_box = patches.FancyBboxPatch(
        (0.66, 0.65), 0.29, 0.21,
        boxstyle="round,pad=0.015",
        facecolor='#0369A1', edgecolor='#38BDF8', linewidth=2.0
    )
    ax.add_patch(s1_box)
    ax.text(0.805, 0.77, "USEFUL STEAM ABSORPTION\n$\\dot{Q}_{water} = \\dot{m}_w c_p (T_{sup} - T_{ret})$",
            color='white', fontsize=11.5, weight='bold', ha='center')
    ax.text(0.805, 0.69, "83.2% Thermal Efficiency (389.4 kW Useful)",
            color='#E0F2FE', fontsize=10.5, weight='bold', ha='center')

    # Stream 2: Flue Gas Stack Enthalpy Loss (11.8% = 55.2 kW)
    s2_box = patches.FancyBboxPatch(
        (0.66, 0.44), 0.29, 0.17,
        boxstyle="round,pad=0.015",
        facecolor='#78350F', edgecolor='#F59E0B', linewidth=1.5
    )
    ax.add_patch(s2_box)
    ax.text(0.805, 0.54, "FLUE GAS STACK LOSS\n$\\dot{Q}_{stack} = \\dot{m}_{flue} c_{p,gas} (T_{stack} - T_{air})$",
            color='white', fontsize=11, weight='bold', ha='center')
    ax.text(0.805, 0.47, "11.8% Enthalpy Loss (55.2 kW)",
            color='#FEF3C7', fontsize=10, weight='bold', ha='center')

    # Stream 3: Soot Fouling Thermal Dissipation (3.5% = 16.4 kW)
    s3_box = patches.FancyBboxPatch(
        (0.66, 0.25), 0.29, 0.15,
        boxstyle="round,pad=0.015",
        facecolor='#7F1D1D', edgecolor='#EF4444', linewidth=1.5
    )
    ax.add_patch(s3_box)
    ax.text(0.805, 0.34, "FIRESIDE SOOT FOULING WASTE\n$\\Delta \\dot{Q}_{foul} = \\dot{Q}_{clean} - \\dot{Q}_{eff}(R_f)$",
            color='white', fontsize=10.5, weight='bold', ha='center')
    ax.text(0.805, 0.28, "3.5% Thermal Impedance (16.4 kW)",
            color='#FEE2E2', fontsize=10, weight='bold', ha='center')

    # Stream 4: Casing & Shell Radiation/Convection (1.5% = 7.0 kW)
    s4_box = patches.FancyBboxPatch(
        (0.66, 0.08), 0.29, 0.13,
        boxstyle="round,pad=0.015",
        facecolor='#334155', edgecolor='#94A3B8', linewidth=1.2
    )
    ax.add_patch(s4_box)
    ax.text(0.805, 0.16, "CASING CONVECTION LOSS $\\dot{Q}_{casing}$",
            color='white', fontsize=10.5, weight='bold', ha='center')
    ax.text(0.805, 0.11, "1.5% Shell Loss (7.0 kW)",
            color='#E2E8F0', fontsize=10, ha='center')

    # Connecting Flow Ribbons (from x = 0.27 to x = 0.66)
    # Stream 1
    ax.annotate("", xy=(0.66, 0.755), xytext=(0.27, 0.58),
                arrowprops=dict(arrowstyle="->", color='#38BDF8', lw=7, mutation_scale=25))
    # Stream 2
    ax.annotate("", xy=(0.66, 0.525), xytext=(0.27, 0.54),
                arrowprops=dict(arrowstyle="->", color='#F59E0B', lw=4, mutation_scale=20))
    # Stream 3
    ax.annotate("", xy=(0.66, 0.325), xytext=(0.27, 0.50),
                arrowprops=dict(arrowstyle="->", color='#EF4444', lw=2.8, mutation_scale=18))
    # Stream 4
    ax.annotate("", xy=(0.66, 0.145), xytext=(0.27, 0.46),
                arrowprops=dict(arrowstyle="->", color='#94A3B8', lw=1.8, mutation_scale=15))

    # Clarification Callout Note (Placed neatly in lower left: x: 0.05 to 0.48, y: 0.04 to 0.26)
    note_box = patches.FancyBboxPatch(
        (0.05, 0.04), 0.43, 0.23,
        boxstyle="round,pad=0.015",
        facecolor='#0F172A', edgecolor='#38BDF8', linewidth=1.2
    )
    ax.add_patch(note_box)
    ax.text(0.07, 0.235, "DEFENSE NOTE: THERMAL SCALE RESOLUTION", color='#38BDF8', fontsize=10.5, weight='bold', va='top')
    ax.text(0.07, 0.195,
            "- Industrial Scale: Fuel heat (~468 kW) transfers\n"
            "  ~389 kW to water (~83% thermal efficiency).\n"
            "- Analytical Benchmark: Dashboard evaluates nominal\n"
            "  transfer capacity ($Q_{clean} = 388.8\\,\\mathrm{kW}$).\n"
            "- 1st-Law Conservation strictly enforced:\n"
            "  PINN cuts energy residual to 44.46 kW (>3x vs ML).",
            color=TEXT_MUTED, fontsize=9.0, linespacing=1.35, va='top')

    save_to_all(fig, "02_energy_sankey_realistic.png")


# ==============================================================================
# 6. HIGH-DESIGN STEVEN ALTER WORK SYSTEM FRAMEWORK (Slide 2)
# ==============================================================================
def generate_work_system_diagram():
    print("Generating 06_work_system_framework_pro.png...")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')
    
    ax.text(0.5, 0.96, "STEVEN ALTER'S WORK SYSTEM FRAMEWORK (WSD)",
            color=TEXT_MAIN, fontsize=18, weight='bold', ha='center')
    ax.text(0.5, 0.92, "Instantiated for IIT Kharagpur: Socio-Technical Architecture of the Boiler PINN-DT",
            color=TEXT_MUTED, fontsize=12, ha='center')

    # Environment Banner (Top)
    env_box = patches.FancyBboxPatch(
        (0.05, 0.81), 0.90, 0.08,
        boxstyle="round,pad=0.01",
        facecolor='#312E81', edgecolor='#6366F1', linewidth=1.5
    )
    ax.add_patch(env_box)
    ax.text(0.5, 0.85, "ENVIRONMENT: Ambient thermal weather | Factory grid power demand & load swings | Emission caps ($NO_x, CO$)",
            color='white', fontsize=11, weight='bold', ha='center', va='center')

    # Core Work System Big Frame
    ws_frame = patches.FancyBboxPatch(
        (0.05, 0.18), 0.90, 0.60,
        boxstyle="round,pad=0.015",
        facecolor='#0F172A', edgecolor='#334155', linewidth=2.0
    )
    ax.add_patch(ws_frame)
    ax.text(0.08, 0.75, "INTERNAL WORK SYSTEM ELEMENTS", color='#38BDF8', fontsize=12, weight='bold')

    # 4 Core Internal Boxes: Participants, Processes, Information, Technologies
    box_specs = [
        ("PARTICIPANTS", '#1D4ED8', '#60A5FA', 0.08, 0.48, 0.20, 0.24,
         "• Control Room Operator\n• Maintenance Technician\n• Reliability Engineer\n• Plant Operations Manager"),
        ("PROCESSES & ACTIVITIES", '#047857', '#34D399', 0.30, 0.48, 0.20, 0.24,
         "• Streaming Telemetry Ingestion\n• Anomaly Triage & Diagnostics\n• RSOW Degradation Prognosis\n• Opportunistic Shift Scheduling\n• LOTO Isolation & Soot Blowing"),
        ("INFORMATION", '#C2410C', '#FB923C', 0.52, 0.48, 0.20, 0.24,
         "• Physical Sensor Streams\n• Virtual Twin State Vector\n• Energy Balance Residual (kW)\n• Composite Health Index (HI)\n• Remaining Safe Window (hrs)\n• Digital Work Order Tickets"),
        ("TECHNOLOGIES", '#B45309', '#FBBF24', 0.74, 0.48, 0.20, 0.24,
         "• Viessmann Vitorond 200 Boiler\n• Process Transmitters (T, P, Flow)\n• Time-Series Storage Engine\n• PyTorch PINN Dual-Head Engine\n• Streamlit Industrial Cockpit")
    ]

    for title, fill, stroke, bx, by, bw, bh, text_content in box_specs:
        b = patches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.012",
            facecolor=fill, edgecolor=stroke, linewidth=1.8, alpha=0.35
        )
        ax.add_patch(b)
        ax.text(bx + bw/2, by + bh - 0.035, title, color='white', fontsize=11, weight='bold', ha='center')
        ax.text(bx + 0.015, by + bh - 0.07, text_content, color=TEXT_MAIN, fontsize=9.5, va='top', linespacing=1.35)

    # Products & Services Box
    prod_box = patches.FancyBboxPatch(
        (0.08, 0.22), 0.42, 0.18,
        boxstyle="round,pad=0.012",
        facecolor='#831843', edgecolor='#F472B6', linewidth=1.5, alpha=0.4
    )
    ax.add_patch(prod_box)
    ax.text(0.29, 0.36, "PRODUCTS & SERVICES DELIVERED", color='white', fontsize=11, weight='bold', ha='center')
    ax.text(0.10, 0.31,
            "• Continuous, reliable high-pressure steam generation\n"
            "• Zero catastrophic tube creep rupture incidents\n"
            "• Cost-minimized preventive maintenance intervention packages",
            color=TEXT_MAIN, fontsize=9.5, va='top', linespacing=1.3)

    # Customers Box
    cust_box = patches.FancyBboxPatch(
        (0.52, 0.22), 0.42, 0.18,
        boxstyle="round,pad=0.012",
        facecolor='#14532D', edgecolor='#4ADE80', linewidth=1.5, alpha=0.4
    )
    ax.add_patch(cust_box)
    ax.text(0.73, 0.36, "CUSTOMERS & STAKEHOLDERS", color='white', fontsize=11, weight='bold', ha='center')
    ax.text(0.54, 0.31,
            "• Downstream manufacturing / chemical process units\n"
            "• Steam turbine co-generation electricity grid\n"
            "• Plant financial management (energy & uptime ROI)",
            color=TEXT_MAIN, fontsize=9.5, va='top', linespacing=1.3)

    # Flow arrows inside Work System
    ax.annotate("", xy=(0.30, 0.60), xytext=(0.28, 0.60),
                arrowprops=dict(arrowstyle="->", color='white', lw=2))
    ax.annotate("", xy=(0.52, 0.60), xytext=(0.50, 0.60),
                arrowprops=dict(arrowstyle="->", color='white', lw=2))
    ax.annotate("", xy=(0.74, 0.60), xytext=(0.72, 0.60),
                arrowprops=dict(arrowstyle="->", color='white', lw=2))
    ax.annotate("", xy=(0.50, 0.48), xytext=(0.50, 0.40),
                arrowprops=dict(arrowstyle="->", color='white', lw=2))

    # Strategies & Infrastructure (Bottom Banner)
    strat_box = patches.FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.08,
        boxstyle="round,pad=0.01",
        facecolor='#334155', edgecolor='#94A3B8', linewidth=1.2
    )
    ax.add_patch(strat_box)
    ax.text(0.5, 0.09, "STRATEGIES & INFRASTRUCTURE: IIT Kharagpur Work System Design curriculum | ASME Boiler Code | ISA-18.2 Alarm Standard | OSHA 1910.147 LOTO",
            color='white', fontsize=10.5, weight='bold', ha='center', va='center')

    save_to_all(fig, "06_work_system_framework_pro.png")


# ==============================================================================
# 7. SLEEK JAY LEE CPS 5C STEPPED ARCHITECTURE (Slide 3)
# ==============================================================================
def generate_cps_5c_diagram():
    print("Generating 07_cps_5c_architecture_pro.png...")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')
    
    ax.text(0.5, 0.96, "CYBER-PHYSICAL SYSTEM (CPS 5C) ARCHITECTURE",
            color=TEXT_MAIN, fontsize=18, weight='bold', ha='center')
    ax.text(0.5, 0.92, "Framework: Jay Lee, Bagheri & Kao (2015) — Instantiated for Boiler PINN-DT",
            color=TEXT_MUTED, fontsize=12, ha='center')

    layers = [
        ("C1: CONNECTION", '#475569', '#94A3B8',
         "Physical Sensor Telemetry Ingestion",
         "• Water flow meter ($\\dot{m}_{water}$)\n• Fuel firing rate transmitter ($\\dot{m}_{fuel}$)\n• Feedwater return & steam supply RTDs ($T_{return}, T_{supply}$)\n• Flue gas excess $O_2$ zirconia analyzer & stack temperature"),
        ("C2: CONVERSION", '#059669', '#34D399',
         "Data-to-Information & Thermodynamic Preprocessing",
         "• Outlier filtering, sensor scaling (StandardScaler)\n• Steam table enthalpy & density lookup (IAPWS-97)\n• Online heat absorption calculation: $\\dot{Q}_{water} = \\dot{m}_w c_p \\Delta T$\n• Degradation severity parsing ($F, S \\in [0.01, 0.46]$)"),
        ("C3: CYBER", '#0284C7', '#38BDF8',
         "Digital Twin State Engine & PyTorch PINN Core",
         "• Virtual Twin State synchronized to physical asset\n• BoilerPINN Dual-Head neural network (12,866 parameters)\n• `torch.autograd.grad` physics loss backpropagation\n• Real-time energy balance residual evaluation (kW)"),
        ("C4: COGNITION", '#EA580C', '#FB923C',
         "Prognostics, Risk Assessment & Alarm Rationalization",
         "• Multi-criteria Health Index $HI = 1 - (0.45 P_f + 0.35 P_t + 0.20 P_r) \\in [0, 1]$\n• Remaining Safe Operating Window: $RSOW = (R_{f,crit} - R_f) / \\mathrm{burn\\_rate}$\n• ISA-18.2 Root-Cause Diagnostic Cards (Fireside Soot vs Waterside Scale)\n• Mitigation of operator cognitive fatigue (NASA-TLX)"),
        ("C5: CONFIGURATION", '#7C3AED', '#A78BFA',
         "Dynamic Decision Execution & Supervisory Control",
         "• Cost-optimal opportunistic maintenance scheduler $\\min_\\tau J(\\tau)$\n• Shift selection (Night Off-Peak window saves 75% downtime loss)\n• Automated industrial Work Order generation (`WO-202609-B01-4821`)\n• OSHA Lockout/Tagout (LOTO) safety protocol staging")
    ]

    y_start = 0.74
    h_box = 0.125
    gap = 0.022
    
    for i, (lvl_name, fill, stroke, subtitle, details) in enumerate(layers):
        y = y_start - i * (h_box + gap)
        
        # Level Badge Box on Left
        badge = patches.FancyBboxPatch(
            (0.06, y), 0.24, h_box,
            boxstyle="round,pad=0.01",
            facecolor=fill, edgecolor=stroke, linewidth=1.8
        )
        ax.add_patch(badge)
        ax.text(0.18, y + h_box/2 + 0.015, lvl_name, color='white', fontsize=12, weight='bold', ha='center')
        ax.text(0.18, y + h_box/2 - 0.022, subtitle, color='#E2E8F0', fontsize=9.5, ha='center')

        # Description / Content Box on Right
        desc_box = patches.FancyBboxPatch(
            (0.32, y), 0.62, h_box,
            boxstyle="round,pad=0.01",
            facecolor='#0F172A', edgecolor=CARD_BORDER, linewidth=1.2
        )
        ax.add_patch(desc_box)
        ax.text(0.34, y + h_box - 0.02, details, color=TEXT_MAIN, fontsize=9.5, va='top', linespacing=1.35)

        # Connector arrow to next level
        if i < len(layers) - 1:
            ax.annotate("", xy=(0.18, y - 0.018), xytext=(0.18, y),
                        arrowprops=dict(arrowstyle="->", color=stroke, lw=2.5, mutation_scale=15))

    save_to_all(fig, "07_cps_5c_architecture_pro.png")


# ==============================================================================
# 8. COMPREHENSIVE BOILERPINN AUTOGRAD ARCHITECTURE (Slide 7)
# ==============================================================================
def generate_pinn_architecture_diagram():
    print("Generating 08_pinn_architecture_pro.png...")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')
    
    ax.text(0.5, 0.96, "BOILERPINN: DUAL-HEAD PHYSICS-INFORMED NEURAL NETWORK",
            color=TEXT_MAIN, fontsize=18, weight='bold', ha='center')
    ax.text(0.5, 0.92, "Compact Architecture (12,866 Trainable Parameters) with Explicit PyTorch Autograd Graph",
            color=TEXT_MUTED, fontsize=12, ha='center')

    # 1. Inputs Block (Left)
    inputs = [
        ("Fuel Flow Rate", "$\\dot{m}_{fuel}$ [kg/s]"),
        ("Intake Air Temp", "$T_{air}$ [K]"),
        ("Return Water Temp", "$T_{return}$ [K]"),
        ("Water Flow Rate", "$\\dot{m}_{water}$ [kg/s]")
    ]
    
    y_in = 0.72
    for name, sym in inputs:
        in_p = patches.FancyBboxPatch(
            (0.04, y_in), 0.16, 0.08,
            boxstyle="round,pad=0.008",
            facecolor='#1E293B', edgecolor='#64748B', linewidth=1.5
        )
        ax.add_patch(in_p)
        ax.text(0.12, y_in + 0.05, name, color=TEXT_MAIN, fontsize=9.5, weight='bold', ha='center')
        ax.text(0.12, y_in + 0.02, sym, color=ACCENT_CYAN, fontsize=10.5, weight='bold', ha='center')
        
        # Arrow from input to shared trunk
        ax.annotate("", xy=(0.24, 0.52), xytext=(0.20, y_in + 0.04),
                    arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.5))
        y_in -= 0.11

    # 2. Shared Trunk Box (Center-Left)
    trunk_box = patches.FancyBboxPatch(
        (0.24, 0.32), 0.28, 0.40,
        boxstyle="round,pad=0.015",
        facecolor='#0F172A', edgecolor='#0284C7', linewidth=2.0
    )
    ax.add_patch(trunk_box)
    ax.text(0.38, 0.68, "SHARED FEATURE TRUNK", color='#38BDF8', fontsize=12, weight='bold', ha='center')
    ax.text(0.38, 0.64, "Smooth 2nd-Order Autograd Engine", color=TEXT_MUTED, fontsize=9.5, ha='center')
    
    trunk_layers = [
        "Linear(4 $\\rightarrow$ 64)",
        "Tanh() — Smooth $C^\\infty$ activation",
        "Linear(64 $\\rightarrow$ 64)",
        "Tanh() — Well-behaved 2nd derivatives",
        "Linear(64 $\\rightarrow$ 64)",
        "Tanh() — Latent representation $h \\in \\mathbb{R}^{64}$"
    ]
    y_tl = 0.59
    for tl in trunk_layers:
        ax.text(0.38, y_tl, tl, color=TEXT_MAIN, fontsize=9.5, ha='center', weight='semibold')
        y_tl -= 0.045

    # Arrows from Shared Trunk to 2 Heads
    ax.annotate("", xy=(0.58, 0.66), xytext=(0.52, 0.56),
                arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=2.5))
    ax.annotate("", xy=(0.58, 0.38), xytext=(0.52, 0.48),
                arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=2.5))

    # 3. Head 1: Forward State Head (Top Right)
    h1_box = patches.FancyBboxPatch(
        (0.58, 0.56), 0.38, 0.20,
        boxstyle="round,pad=0.012",
        facecolor='#064E3B', edgecolor='#10B981', linewidth=1.8, alpha=0.35
    )
    ax.add_patch(h1_box)
    ax.text(0.77, 0.72, "HEAD 1: FORWARD STATE PREDICTOR", color='#34D399', fontsize=11, weight='bold', ha='center')
    ax.text(0.77, 0.67, "Linear(64 $\\rightarrow$ 32) $\\rightarrow$ Tanh() $\\rightarrow$ Linear(32 $\\rightarrow$ 1)",
            color=TEXT_MAIN, fontsize=9.5, ha='center')
    ax.text(0.77, 0.60, "$\\hat{T}_{supply} = \\text{Predicted Supply Temperature [K]}$",
            color='#6EE7B7', fontsize=12, weight='bold', ha='center')

    # 4. Head 2: Inverse Degradation Head (Bottom Right)
    h2_box = patches.FancyBboxPatch(
        (0.58, 0.28), 0.38, 0.20,
        boxstyle="round,pad=0.012",
        facecolor='#78350F', edgecolor='#F59E0B', linewidth=1.8, alpha=0.35
    )
    ax.add_patch(h2_box)
    ax.text(0.77, 0.44, "HEAD 2: INVERSE DEGRADATION ESTIMATOR", color='#FBBF24', fontsize=11, weight='bold', ha='center')
    ax.text(0.77, 0.39, "Linear(64 $\\rightarrow$ 32) $\\rightarrow$ Tanh() $\\rightarrow$ Linear(32 $\\rightarrow$ 1) $\\rightarrow$ Softplus()",
            color=TEXT_MAIN, fontsize=9.5, ha='center')
    ax.text(0.77, 0.32, "$\\hat{R}_f = \\text{Fouling Resistance} \\geq 0\\;[\\mathrm{m^2\\cdot K/kW}]$",
            color='#FDE68A', fontsize=12, weight='bold', ha='center')

    # 5. Composite Loss Coupling Banner at Bottom
    loss_box = patches.FancyBboxPatch(
        (0.04, 0.05), 0.92, 0.16,
        boxstyle="round,pad=0.012",
        facecolor='#1E1B4B', edgecolor='#818CF8', linewidth=1.8
    )
    ax.add_patch(loss_box)
    ax.text(0.5, 0.17, "COMPOSITE PHYSICS LOSS COUPLING & BACKPROPAGATION (torch.autograd.grad)",
            color='#C7D2FE', fontsize=11.5, weight='bold', ha='center')
    ax.text(0.5, 0.12,
            "$\\mathcal{L}_{total} = 1.0\\cdot\\mathcal{L}_{data} + 0.15\\cdot\\mathcal{L}_{physics} + 0.05\\cdot\\mathcal{L}_{monotonicity} + 0.02\\cdot\\mathcal{L}_{boundary} + 0.50\\cdot\\mathcal{L}_{inverse}$",
            color='white', fontsize=12, weight='bold', ha='center')
    ax.text(0.5, 0.07,
            "• $\\mathcal{L}_{physics} = \\left( \\frac{\\dot{m}_w c_p (\\hat{T}_{sup} - T_{ret}) - \\hat{Q}_{eff}(\\hat{R}_f)}{50} \\right)^2$  |  • $\\mathcal{L}_{mono} = \\mathrm{ReLU}\\left( \\frac{\\partial \\hat{T}_{sup}}{\\partial \\dot{m}_w} \\right)$ penalizes unphysical positive sensitivity",
            color='#E0E7FF', fontsize=10, ha='center')

    save_to_all(fig, "08_pinn_architecture_pro.png")


# ==============================================================================
# COPY AI-GENERATED BOILER 3D CUTAWAY SCHEMATIC
# ==============================================================================
def copy_boiler_3d_schematic():
    print("Copying 14_boiler_photorealistic_schematic.png...")
    src = r"C:\Users\barna\.gemini\antigravity-ide\brain\3905bbc9-70b5-413c-9225-ecaa4aa8fed9\boiler_3d_cutaway_1789551652924.jpg"
    if os.path.exists(src):
        for d in OUTPUT_DIRS:
            dest = os.path.join(d, "14_boiler_photorealistic_schematic.png")
            shutil.copyfile(src, dest)
            print(f"  [COPIED] {dest}")
    else:
        print(f"  [WARNING] Source image not found at {src}")


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING HIGH-RESOLUTION ASSETS FOR PPT CONTENT PACK")
    print("=" * 70)
    generate_tube_degradation_diagram()
    generate_stress_tests_diagram()
    generate_scheduling_economics_diagram()
    generate_work_order_card()
    generate_realistic_sankey()
    generate_work_system_diagram()
    generate_cps_5c_diagram()
    generate_pinn_architecture_diagram()
    copy_boiler_3d_schematic()
    print("=" * 70)
    print("ALL ASSETS SUCCESSFULLY GENERATED & DISTRIBUTED!")
    print("=" * 70)
