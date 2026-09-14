"""
ISA-18.2 Compliant Anomaly Detection & Root-Cause Isolation Module.
Classifies operational alarms (Normal, Advisory, Warning, Critical)
and isolates physical root causes using thermodynamic physics signatures.
Prevents operator alarm fatigue through physics-grounded cognitive explanations.
"""

from typing import Dict, Any


class BoilerAnomalyDetector:
    def __init__(
        self,
        crit_tube_temp_c: float = 560.0,
        warn_tube_temp_c: float = 548.0,
        crit_fouling_rf: float = 0.035,
        warn_fouling_rf: float = 0.022,
        residual_limit_kw: float = 35.0
    ):
        self.crit_tube_temp_c = crit_tube_temp_c
        self.warn_tube_temp_c = warn_tube_temp_c
        self.crit_fouling_rf = crit_fouling_rf
        self.warn_fouling_rf = warn_fouling_rf
        self.residual_limit_kw = residual_limit_kw

    def evaluate_alarm(
        self,
        tube_metal_temp_c: float,
        estimated_rf: float,
        physics_residual_kw: float,
        flue_gas_o2_pct: float = 3.5,
        water_flow_kg_s: float = 7.75
    ) -> Dict[str, Any]:
        """
        Evaluates ISA-18.2 alarm priority and diagnoses physical root cause.
        """
        is_temp_crit = tube_metal_temp_c >= self.crit_tube_temp_c
        is_temp_warn = tube_metal_temp_c >= self.warn_tube_temp_c
        is_foul_crit = estimated_rf >= self.crit_fouling_rf
        is_foul_warn = estimated_rf >= self.warn_fouling_rf
        is_residual_high = physics_residual_kw >= self.residual_limit_kw
        
        # 1. Determine Alarm Priority (ISA-18.2 Standard)
        if is_temp_crit or is_foul_crit:
            alarm_level = "CRITICAL"
            color = "#E63946" # Vibrant Red
            priority = 1
        elif is_temp_warn or is_foul_warn or is_residual_high:
            alarm_level = "WARNING"
            color = "#F4A261" # Amber / Orange
            priority = 2
        elif estimated_rf >= (self.warn_fouling_rf * 0.70) or physics_residual_kw >= (self.residual_limit_kw * 0.60):
            alarm_level = "ADVISORY"
            color = "#2A9D8F" # Teal
            priority = 3
        else:
            alarm_level = "NORMAL"
            color = "#52B788" # Green
            priority = 4
            
        # 2. Physics-Based Root-Cause Isolation
        root_cause = "Nominal Operation"
        explanation = "All thermal parameters, heat transfer rates, and energy balances within design tolerances."
        action = "Maintain standard supervisory control."
        
        if is_foul_crit or is_foul_warn:
            root_cause = "Fireside Soot & Slag Deposition"
            explanation = (
                f"Estimated thermal fouling resistance Rf={estimated_rf:.4f} m²·K/kW exceeds baseline by "
                f"{(estimated_rf / 0.005):.1f}x. Impedes heat transfer from combustion gas to boiler tubes, "
                f"elevating stack losses and forcing excessive burner firing."
            )
            action = "Trigger automated soot blowers or schedule manual waterwall lance cleaning."
            
        elif is_temp_crit or is_temp_warn:
            if estimated_rf < self.warn_fouling_rf:
                root_cause = "Waterside Scale Deposition / Localized Hotspot"
                explanation = (
                    f"Tube metal temperature reached {tube_metal_temp_c:.1f}°C with moderate fireside fouling. "
                    f"Signature indicates waterside calcium/silica scale insulating tubes from internal water cooling, "
                    f"inducing creep-rupture hazard."
                )
                action = "Inspect feedwater chemical softening; prepare for chemical acid descaling."
            else:
                root_cause = "Combined Thermal Stress (Fouling + Elevated Metal Temp)"
                explanation = f"High thermal resistance compounded by elevated tube temperature ({tube_metal_temp_c:.1f}°C)."
                action = "Derate boiler firing rate by 15% immediately to pull tube metal out of creep regime."
                
        elif is_residual_high:
            root_cause = "Thermodynamic Mass/Energy Imbalance (Sensor Drift or Combustion Anomaly)"
            explanation = (
                f"Physics conservation residual ΔQ={physics_residual_kw:.1f} kW exceeds limit. "
                f"Indicates unmetered air leakage (air infiltration) or flowmeter calibration drift."
            )
            action = "Perform flue gas oxygen cross-check and recalibrate feedwater orifice flow transmitter."
            
        return {
            "alarm_level": alarm_level,
            "priority": priority,
            "badge_color": color,
            "root_cause": root_cause,
            "explanation": explanation,
            "recommended_action": action,
            "is_anomalous": (alarm_level != "NORMAL")
        }
