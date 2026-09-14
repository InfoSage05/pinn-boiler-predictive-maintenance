"""
Composite Health Index (HI) and Prognostics Module.
Combines thermodynamic residuals, latent fouling resistance,
and metallurgical thermal stress into a normalized Health Index in [0, 1].
Calculates Remaining Safe Operating Window (RSOW).
"""

from typing import Dict, Any
import numpy as np


class BoilerHealthPrognostics:
    def __init__(
        self,
        critical_rf: float = 0.035,
        max_tube_metal_temp_c: float = 560.0,
        baseline_tube_metal_temp_c: float = 520.0,
        residual_threshold_kw: float = 40.0,
        w_fouling: float = 0.45,
        w_thermal: float = 0.35,
        w_residual: float = 0.20
    ):
        self.critical_rf = critical_rf
        self.max_tube_metal_temp_c = max_tube_metal_temp_c
        self.baseline_tube_metal_temp_c = baseline_tube_metal_temp_c
        self.residual_threshold_kw = residual_threshold_kw
        self.w_fouling = w_fouling
        self.w_thermal = w_thermal
        self.w_residual = w_residual

    def compute_health_index(
        self,
        r_fouling: float,
        tube_metal_temp_c: float,
        physics_residual_kw: float
    ) -> Dict[str, Any]:
        """
        Computes composite Health Index (HI) in [0, 1] and sub-penalties.
        """
        # Sub-penalty 1: Fouling degradation ratio
        p_foul = float(np.clip(r_fouling / self.critical_rf, 0.0, 1.5))
        
        # Sub-penalty 2: Thermal tube overheating stress
        temp_excess = max(0.0, tube_metal_temp_c - self.baseline_tube_metal_temp_c)
        temp_span = self.max_tube_metal_temp_c - self.baseline_tube_metal_temp_c
        p_thermal = float(np.clip(temp_excess / temp_span, 0.0, 2.0))
        
        # Sub-penalty 3: Thermodynamic energy balance violation
        p_residual = float(np.clip(physics_residual_kw / self.residual_threshold_kw, 0.0, 2.0))
        
        # Total composite Health Index (1.0 = pristine brand-new asset, 0.0 = failure)
        hi = 1.0 - (self.w_fouling * p_foul + self.w_thermal * p_thermal + self.w_residual * p_residual)
        hi = float(np.clip(hi, 0.02, 1.0))
        
        # Categorical Health State
        if hi >= 0.75 and tube_metal_temp_c < (self.max_tube_metal_temp_c - 12.0):
            state = "HEALTHY"
            recommended_action = "Continue normal automated operation."
        elif hi >= 0.50:
            state = "ADVISORY"
            recommended_action = "Monitor soot fouling accumulation. Plan routine soot blowing."
        elif hi >= 0.30:
            state = "WARNING"
            recommended_action = "Elevated thermal stress. Schedule maintenance within 24-48 hours."
        else:
            state = "CRITICAL"
            recommended_action = "Creep rupture hazard! Derate load immediately and dispatch maintenance crew."
            
        return {
            "health_index": hi,
            "state": state,
            "penalty_fouling": p_foul,
            "penalty_thermal": p_thermal,
            "penalty_residual": p_residual,
            "recommended_action": recommended_action
        }

    def compute_remaining_safe_operating_window(
        self,
        current_rf: float,
        current_burn_rate_per_hour: float = 0.00032
    ) -> float:
        """
        Calculates Remaining Safe Operating Window (RSOW) in operating hours
        before critical fouling limit is breached.
        """
        if current_rf >= self.critical_rf:
            return 1.0 # Emergency shutdown / immediate intervention needed
            
        rsow = (self.critical_rf - current_rf) / max(current_burn_rate_per_hour, 1e-6)
        return float(np.clip(rsow, 1.0, 168.0)) # max 7 days
