"""
Dynamic Opportunistic Maintenance Scheduler & Work Order Engine.
Implements cost-optimal maintenance scheduling based on predicted Remaining Safe Operating Window (RSOW).
Optimizes intervention timing across plant operating shifts to minimize downtime cost and fuel waste.
Generates industrial-grade Work Order tickets for maintenance crews.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np


@dataclass
class MaintenanceShift:
    shift_name: str
    start_hour_offset: float
    duration_hours: float
    labor_rate_multiplier: float
    plant_downtime_cost_per_hour_usd: float
    is_off_peak: bool


@dataclass
class WorkOrder:
    work_order_id: str
    asset_id: str
    generated_at: str
    priority: str # ROUTINE, EXPEDITED, EMERGENCY
    scheduled_start: str
    scheduled_end: str
    task_description: str
    assigned_crew: str
    required_spares: List[str]
    safety_protocols: List[str]
    estimated_duration_hours: float
    projected_maintenance_cost_usd: float
    projected_net_savings_usd: float


class OpportunisticMaintenanceScheduler:
    def __init__(
        self,
        base_labor_cost_per_hour: float = 85.0,
        unplanned_failure_cost: float = 18000.0,
        routine_soot_clean_cost: float = 1200.0,
        hourly_fuel_waste_per_rf: float = 350.0
    ):
        self.base_labor_cost = base_labor_cost_per_hour
        self.unplanned_failure_cost = unplanned_failure_cost
        self.routine_clean_cost = routine_soot_clean_cost
        self.hourly_waste_rf = hourly_fuel_waste_per_rf

    def generate_upcoming_shifts(self, current_time: datetime, horizon_hours: float = 72.0) -> List[MaintenanceShift]:
        """
        Builds schedule of upcoming operating shifts (Day, Evening, Night/Off-peak).
        """
        shifts = []
        n_shifts = int(horizon_hours / 8.0)
        
        for i in range(n_shifts):
            shift_offset = i * 8.0
            shift_time = current_time + timedelta(hours=shift_offset)
            hour_of_day = shift_time.hour
            is_weekend = shift_time.weekday() >= 5
            
            if 0 <= hour_of_day < 8 or is_weekend:
                name = f"Night / Off-Peak Window (Day {i//3 + 1}, Shift {i%3 + 1})"
                mult = 0.75 # Cheaper production penalty during off-peak
                downtime_penalty = 400.0 # $/hr
                off_peak = True
            elif 8 <= hour_of_day < 16:
                name = f"Peak Day Production (Day {i//3 + 1}, Shift {i%3 + 1})"
                mult = 1.35
                downtime_penalty = 1800.0 # High penalty for interrupting daytime production
                off_peak = False
            else:
                name = f"Evening Shift (Day {i//3 + 1}, Shift {i%3 + 1})"
                mult = 1.0
                downtime_penalty = 900.0
                off_peak = False
                
            shifts.append(MaintenanceShift(
                shift_name=name,
                start_hour_offset=shift_offset,
                duration_hours=8.0,
                labor_rate_multiplier=mult,
                plant_downtime_cost_per_hour_usd=downtime_penalty,
                is_off_peak=off_peak
            ))
        return shifts

    def optimize_maintenance_schedule(
        self,
        current_rf: float,
        rsow_hours: float,
        current_time: datetime = None
    ) -> Dict[str, Any]:
        """
        Evaluates candidate maintenance slots within RSOW and identifies
        the global cost-minimizing intervention window.
        """
        if current_time is None:
            current_time = datetime.now()
            
        shifts = self.generate_upcoming_shifts(current_time, horizon_hours=max(rsow_hours + 12.0, 48.0))
        
        shift_evaluations = []
        best_shift = None
        min_total_cost = float("inf")
        
        # Burn rate of fouling per hour
        rf_rate = 0.00035
        
        for shift in shifts:
            t_start = shift.start_hour_offset
            
            # Cumulative fuel waste cost from now until maintenance occurs:
            # integral of hourly fuel waste caused by fouling
            fuel_waste_total = sum(
                (current_rf + rf_rate * h) * self.hourly_waste_rf
                for h in np.arange(0, t_start, 1.0)
            )
            
            # Maintenance execution cost during this shift (cleaning duration ~ 3.5 hrs)
            clean_duration = 3.5
            labor_cost = clean_duration * self.base_labor_cost * shift.labor_rate_multiplier * 2.0 # 2 technicians
            downtime_cost = clean_duration * shift.plant_downtime_cost_per_hour_usd
            direct_maint_cost = self.routine_clean_cost + labor_cost + downtime_cost
            
            # Catastrophic failure risk penalty if scheduled beyond or close to RSOW
            if t_start >= rsow_hours:
                # High risk of tube rupture or emergency shutdown
                p_fail = min(1.0, 0.25 + 0.15 * (t_start - rsow_hours))
            elif t_start > (rsow_hours * 0.80):
                p_fail = 0.08
            else:
                p_fail = 0.01
                
            risk_penalty = p_fail * self.unplanned_failure_cost
            total_cost = fuel_waste_total + direct_maint_cost + risk_penalty
            
            record = {
                "shift_name": shift.shift_name,
                "start_hour_offset": t_start,
                "start_time": (current_time + timedelta(hours=t_start)).strftime("%Y-%m-%d %H:%M"),
                "is_within_rsow": (t_start <= rsow_hours),
                "fuel_waste_cost": round(fuel_waste_total, 2),
                "maintenance_cost": round(direct_maint_cost, 2),
                "failure_risk_penalty": round(risk_penalty, 2),
                "total_cost": round(total_cost, 2),
                "is_off_peak": shift.is_off_peak
            }
            shift_evaluations.append(record)
            
            # Select optimal shift strictly within RSOW (or earliest if RSOW is very short)
            if t_start <= rsow_hours and total_cost < min_total_cost:
                min_total_cost = total_cost
                best_shift = record
                
        # If no shift within RSOW, select earliest shift
        if best_shift is None and shift_evaluations:
            best_shift = shift_evaluations[0]
            
        # Generate official Work Order
        wo = self._build_work_order(best_shift, current_rf, current_time)
        
        return {
            "optimal_shift": best_shift,
            "all_shifts": shift_evaluations,
            "work_order": wo,
            "rsow_hours": rsow_hours
        }

    def _build_work_order(self, shift_record: dict, current_rf: float, current_time: datetime) -> WorkOrder:
        """Constructs an executable industrial maintenance work order."""
        start_dt = datetime.strptime(shift_record["start_time"], "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(hours=3.5)
        
        wo_id = f"WO-{current_time.strftime('%Y%m')}-B01-{np.random.randint(1000, 9999)}"
        
        if shift_record["start_hour_offset"] <= 4.0:
            priority = "EMERGENCY"
        elif shift_record["start_hour_offset"] <= 24.0:
            priority = "EXPEDITED"
        else:
            priority = "ROUTINE CBM"
            
        net_savings = max(0.0, self.unplanned_failure_cost - shift_record["total_cost"])
        
        return WorkOrder(
            work_order_id=wo_id,
            asset_id="BOILER-UNIT-01 (Viessmann Vitorond 200)",
            generated_at=current_time.strftime("%Y-%m-%d %H:%M:%S"),
            priority=priority,
            scheduled_start=start_dt.strftime("%Y-%m-%d %H:%M"),
            scheduled_end=end_dt.strftime("%Y-%m-%d %H:%M"),
            task_description="Execute high-pressure soot blowing on convection tube bundles and fireside waterwall lance wash.",
            assigned_crew="Mechanical Maintenance Crew B (2 Techs, 1 Safety Supervisor)",
            required_spares=["Soot Blower Packing Gaskets (x2)", "High-Pressure Steam Nozzles (x4)", "Flange Seals (x2)"],
            safety_protocols=["Lockout / Tagout (LOTO) Burner Fuel Supply", "Furnace Draft Purge Verification", "Thermal Personal Protective Equipment (PPE)"],
            estimated_duration_hours=3.5,
            projected_maintenance_cost_usd=shift_record["maintenance_cost"],
            projected_net_savings_usd=round(net_savings, 2)
        )
