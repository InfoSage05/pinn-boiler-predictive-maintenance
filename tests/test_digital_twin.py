"""Unit tests for Digital Twin synchronization, anomaly detection, and scheduling."""

import os
import sys
from datetime import datetime
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.digital_twin.synchronizer import DigitalTwinSynchronizer
from src.digital_twin.what_if_simulator import WhatIfScenarioSimulator
from src.maintenance.anomaly_detector import BoilerAnomalyDetector
from src.maintenance.scheduler import OpportunisticMaintenanceScheduler


def test_digital_twin_synchronization():
    sync = DigitalTwinSynchronizer()
    telemetry = {
        "fuel_flow_kg_s": 2.5,
        "water_flow_kg_s": 7.75,
        "air_temp_k": 293.15,
        "return_water_temp_k": 333.0,
        "supply_water_temp_k": 345.0,
        "load_pct": 80.0
    }
    snapshot = sync.process_telemetry_frame(telemetry)
    assert snapshot.health_index > 0.0
    assert snapshot.health_index <= 1.0
    assert snapshot.alarm_level in ["NORMAL", "ADVISORY", "WARNING", "CRITICAL"]
    assert snapshot.remaining_safe_operating_window_hours > 0.0


def test_anomaly_detection_root_cause():
    detector = BoilerAnomalyDetector()
    
    # Clean nominal conditions
    clean_eval = detector.evaluate_alarm(tube_metal_temp_c=530.0, estimated_rf=0.005, physics_residual_kw=5.0)
    assert clean_eval["alarm_level"] == "NORMAL"
    assert clean_eval["is_anomalous"] is False
    
    # Severe fouling condition
    foul_eval = detector.evaluate_alarm(tube_metal_temp_c=545.0, estimated_rf=0.038, physics_residual_kw=12.0)
    assert foul_eval["alarm_level"] == "CRITICAL"
    assert "Soot" in foul_eval["root_cause"] or "Fouling" in foul_eval["root_cause"]


def test_maintenance_scheduling():
    scheduler = OpportunisticMaintenanceScheduler()
    res = scheduler.optimize_maintenance_schedule(current_rf=0.028, rsow_hours=24.0, current_time=datetime(2026, 3, 1, 8, 0))
    
    assert res["optimal_shift"] is not None
    assert res["work_order"] is not None
    assert res["work_order"].work_order_id.startswith("WO-")
    assert res["work_order"].projected_net_savings_usd > 0.0
