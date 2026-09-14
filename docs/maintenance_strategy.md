# Maintenance Strategy & Opportunistic Scheduling Theory

## 1. Evolution of Maintenance Paradigms

```
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│  Run-to-Failure (RTF)   │ Time-Based Maint. (TBM) │ Condition-Based (CBM)   │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ Fix only when broken    │ Inspect every 30 days   │ Continuous PINN Digital │
│                         │ regardless of condition │ Twin state monitoring   │
│ Catastrophic downtime   │ Unnecessary downtime    │ Cost-optimal dynamic    │
│ High emergency cost     │ Premature part changes  │ intervention window     │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

In industrial boiler operations, tube rupture is the single most costly unplanned failure event, resulting in emergency shutdowns, thermal shock to refractory linings, loss of factory production, and severe safety hazards.

---

## 2. Opportunistic Maintenance Optimization Formulation

Condition-Based Maintenance (CBM) is combined with **Opportunistic Scheduling**:
Given the predicted Remaining Safe Operating Window ($RSOW$), maintenance should be scheduled at an intervention time $\tau^* \in [0, RSOW]$ that minimizes the total expected cost function:

$$\min_{\tau} J(\tau) = C_{fuel\_waste}(\tau) + C_{intervention}(\text{shift}_\tau) + C_{failure\_risk}(\tau)$$

### 2.1 Cumulative Fuel Inefficiency Waste
As soot deposits build up on waterwall tubes, thermal transfer efficiency degrades, forcing the boiler to burn excess fuel:

$$C_{fuel\_waste}(\tau) = \int_0^\tau \dot{m}_{excess\_fuel}(R_f(t)) \cdot \text{Price}_{fuel} \, dt$$

### 2.2 Shift-Dependent Production Downtime Penalty
Shutting down a boiler during a peak daytime production shift incurs massive factory downtime losses, whereas shutting down during a scheduled off-peak night shift or planned weekend outage reduces production disruption by up to 75%:

$$C_{intervention}(\text{shift}_\tau) = C_{routine\_cleaning} + t_{duration} \cdot (C_{labor} \cdot M_{shift} + C_{production\_loss}(\text{shift}_\tau))$$

### 2.3 Failure Probability & Risk Penalty
The probability of localized tube metal overheating and creep rupture increases as the intervention is delayed:

$$C_{failure\_risk}(\tau) = P_{rupture}(\tau) \cdot C_{unplanned\_outage}$$

where $P_{rupture}(\tau)$ is derived from the Larson-Miller parameter for tube metal creep at temperature $T_{metal}(t)$.

---

## 3. Human Factors & NASA-TLX Cognitive Work Design

In modern industrial control rooms, operators are subjected to "alarm flooding" when multiple raw sensor thresholds are breached simultaneously. This leads to **Alarm Fatigue** and cognitive overload (high NASA-TLX workload scores).

### Cognitive Ergonomics Principles Implemented:
1. **Single Root-Cause Diagnostic Cards**: The Digital Twin synthesizes multiple raw process spikes into a single, physics-grounded diagnostic card with clear root-cause isolation (Fireside Soot vs. Waterside Scale vs. Excess Air).
2. **Predictive Horizon vs. Reactive Scrambling**: Provides a clear countdown to critical limits ($RSOW$), transitioning the operator from stressful reactive firefighting to structured proactive planning.
3. **Automated Work Package Assembly**: Generates pre-populated Work Orders with assigned crews, required spare parts, and Lockout/Tagout (LOTO) safety checklists, eliminating administrative cognitive overhead.
