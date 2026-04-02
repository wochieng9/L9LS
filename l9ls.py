# ============================================
# COMPLETE MICROSIMULATION CODE
# Post-discharge malaria prevention:
# L9LS vs DP
# ============================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. HEALTH STATES
# =========================================================

class HealthState(str, Enum):
    HEALTHY = "Healthy"
    SEVERE_MALARIA = "Severe_Malaria"
    SEVERE_ANEMIA = "Severe_Anemia"
    READMISSION = "Readmission"
    DEAD = "Dead"


# =========================================================
# 2. PARAMETERS
# =========================================================

@dataclass
class SimulationParams:
    """
    Central container for all model inputs.

    Replace placeholder values with study-specific estimates.
    """

    # -----------------------------
    # File paths
    # -----------------------------
    who_wfa_path: str = "/mnt/data/wfa_boys_0-to-5-years_zscores.xlsx"

    # -----------------------------
    # Cost inputs
    # -----------------------------
    cost_per_mg_L9LS: float = 1.00
    admin_cost_L9LS: float = 1.50

    dp_cost_per_tablet: float = 0.25
    admin_cost_DP_course: float = 0.50

    event_costs: Dict[HealthState, float] = field(default_factory=lambda: {
        HealthState.SEVERE_MALARIA: 15.0,
        HealthState.SEVERE_ANEMIA: 12.0,
        HealthState.READMISSION: 10.0,
        HealthState.DEAD: 0.0,
    })

    # -----------------------------
    # Disability weights
    # -----------------------------
    disability_weights: Dict[HealthState, float] = field(default_factory=lambda: {
        HealthState.HEALTHY: 0.0,
        HealthState.SEVERE_MALARIA: 0.210,
        HealthState.SEVERE_ANEMIA: 0.149,
        HealthState.READMISSION: 0.100,
        HealthState.DEAD: 1.0,  # not used for YLD
    })

    # -----------------------------
    # Baseline monthly risks
    # -----------------------------
    base_p_severe_malaria: float = 0.06
    base_p_severe_anemia: float = 0.03
    base_p_readmission: float = 0.02
    base_p_background_death: float = 0.001

    # -----------------------------
    # Case fatality ratios
    # -----------------------------
    cfr: Dict[HealthState, float] = field(default_factory=lambda: {
        HealthState.SEVERE_MALARIA: 0.03,
        HealthState.SEVERE_ANEMIA: 0.02,
        HealthState.READMISSION: 0.01,
    })

    # -----------------------------
    # L9LS efficacy
    # -----------------------------
    l9ls_P0: float = 0.80
    l9ls_half_life_days: int = 150
    l9ls_max_days: int = 180

    # -----------------------------
    # DP efficacy
    # -----------------------------
    dp_courses_months: Tuple[int, ...] = (0, 1, 2)
    dp_protection_in_course_month: float = 0.75
    dp_half_life_days: int = 30

    # -----------------------------
    # Dosing constants
    # -----------------------------
    l9ls_mg_per_kg: float = 10.0
    dp_tabs_per_day: float = 1.5
    dp_course_days: int = 3

    # -----------------------------
    # Age-at-discharge distribution
    # -----------------------------
    age_median_months: float = 14.0
    age_lognorm_sigma: float = 0.60
    age_min_months: int = 0
    age_max_months: int = 59

    # -----------------------------
    # WFA z-score distribution
    # -----------------------------
    wfa_z_mean: float = 0.0
    wfa_z_sd: float = 0.9
    wfa_z_min: float = -3.0
    wfa_z_max: float = 3.0

    # -----------------------------
    # Time preference
    # -----------------------------
    annual_discount_rate_costs: float = 0.03
    annual_discount_rate_health: float = 0.03

    # -----------------------------
    # YLL approximation
    # -----------------------------
    remaining_LE_at_birth_years: float = 65.0

    # -----------------------------
    # Simulation control
    # -----------------------------
    days_per_month: int = 30
    horizon_months: int = 6
    seed: int = 2025


# =========================================================
# 3. DISCOUNTING HELPERS
# =========================================================

def discount_factor(month_index: int, annual_rate: float) -> float:
    if annual_rate <= 0:
        return 1.0
    return 1.0 / ((1.0 + annual_rate) ** (month_index / 12.0))


def discounted_future_months(n_months: int, annual_rate: float) -> float:
    if n_months <= 0:
        return 0.0
    if annual_rate <= 0:
        return n_months / 12.0
    r_m = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    return (1.0 - (1.0 + r_m) ** (-n_months)) / (r_m * 12.0)


# =========================================================
# 4. WHO WEIGHT-FOR-AGE HELPERS
# =========================================================

def load_who_wfa(path: str) -> pd.DataFrame:
    """
    Load WHO WFA Excel table with columns Month, L, M, S.
    """
    df = pd.read_excel(path, sheet_name=0)
    df = df[["Month", "L", "M", "S"]].copy()
    df["Month"] = df["Month"].astype(int)
    df = df.set_index("Month").sort_index()
    return df


def lms_weight(age_months: float, z: float, who_df: pd.DataFrame) -> float:
    """
    WHO LMS formula with linear interpolation across months.
    """
    a = max(0.0, min(float(age_months), float(who_df.index.max())))
    a0 = int(np.floor(a))
    a1 = int(np.ceil(a))
    if a1 > who_df.index.max():
        a1 = a0

    L0, M0, S0 = who_df.loc[a0, ["L", "M", "S"]].astype(float)
    L1, M1, S1 = who_df.loc[a1, ["L", "M", "S"]].astype(float)

    w = a - a0
    L = (1.0 - w) * L0 + w * L1
    M = (1.0 - w) * M0 + w * M1
    S = (1.0 - w) * S0 + w * S1

    if abs(L) < 1e-8:
        return float(M * math.exp(S * z))
    return float(M * ((1.0 + L * S * z) ** (1.0 / L)))


def build_weight_trajectory_from_who(who_df: pd.DataFrame, z: float, max_age_months: int = 60) -> List[float]:
    return [lms_weight(a, z, who_df) for a in range(max_age_months + 1)]


def sample_age_at_discharge_months(
    rng: random.Random,
    median_m: float,
    sigma_log: float,
    a_min: int,
    a_max: int,
) -> int:
    mu = math.log(max(1e-8, median_m))
    x = rng.lognormvariate(mu, sigma_log)
    x = max(a_min, min(a_max, x))
    return int(round(x))


def sample_wfa_z(
    rng: random.Random,
    mean: float,
    sd: float,
    zmin: float,
    zmax: float,
) -> float:
    z = rng.normalvariate(mean, sd)
    return float(max(zmin, min(zmax, z)))


# =========================================================
# 5. CHILD CLASS
# =========================================================

class Child:
    def __init__(
        self,
        child_id: int,
        intervention: str,
        weight_trajectory: List[float],
        start_age_months: int = 0,
    ):
        self.id = child_id
        self.age_months = int(start_age_months)
        self.weight_trajectory = weight_trajectory
        self.weight_kg = self.weight_trajectory[self.age_months]
        self.intervention = intervention

        self.status: HealthState = HealthState.HEALTHY
        self.time_in_state = 0
        self.alive = True

        self.malaria_episodes = 0
        self.anemia_episodes = 0
        self.readmissions = 0
        self.death_month: Optional[int] = None

        self.costs = 0.0
        self.DALYs = 0.0
        self.life_months = 0

        self.entered_state_this_month = False
        self.dp_courses_given: List[int] = []

        self.history: List[Dict] = []

    @property
    def month_index(self) -> int:
        return self.life_months

    def update_weight(self):
        idx = min(self.age_months, len(self.weight_trajectory) - 1)
        self.weight_kg = self.weight_trajectory[idx]

    def log_month(self, extra: Dict):
        row = {
            "id": self.id,
            "age_months": self.age_months,
            "month_index": self.month_index,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "time_in_state": self.time_in_state,
            "weight_kg": self.weight_kg,
            "alive": self.alive,
            "costs_cum": self.costs,
            "DALYs_cum": self.DALYs,
        }
        row.update(extra)
        self.history.append(row)

    def simulate_month(
        self,
        params: SimulationParams,
        transition_fn,
        cost_fn,
        daly_fn,
        rng: random.Random,
    ):
        if not self.alive:
            return

        self.age_months += 1
        self.update_weight()

        prev_status = self.status
        new_status = transition_fn(self, params, rng)

        self.entered_state_this_month = (new_status != prev_status)
        self.time_in_state = 0 if self.entered_state_this_month else self.time_in_state + 1
        self.status = new_status

        if self.entered_state_this_month:
            if new_status == HealthState.SEVERE_MALARIA:
                self.malaria_episodes += 1
            elif new_status == HealthState.SEVERE_ANEMIA:
                self.anemia_episodes += 1
            elif new_status == HealthState.READMISSION:
                self.readmissions += 1
            elif new_status == HealthState.DEAD:
                self.alive = False
                self.death_month = self.month_index

        c = cost_fn(self, params)
        d = daly_fn(self, params)

        self.costs += c
        self.DALYs += d

        self.log_month({
            "entered_state": self.entered_state_this_month,
            "costs_month": c,
            "DALYs_month": d,
        })

        self.life_months += 1


# =========================================================
# 6. DAY-BASED EXPONENTIAL PROTECTION
# =========================================================

def _daily_hazard_from_monthly(p_month: float, days_per_month: int) -> float:
    p = max(0.0, min(0.999999, p_month))
    return -math.log(1.0 - p) / days_per_month


def _prot_day_array_for_arm(intervention: str, params: SimulationParams) -> np.ndarray:
    D = params.days_per_month
    T = params.horizon_months * D
    t = np.arange(T, dtype=float)

    if intervention.upper() == "L9LS":
        k = math.log(2.0) / max(1e-9, params.l9ls_half_life_days)
        prot = params.l9ls_P0 * np.exp(-k * t)
        if params.l9ls_max_days < T:
            prot[params.l9ls_max_days:] = 0.0
        return np.clip(prot, 0.0, 1.0)

    k = math.log(2.0) / max(1e-9, params.dp_half_life_days)
    E0 = params.dp_protection_in_course_month
    surv = np.ones(T, dtype=float)

    for s_m in params.dp_courses_months:
        s_day = s_m * D
        delta = t - s_day
        contrib = np.zeros_like(t)
        mask = delta >= 0
        contrib[mask] = E0 * np.exp(-k * delta[mask])
        surv *= (1.0 - np.clip(contrib, 0.0, 1.0))

    prot = 1.0 - surv
    return np.clip(prot, 0.0, 1.0)


def _monthly_integrated_hazards(intervention: str, params: SimulationParams) -> Dict[str, np.ndarray]:
    D = params.days_per_month
    T = params.horizon_months * D

    prot_day = _prot_day_array_for_arm(intervention, params)
    scaler_sum = np.add.reduceat(1.0 - prot_day, np.arange(0, T, D))

    lam_sm = _daily_hazard_from_monthly(params.base_p_severe_malaria, D)
    lam_sa = _daily_hazard_from_monthly(params.base_p_severe_anemia, D)
    lam_rd = _daily_hazard_from_monthly(params.base_p_readmission, D)
    lam_bg = _daily_hazard_from_monthly(params.base_p_background_death, D)

    H_sm = lam_sm * scaler_sum
    H_sa = lam_sa * scaler_sum
    H_rd = lam_rd * scaler_sum
    H_total = H_sm + H_sa + H_rd
    p_any = 1.0 - np.exp(-H_total)

    p_bg = np.full_like(p_any, 1.0 - np.exp(-lam_bg * D))

    return {
        "H_sm": H_sm,
        "H_sa": H_sa,
        "H_rd": H_rd,
        "H_total": H_total,
        "p_any": p_any,
        "p_bg": p_bg,
    }


def make_transition_fn_fast(precomp: Dict[str, np.ndarray]):
    H_sm = precomp["H_sm"]
    H_sa = precomp["H_sa"]
    H_rd = precomp["H_rd"]
    H_total = precomp["H_total"]
    p_any = precomp["p_any"]
    p_bg = precomp["p_bg"]

    def _fn(child: Child, params: SimulationParams, rng: random.Random) -> HealthState:
        if not child.alive:
            return HealthState.DEAD

        m = child.month_index

        if H_total[m] > 0.0 and rng.random() < float(p_any[m]):
            u = rng.random() * float(H_total[m])
            if u < H_sm[m]:
                cause = HealthState.SEVERE_MALARIA
            elif u < H_sm[m] + H_sa[m]:
                cause = HealthState.SEVERE_ANEMIA
            else:
                cause = HealthState.READMISSION

            if rng.random() < params.cfr.get(cause, 0.0):
                return HealthState.DEAD
            return cause

        if rng.random() < float(p_bg[m]):
            return HealthState.DEAD

        return HealthState.HEALTHY

    return _fn


# =========================================================
# 7. COSTS AND DALYS
# =========================================================

def intervention_cost_this_month(child: Child, params: SimulationParams) -> float:
    m = child.month_index
    df = discount_factor(m, params.annual_discount_rate_costs)

    if child.intervention.upper() == "L9LS" and m == 0:
        dose_mg = params.l9ls_mg_per_kg * child.weight_kg
        return df * (dose_mg * params.cost_per_mg_L9LS + params.admin_cost_L9LS)

    if child.intervention.upper() == "DP" and m in params.dp_courses_months:
        tablets = params.dp_tabs_per_day * params.dp_course_days
        return df * (tablets * params.dp_cost_per_tablet + params.admin_cost_DP_course)

    return 0.0


def event_cost_this_month(child: Child, params: SimulationParams) -> float:
    m = child.month_index
    df = discount_factor(m, params.annual_discount_rate_costs)

    c = 0.0
    if child.entered_state_this_month:
        if child.status in (HealthState.SEVERE_MALARIA, HealthState.SEVERE_ANEMIA, HealthState.READMISSION):
            c += params.event_costs.get(child.status, 0.0)
        elif child.status == HealthState.DEAD:
            c += params.event_costs.get(HealthState.DEAD, 0.0)

    return df * c


def cost_fn(child: Child, params: SimulationParams) -> float:
    return intervention_cost_this_month(child, params) + event_cost_this_month(child, params)


def yld_this_month(child: Child, params: SimulationParams) -> float:
    m = child.month_index
    df = discount_factor(m, params.annual_discount_rate_health)
    dw = params.disability_weights.get(child.status, 0.0) if child.alive else 0.0
    return df * (dw / 12.0)


def yll_on_death(child: Child, params: SimulationParams) -> float:
    if not child.alive and child.death_month == child.month_index:
        age_years = child.age_months / 12.0
        remaining_years = max(0.0, params.remaining_LE_at_birth_years - age_years)
        remaining_months = int(round(remaining_years * 12.0))
        df = discount_factor(child.month_index, params.annual_discount_rate_health)
        return df * discounted_future_months(remaining_months, params.annual_discount_rate_health)
    return 0.0


def daly_fn(child: Child, params: SimulationParams) -> float:
    return yld_this_month(child, params) + yll_on_death(child, params)


# =========================================================
# 8. RESULTS CONTAINER
# =========================================================

@dataclass
class CohortResult:
    n: int
    mean_cost: float
    mean_dalys: float
    mean_life_months: float
    totals: Dict[str, float]
    traces: Optional[List[List[Dict]]] = None


# =========================================================
# 9. FAST WHO-BASED COHORT SIMULATION
# =========================================================

def simulate_cohort_with_who_fast(
    n_children: int,
    intervention: str,
    params: SimulationParams,
    keep_traces: bool = False,
    common_population_seed: Optional[int] = None,
) -> CohortResult:
    """
    Fast simulation using:
    - WHO WFA for child-specific weight trajectories
    - sampled age at discharge
    - sampled WFA z-score
    - precomputed monthly integrated hazards
    """

    rng_pop = random.Random(common_population_seed if common_population_seed is not None else params.seed)
    rng_sim = random.Random(params.seed if intervention.upper() == "L9LS" else params.seed + 1)

    who_df = load_who_wfa(params.who_wfa_path)

    ages = [
        sample_age_at_discharge_months(
            rng_pop,
            params.age_median_months,
            params.age_lognorm_sigma,
            params.age_min_months,
            params.age_max_months,
        )
        for _ in range(n_children)
    ]

    zs = [
        sample_wfa_z(
            rng_pop,
            params.wfa_z_mean,
            params.wfa_z_sd,
            params.wfa_z_min,
            params.wfa_z_max,
        )
        for _ in range(n_children)
    ]

    children: List[Child] = []
    for i in range(n_children):
        traj = build_weight_trajectory_from_who(who_df, zs[i], max_age_months=60)
        ch = Child(i, intervention, traj, start_age_months=ages[i])
        children.append(ch)

    precomp = _monthly_integrated_hazards(intervention, params)
    transition_fast = make_transition_fn_fast(precomp)

    for m in range(params.horizon_months):
        for ch in children:
            if ch.intervention.upper() == "DP" and m in params.dp_courses_months:
                ch.dp_courses_given.append(m)
            ch.simulate_month(params, transition_fast, cost_fn, daly_fn, rng_sim)

    total_cost = sum(ch.costs for ch in children)
    total_dalys = sum(ch.DALYs for ch in children)
    total_life_months = sum(ch.life_months for ch in children)

    totals = {
        "episodes_malaria": sum(ch.malaria_episodes for ch in children),
        "episodes_anemia": sum(ch.anemia_episodes for ch in children),
        "readmissions": sum(ch.readmissions for ch in children),
        "deaths": sum(1 for ch in children if not ch.alive),
        "dp_courses": sum(len(ch.dp_courses_given) for ch in children if ch.intervention.upper() == "DP"),
        "l9ls_doses": sum(1 for ch in children if ch.intervention.upper() == "L9LS"),
    }

    traces = [ch.history for ch in children] if keep_traces else None

    return CohortResult(
        n=n_children,
        mean_cost=total_cost / n_children,
        mean_dalys=total_dalys / n_children,
        mean_life_months=total_life_months / n_children,
        totals=totals,
        traces=traces,
    )


# =========================================================
# 10. ICER
# =========================================================

def icer(result_a: CohortResult, result_b: CohortResult) -> Tuple[float, Dict[str, float]]:
    d_cost = result_a.mean_cost - result_b.mean_cost
    d_dalys = result_a.mean_dalys - result_b.mean_dalys
    val = math.inf if abs(d_dalys) < 1e-12 else d_cost / d_dalys
    return val, {"d_cost": d_cost, "d_dalys": d_dalys}


# =========================================================
# 11. DATAFRAME EXPORT HELPERS
# =========================================================

def cohort_summary_df(res: CohortResult, arm_label: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "Arm": arm_label,
        "Mean cost (US$)": res.mean_cost,
        "Mean DALYs": res.mean_dalys,
        "Mean life-months": res.mean_life_months,
        "Total severe malaria": res.totals["episodes_malaria"],
        "Total severe anemia": res.totals["episodes_anemia"],
        "Total readmissions": res.totals["readmissions"],
        "Total deaths": res.totals["deaths"],
        "Total L9LS doses": res.totals["l9ls_doses"],
        "Total DP courses": res.totals["dp_courses"],
    }])


def icer_df(res_a: CohortResult, res_b: CohortResult, label_a="L9LS", label_b="DP") -> pd.DataFrame:
    d_cost = res_a.mean_cost - res_b.mean_cost
    d_dalys = res_a.mean_dalys - res_b.mean_dalys
    icer_raw = math.inf if abs(d_dalys) < 1e-12 else d_cost / d_dalys
    icer_averted = math.inf if abs(d_dalys) < 1e-12 else d_cost / (-d_dalys)
    return pd.DataFrame([{
        "Comparison": f"{label_a} vs {label_b}",
        "ΔCost (US$)": d_cost,
        "ΔDALYs (A − B)": d_dalys,
        "ICER (US$ / ΔDALY)": icer_raw,
        "ICER (US$ / DALY averted)": icer_averted,
    }])


def child_totals_df(res: CohortResult, arm_label: str) -> pd.DataFrame:
    if res.traces is None:
        raise ValueError("Run with keep_traces=True.")
    rows = []
    for child_hist in res.traces:
        if not child_hist:
            continue
        last = child_hist[-1]
        ep_sm = sum(1 for r in child_hist if r.get("entered_state") and r.get("status") == "Severe_Malaria")
        ep_sa = sum(1 for r in child_hist if r.get("entered_state") and r.get("status") == "Severe_Anemia")
        ep_rd = sum(1 for r in child_hist if r.get("entered_state") and r.get("status") == "Readmission")
        rows.append({
            "Arm": arm_label,
            "id": last["id"],
            "age_months_final": last["age_months"],
            "weight_kg_final": last["weight_kg"],
            "life_months": len(child_hist),
            "cost_total": last["costs_cum"],
            "dalys_total": last["DALYs_cum"],
            "episodes_severe_malaria": ep_sm,
            "episodes_severe_anemia": ep_sa,
            "readmissions": ep_rd,
            "died": (not last["alive"]),
        })
    return pd.DataFrame(rows)


def monthly_traces_df(res: CohortResult, arm_label: str) -> pd.DataFrame:
    if res.traces is None:
        raise ValueError("Run with keep_traces=True.")
    rows = []
    for child_hist in res.traces:
        for r in child_hist:
            rows.append({
                "Arm": arm_label,
                "id": r["id"],
                "month_index": r["month_index"],
                "age_months": r["age_months"],
                "status": r["status"],
                "alive": r["alive"],
                "weight_kg": r["weight_kg"],
                "time_in_state": r["time_in_state"],
                "costs_month": r.get("costs_month", np.nan),
                "DALYs_month": r.get("DALYs_month", np.nan),
                "costs_cum": r.get("costs_cum", np.nan),
                "DALYs_cum": r.get("DALYs_cum", np.nan),
                "entered_state": r.get("entered_state", False),
            })
    return pd.DataFrame(rows)


# =========================================================
# 12. APPENDIX FIGURE HELPERS
# =========================================================

def traces_to_long_df(traces: List[List[Dict]], arm_label: str) -> pd.DataFrame:
    rows = []
    for child_hist in traces:
        for r in child_hist:
            rows.append({
                "Arm": arm_label,
                "id": r["id"],
                "month_index": r["month_index"],
                "status": r["status"],
                "alive": bool(r["alive"]),
                "costs_cum": r.get("costs_cum", np.nan),
                "DALYs_cum": r.get("DALYs_cum", np.nan),
            })
    return pd.DataFrame(rows)


def occupancy_table(df_long: pd.DataFrame, n_children: int) -> pd.DataFrame:
    grp = df_long.groupby(["Arm", "month_index", "status"]).size().reset_index(name="count")
    grp["prop"] = grp["count"] / n_children
    wide = grp.pivot_table(index=["Arm", "month_index"], columns="status", values="prop", fill_value=0.0)
    return wide.reset_index().sort_values(["Arm", "month_index"])


def mean_cum_trajectories(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long[["Arm", "id", "month_index", "costs_cum", "DALYs_cum"]].copy()
    df = df.sort_values(["Arm", "id", "month_index"]).drop_duplicates(["Arm", "id", "month_index"], keep="last")

    def agg(g):
        return pd.Series({
            "mean_costs_cum": g["costs_cum"].mean(),
            "p2p5_costs_cum": g["costs_cum"].quantile(0.025),
            "p97p5_costs_cum": g["costs_cum"].quantile(0.975),
            "mean_dalys_cum": g["DALYs_cum"].mean(),
            "p2p5_dalys_cum": g["DALYs_cum"].quantile(0.025),
            "p97p5_dalys_cum": g["DALYs_cum"].quantile(0.975),
        })

    return df.groupby(["Arm", "month_index"]).apply(agg).reset_index()


def make_appendix_figures(res_L9LS: CohortResult, res_DP: CohortResult, output_prefix: str = "appendix"):
    if res_L9LS.traces is None or res_DP.traces is None:
        raise ValueError("Appendix figures require keep_traces=True.")

    df_L = traces_to_long_df(res_L9LS.traces, "L9LS")
    df_D = traces_to_long_df(res_DP.traces, "DP")
    df_all = pd.concat([df_L, df_D], ignore_index=True)

    n_children = df_L["id"].nunique()
    occ = occupancy_table(df_all, n_children)
    traj = mean_cum_trajectories(df_all)

    occ.to_csv(f"{output_prefix}_state_occupancy.csv", index=False)
    traj.to_csv(f"{output_prefix}_mean_cumulative_costs_dalys.csv", index=False)

    # A1 state occupancy
    def plot_occupancy_for_arm(arm: str, out_path: str):
        sub = occ[occ["Arm"] == arm].copy().sort_values("month_index")
        states = [c for c in occ.columns if c not in ["Arm", "month_index"]]
        x = sub["month_index"].values
        y = sub[states].values.T
        plt.figure(figsize=(7, 4))
        plt.stackplot(x, *y, labels=states)
        plt.xlabel("Months since discharge")
        plt.ylabel("Proportion of cohort")
        plt.title(f"State occupancy over time ({arm})")
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    plot_occupancy_for_arm("L9LS", f"{output_prefix}_figA1_state_occupancy_L9LS.png")
    plot_occupancy_for_arm("DP", f"{output_prefix}_figA1_state_occupancy_DP.png")

    # A2 raster
    state_levels = ["Healthy", "Severe_Malaria", "Severe_Anemia", "Readmission", "Dead"]
    state_to_int = {s: i for i, s in enumerate(state_levels)}

    def raster_for_arm(arm: str, sample_n: int, out_path: str):
        sub = df_all[df_all["Arm"] == arm].copy()
        ids = sub["id"].drop_duplicates().sample(n=min(sample_n, sub["id"].nunique()), random_state=123)
        sub = sub[sub["id"].isin(ids)]
        piv = sub.pivot_table(index="id", columns="month_index", values="status", aggfunc="last")
        mat = piv.applymap(lambda s: state_to_int.get(str(s), np.nan)).values
        plt.figure(figsize=(7, 6))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.xlabel("Months since discharge")
        plt.ylabel("Child (sample)")
        plt.title(f"Individual state traces (raster) — {arm}")
        cbar = plt.colorbar()
        cbar.set_ticks(range(len(state_levels)))
        cbar.set_ticklabels(state_levels)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    raster_for_arm("L9LS", 150, f"{output_prefix}_figA2_raster_L9LS.png")
    raster_for_arm("DP", 150, f"{output_prefix}_figA2_raster_DP.png")

    # A3 cumulative costs and DALYs
    def plot_cumulative_with_bands(arm: str, out_cost: str, out_daly: str):
        sub = traj[traj["Arm"] == arm].sort_values("month_index")
        x = sub["month_index"].values

        plt.figure(figsize=(7, 4))
        plt.plot(x, sub["mean_costs_cum"].values, label="Mean")
        plt.fill_between(x, sub["p2p5_costs_cum"].values, sub["p97p5_costs_cum"].values, alpha=0.3, label="95% band")
        plt.xlabel("Months since discharge")
        plt.ylabel("Cumulative cost (US$) per child")
        plt.title(f"Cumulative costs over time — {arm}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_cost, dpi=300)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(x, sub["mean_dalys_cum"].values, label="Mean")
        plt.fill_between(x, sub["p2p5_dalys_cum"].values, sub["p97p5_dalys_cum"].values, alpha=0.3, label="95% band")
        plt.xlabel("Months since discharge")
        plt.ylabel("Cumulative DALYs per child")
        plt.title(f"Cumulative DALYs over time — {arm}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_daly, dpi=300)
        plt.close()

    plot_cumulative_with_bands("L9LS", f"{output_prefix}_figA3_cumcost_L9LS.png", f"{output_prefix}_figA3_cumdalys_L9LS.png")
    plot_cumulative_with_bands("DP", f"{output_prefix}_figA3_cumcost_DP.png", f"{output_prefix}_figA3_cumdalys_DP.png")


# =========================================================
# 13. EXAMPLE RUN
# =========================================================

if __name__ == "__main__":
    params = SimulationParams()

    # Point this to your WHO file
    params.who_wfa_path = "/mnt/data/wfa_boys_0-to-5-years_zscores.xlsx"

    # Example: same sampled population across arms
    n = 10000
    pop_seed = 4242

    # Main results
    res_L9LS = simulate_cohort_with_who_fast(
        n_children=n,
        intervention="L9LS",
        params=params,
        keep_traces=False,
        common_population_seed=pop_seed,
    )

    res_DP = simulate_cohort_with_who_fast(
        n_children=n,
        intervention="DP",
        params=params,
        keep_traces=False,
        common_population_seed=pop_seed,
    )

    icer_val, deltas = icer(res_L9LS, res_DP)

    print("=== Means per child over horizon ===")
    print(f"L9LS : Cost=${res_L9LS.mean_cost:,.2f} | DALYs={res_L9LS.mean_dalys:.4f} | Life-months={res_L9LS.mean_life_months:.2f}")
    print(f"DP   : Cost=${res_DP.mean_cost:,.2f} | DALYs={res_DP.mean_dalys:.4f} | Life-months={res_DP.mean_life_months:.2f}")
    print()
    print("=== Totals across cohort ===")
    print("L9LS:", res_L9LS.totals)
    print("DP  :", res_DP.totals)
    print()
    print("=== Incremental (L9LS - DP) ===")
    print(f"ΔCost=${deltas['d_cost']:,.2f} | ΔDALYs={deltas['d_dalys']:.4f}")
    if math.isfinite(icer_val):
        print(f"ICER = ${icer_val:,.2f} per ΔDALY")
        if deltas["d_dalys"] < 0:
            print(f"Equivalent = ${deltas['d_cost']/(-deltas['d_dalys']):,.2f} per DALY averted")
    else:
        print("ICER undefined (ΔDALYs≈0)")

    # Summary DataFrames
    summary_df = pd.concat([
        cohort_summary_df(res_L9LS, "L9LS"),
        cohort_summary_df(res_DP, "DP"),
    ], ignore_index=True)

    icer_table = icer_df(res_L9LS, res_DP, "L9LS", "DP")

    print()
    print(summary_df)
    print()
    print(icer_table)

    # Optional trace run for appendix figures
    res_L9LS_trace = simulate_cohort_with_who_fast(
        n_children=2000,
        intervention="L9LS",
        params=params,
        keep_traces=True,
        common_population_seed=pop_seed,
    )

    res_DP_trace = simulate_cohort_with_who_fast(
        n_children=2000,
        intervention="DP",
        params=params,
        keep_traces=True,
        common_population_seed=pop_seed,
    )

    make_appendix_figures(res_L9LS_trace, res_DP_trace, output_prefix="appendix")

    # Optional detailed DataFrames
    df_children = pd.concat([
        child_totals_df(res_L9LS_trace, "L9LS"),
        child_totals_df(res_DP_trace, "DP"),
    ], ignore_index=True)

    df_monthly = pd.concat([
        monthly_traces_df(res_L9LS_trace, "L9LS"),
        monthly_traces_df(res_DP_trace, "DP"),
    ], ignore_index=True)

    # Save outputs
    summary_df.to_csv("summary_per_arm.csv", index=False)
    icer_table.to_csv("icer_table.csv", index=False)
    df_children.to_csv("child_totals.csv", index=False)
    df_monthly.to_csv("monthly_traces.csv", index=False)
