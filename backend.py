"""
gcd_backend.py  —  GCD (Galvanostatic Charge-Discharge) Analyzer
=================================================================
Scientific engine for full GCD analysis.

Supported modes
---------------
A. Normal GCD   — per current-density metrics from best-quality cycles
B. Stability    — cycle-by-cycle capacitance retention tracking

Input format (wide table)
--------------------------
Row 0 (header): Time (s) | I1 [or J1] | I2 | I3 | ...
Rows 1-N:       float     | voltage    | voltage | ...

Column headers for current-density columns can be:
  • bare numbers:        "1", "2", "5", "10"
  • with units:          "1 A/g", "2mA/cm2", "0.5 A", "10 mA"
  • descriptive:         "1A/g", "5mA"

The first column is always Time (s).
All other columns are potential (V) at a given current or current density.

Cycle detection
---------------
Real GCD data may contain multiple full or partial charge–discharge cycles
per column.  We detect cycle boundaries by finding local minima/maxima in
the potential trace, then collect all complete cycles (one charge + one
discharge half).  If the cell doesn't reach the full potential window,
that's fine — we work with what the data provides.

Best-cycle selection
---------------------
We always pick the cycle with the longest discharge half — this gives the
most accurate capacitance because integration over a longer, cleaner trace
reduces noise effects.

All scientific equations are cited inline.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — FILE LOADING
# ═══════════════════════════════════════════════════════════════

def load_gcd_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load CSV or Excel GCD data.
    Column 0  = Time (s)
    Column 1+ = Potential (V) at each current/current-density
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=";")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported format: upload .csv, .xlsx, or .xls")

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("File contains no usable data.")
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — COLUMN PARSING
# ═══════════════════════════════════════════════════════════════

_UNIT_SCALE = {
    # key: (base_unit_label, multiplier_to_A_per_g_or_A)
    "a/g":   ("A/g",   1.0),
    "ma/g":  ("A/g",   1e-3),
    "a/cm2": ("A/cm²", 1.0),
    "ma/cm2":("A/cm²", 1e-3),
    "a":     ("A",     1.0),
    "ma":    ("A",     1e-3),
    "ua":    ("A",     1e-6),
}


def parse_gcd_columns(df: pd.DataFrame, data_type: str) -> dict:
    """
    Parse the wide GCD table.

    Parameters
    ----------
    df        : raw DataFrame (first column = time, rest = potential)
    data_type : "current_density" | "current"

    Returns
    -------
    {
        "time_col": str,
        "cd_columns": [
            {
                "col":         original column name,
                "value":       numeric value (A/g or A),
                "unit":        unit string,
                "label":       display label,
            }, ...
        ]
    }
    """
    cols = list(df.columns)
    time_col = cols[0]   # first column is always time

    cd_columns = []
    for c in cols[1:]:
        raw = str(c).strip()
        # Try to extract a number + optional unit
        m = re.match(
            r"^\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([a-zA-Z/²]*)\s*$",
            raw,
            re.IGNORECASE,
        )
        if m:
            num = float(m.group(1))
            unit_raw = m.group(2).strip().lower().replace("²", "2")
            if unit_raw in _UNIT_SCALE:
                unit_label, scale = _UNIT_SCALE[unit_raw]
                value = num * scale
            else:
                # No unit: treat as the native data_type unit
                unit_label = "A/g" if data_type == "current_density" else "A"
                value = num
            label = f"{num:g} {unit_label}"
        else:
            # Can't parse — use column name as label, value = NaN
            value = np.nan
            unit_label = "A/g" if data_type == "current_density" else "A"
            label = raw

        cd_columns.append({
            "col":   c,
            "value": value,
            "unit":  unit_label,
            "label": label,
        })

    # Sort by numeric current/density value (NaN last)
    cd_columns.sort(key=lambda x: x["value"] if np.isfinite(x["value"]) else 1e18)
    return {"time_col": time_col, "cd_columns": cd_columns}


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CYCLE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_cycles(
    time: np.ndarray,
    voltage: np.ndarray,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    min_half_cycle_points: int = 5,
) -> list[dict]:
    """
    Detect charge–discharge cycles in a GCD voltage trace.

    Strategy
    --------
    1. Smooth the signal slightly to reduce noise.
    2. Find turning points: local maxima (end of charge) and
       local minima (end of discharge).
    3. Pair consecutive min→max (charge) with max→min (discharge)
       to form complete cycles.
    4. Filter out incomplete or very short half-cycles.

    Real GCD data often does NOT reach the full potential window —
    this is handled by working purely with the detected turning
    points rather than enforcing hard voltage limits.

    Parameters
    ----------
    time, voltage         : arrays from the data column
    v_min, v_max          : optional nominal window limits (for IR-drop calc)
    min_half_cycle_points : discard half-cycles with fewer points than this

    Returns
    -------
    List of cycle dicts:
        {
            "cycle_index":    int (1-based),
            "charge":         {"time": arr, "voltage": arr},
            "discharge":      {"time": arr, "voltage": arr},
            "charge_time":    float (s),
            "discharge_time": float (s),
            "v_top":          float (V at top of cycle),
            "v_bottom":       float (V at bottom of cycle),
            "ir_drop":        float (V, estimated from discharge start)
        }
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)

    # Remove NaN rows
    mask = np.isfinite(time) & np.isfinite(voltage)
    time = time[mask]
    voltage = voltage[mask]

    if len(time) < 2 * min_half_cycle_points:
        return []

    # ---- Find turning points ----
    # Use a prominence-based peak finder so small noise bumps are ignored.
    # We look for both peaks (charge tops) and valleys (discharge bottoms).
    voltage_range = voltage.max() - voltage.min()
    prominence_threshold = max(voltage_range * 0.10, 0.005)   # ≥10% of range or 5 mV

    peak_idx, _   = find_peaks(voltage,  prominence=prominence_threshold)
    valley_idx, _ = find_peaks(-voltage, prominence=prominence_threshold)

    # Combine and sort all turning points
    turning_pts = sorted(
        [(i, "peak") for i in peak_idx] + [(i, "valley") for i in valley_idx],
        key=lambda x: x[0],
    )

    if len(turning_pts) < 2:
        # Fallback: treat the whole trace as one cycle by finding global max/min
        i_max = int(np.argmax(voltage))
        i_min = int(np.argmin(voltage))
        turning_pts = sorted(
            [(i_min, "valley"), (i_max, "peak")],
            key=lambda x: x[0],
        )

    # ---- Pair turning points into cycles ----
    cycles = []
    i = 0
    while i < len(turning_pts) - 1:
        # A complete cycle = valley → peak (charge) + peak → next valley (discharge)
        # OR peak → valley (discharge) + valley → next peak (charge)
        # We always want: some start → peak → valley
        idx0, t0 = turning_pts[i]
        idx1, t1 = turning_pts[i + 1]

        if t0 == "valley" and t1 == "peak":
            # Charge segment: valley → peak
            charge_start = idx0
            charge_end = idx1
            # Look for the next valley (discharge end)
            if i + 2 < len(turning_pts):
                idx2, t2 = turning_pts[i + 2]
                if t2 == "valley":
                    discharge_start = idx1
                    discharge_end = idx2
                else:
                    i += 1
                    continue
            else:
                # Incomplete: discharge goes to end of data
                discharge_start = idx1
                discharge_end = len(time) - 1

            charge_t   = time[charge_start:charge_end + 1]
            charge_v   = voltage[charge_start:charge_end + 1]
            disch_t    = time[discharge_start:discharge_end + 1]
            disch_v    = voltage[discharge_start:discharge_end + 1]

            if (len(charge_t) >= min_half_cycle_points and
                    len(disch_t) >= min_half_cycle_points):

                # IR drop: voltage drop at the very start of discharge
                # (instantaneous resistance effect)
                # ΔV_IR = V(end_of_charge) - V(start_of_discharge_after_switch)
                # Estimated as the drop between the peak and the next point
                if len(disch_v) >= 2:
                    ir_drop = abs(float(disch_v[0]) - float(disch_v[1]))
                else:
                    ir_drop = 0.0

                cycles.append({
                    "cycle_index":    len(cycles) + 1,
                    "charge":         {"time": charge_t, "voltage": charge_v},
                    "discharge":      {"time": disch_t,  "voltage": disch_v},
                    "charge_time":    float(charge_t[-1] - charge_t[0]),
                    "discharge_time": float(disch_t[-1] - disch_t[0]),
                    "v_top":          float(np.max(charge_v)),
                    "v_bottom":       float(np.min(disch_v)),
                    "ir_drop":        ir_drop,
                })
            i += 2   # advance past the charge+discharge pair
        elif t0 == "peak" and t1 == "valley":
            # Only a discharge segment visible at the start of data → skip to next valley
            i += 1
        else:
            i += 1

    return cycles


def select_best_cycle(cycles: list[dict]) -> Optional[dict]:
    """
    Select the cycle with the longest discharge time.

    Rationale: longer discharge → more of the potential window was swept →
    more accurate capacitance integral; less affected by IR drop artefacts.
    """
    if not cycles:
        return None
    return max(cycles, key=lambda c: c["discharge_time"])


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PER-CYCLE ELECTROCHEMICAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def calc_specific_capacitance(
    discharge_time: float,
    current_or_density: float,
    delta_v: float,
    data_type: str,
    mass_g: float = 1.0,
) -> float:
    """
    Specific capacitance from GCD discharge.

    Law / Equation
    --------------
    From Q = I · t and C = Q / ΔV:

        C_s = (I · t_d) / (m · ΔV)          [F/g]   if data_type = "current"
        C_s =  J · t_d  / ΔV                [F/g]   if data_type = "current_density"

    where
        I      = applied current (A)
        J      = applied current density (A/g)
        t_d    = discharge time (s)
        m      = active mass (g)
        ΔV     = usable potential window during discharge (V)

    Reference: Conway, B.E. (1999) Electrochemical Supercapacitors.
               Stoller & Ruoff, Energy Environ. Sci., 2010.
    """
    if delta_v <= 0:
        return np.nan
    if data_type == "current_density":
        # J already in A/g  → C_s = J·t / ΔV
        return (current_or_density * discharge_time) / delta_v   # F/g
    else:
        # I in A → C_s = I·t / (m·ΔV)
        if mass_g <= 0:
            return np.nan
        return (current_or_density * discharge_time) / (mass_g * delta_v)  # F/g


def calc_energy_density(
    specific_capacitance_F_g: float,
    delta_v: float,
) -> float:
    """
    Gravimetric energy density.

    Law / Equation
    --------------
    E = ½ · C_s · ΔV²        [J/g → convert to Wh/kg: × 1000/3600]

    Reference: Ragone, D.V. (1968). SAE Technical Paper 680453.
    """
    if not np.isfinite(specific_capacitance_F_g) or delta_v <= 0:
        return np.nan
    # Wh/kg = F/g · V² / (2 × 3.6)
    return (specific_capacitance_F_g * delta_v**2) / (2 * 3.6)


def calc_power_density(
    energy_density_Wh_kg: float,
    discharge_time_s: float,
) -> float:
    """
    Average power density over the discharge period.

    Law / Equation
    --------------
    P = E / t_d         [W/kg]

    where E is in Wh/kg and t_d is in hours → P [W/kg] = E / (t_d / 3600)

    Reference: Ragone, D.V. (1968).
    """
    if not np.isfinite(energy_density_Wh_kg) or discharge_time_s <= 0:
        return np.nan
    t_h = discharge_time_s / 3600.0
    return energy_density_Wh_kg / t_h   # W/kg


def calc_coulombic_efficiency(
    charge_time: float,
    discharge_time: float,
) -> float:
    """
    Coulombic efficiency (charge efficiency).

    Law / Equation
    --------------
    η_CE = Q_discharge / Q_charge × 100 %
         = t_discharge / t_charge × 100 %

    (valid for constant-current GCD where Q = I·t and I is the same
    magnitude for charge and discharge)

    Reference: Linden & Reddy (2010) Handbook of Batteries, 4th ed.
    """
    if charge_time <= 0:
        return np.nan
    return (discharge_time / charge_time) * 100.0


def calc_ir_drop_resistance(
    ir_drop_V: float,
    current_A: float,
) -> float:
    """
    Internal (ohmic) resistance from IR drop.

    Law / Equation
    --------------
    R_ESR = ΔV_IR / (2 · I)

    The factor of 2 accounts for the full current reversal.

    Reference: Miller, J.R. (2006) Electrochim. Acta, 52, 1703.
    """
    if current_A <= 0:
        return np.nan
    return ir_drop_V / (2.0 * current_A)


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — NORMAL GCD ANALYSIS (per current density)
# ═══════════════════════════════════════════════════════════════

def run_normal_gcd_analysis(
    df: pd.DataFrame,
    col_info: dict,
    data_type: str,
    mass_g: float = 1.0,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    extract_cycle_number: Optional[int] = None,
) -> dict:
    """
    Full normal GCD analysis: per current/current-density column.

    For each column:
      1. Detect all cycles
      2. Select the best cycle (longest discharge) unless a specific
         cycle number is requested
      3. Compute Cs, E, P, η_CE, IR drop, ESR

    Returns a results dict with DataFrames and per-column cycle lists.
    """
    time_col = col_info["time_col"]
    cd_columns = col_info["cd_columns"]

    time_arr = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

    results_rows = []
    all_cycles_by_col = {}          # column label → list of cycle dicts
    best_cycle_by_col = {}          # column label → best cycle dict

    for info in cd_columns:
        col      = info["col"]
        cd_value = info["value"]    # numeric A/g or A
        label    = info["label"]

        voltage_arr = pd.to_numeric(df[col], errors="coerce").to_numpy()

        # Detect cycles
        cycles = detect_cycles(time_arr, voltage_arr, v_min=v_min, v_max=v_max)

        if not cycles:
            results_rows.append({
                "Label": label, "Value (A/g or A)": cd_value,
                "N Cycles Detected": 0,
                "Cycle Used": "N/A",
                "Specific Capacitance (F/g)": np.nan,
                "Energy Density (Wh/kg)": np.nan,
                "Power Density (W/kg)": np.nan,
                "Coulombic Efficiency (%)": np.nan,
                "Charge Time (s)": np.nan,
                "Discharge Time (s)": np.nan,
                "ΔV_discharge (V)": np.nan,
                "V_top (V)": np.nan,
                "V_bottom (V)": np.nan,
                "IR Drop (V)": np.nan,
                "ESR (Ω)": np.nan,
            })
            all_cycles_by_col[label] = []
            best_cycle_by_col[label] = None
            continue

        # Choose cycle
        if extract_cycle_number is not None:
            # User asked for a specific cycle index (1-based)
            idx = extract_cycle_number - 1
            chosen = cycles[idx] if 0 <= idx < len(cycles) else select_best_cycle(cycles)
            cycle_used = extract_cycle_number if 0 <= idx < len(cycles) else "best"
        else:
            chosen = select_best_cycle(cycles)
            cycle_used = chosen["cycle_index"]

        disch = chosen["discharge"]
        chg   = chosen["charge"]

        # ΔV during discharge (actual, not nominal)
        # We use the range actually traversed in the discharge half-cycle
        delta_v = float(np.max(disch["voltage"]) - np.min(disch["voltage"]))
        t_d     = chosen["discharge_time"]
        t_c     = chosen["charge_time"]
        ir_drop = chosen["ir_drop"]

        # If data_type is current, convert to A/g using mass for Cs calc
        # But for ESR we always need current in A
        if data_type == "current_density":
            current_for_esr = cd_value * mass_g if np.isfinite(cd_value) else np.nan
        else:
            current_for_esr = cd_value

        cs   = calc_specific_capacitance(t_d, cd_value, delta_v, data_type, mass_g)
        E    = calc_energy_density(cs, delta_v)
        P    = calc_power_density(E, t_d)
        eta  = calc_coulombic_efficiency(t_c, t_d)
        esr  = calc_ir_drop_resistance(ir_drop, current_for_esr) if data_type == "current" else np.nan

        results_rows.append({
            "Label": label,
            "Value (A/g or A)": cd_value,
            "N Cycles Detected": len(cycles),
            "Cycle Used": cycle_used,
            "Specific Capacitance (F/g)": cs,
            "Energy Density (Wh/kg)": E,
            "Power Density (W/kg)": P,
            "Coulombic Efficiency (%)": eta,
            "Charge Time (s)": t_c,
            "Discharge Time (s)": t_d,
            "ΔV_discharge (V)": delta_v,
            "V_top (V)": chosen["v_top"],
            "V_bottom (V)": chosen["v_bottom"],
            "IR Drop (V)": ir_drop,
            "ESR (Ω)": esr,
        })

        all_cycles_by_col[label] = cycles
        best_cycle_by_col[label] = chosen

    summary_df = pd.DataFrame(results_rows)

    return {
        "mode": "normal",
        "summary": summary_df,
        "all_cycles": all_cycles_by_col,
        "best_cycles": best_cycle_by_col,
        "data_type": data_type,
        "mass_g": mass_g,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — STABILITY TEST ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_stability_gcd_analysis(
    df: pd.DataFrame,
    col_info: dict,
    data_type: str,
    mass_g: float = 1.0,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    extract_cycle_number: Optional[int] = None,
    stability_column_label: Optional[str] = None,
) -> dict:
    """
    Stability test GCD analysis.

    For stability tests the data column usually represents a single
    current/current density repeated thousands of times (up to 20,000+).

    For each detected cycle we compute Cs, E, P, η_CE and track
    retention relative to the first valid cycle.

    If multiple columns are present, the user can specify which one
    to analyse (stability_column_label); otherwise the first column is used.

    Parameters
    ----------
    extract_cycle_number : if set, also return a DataFrame with the
                           raw data for that specific cycle number
    """
    time_col  = col_info["time_col"]
    cd_columns = col_info["cd_columns"]

    # Pick the column to analyse
    if stability_column_label:
        matching = [c for c in cd_columns if c["label"] == stability_column_label
                    or c["col"] == stability_column_label]
        col_info_sel = matching[0] if matching else cd_columns[0]
    else:
        col_info_sel = cd_columns[0]

    col      = col_info_sel["col"]
    cd_value = col_info_sel["value"]
    label    = col_info_sel["label"]

    time_arr    = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    voltage_arr = pd.to_numeric(df[col], errors="coerce").to_numpy()

    cycles = detect_cycles(time_arr, voltage_arr, v_min=v_min, v_max=v_max)

    if not cycles:
        return {
            "mode": "stability",
            "label": label,
            "per_cycle": pd.DataFrame(),
            "extracted_cycle": None,
            "data_type": data_type,
            "mass_g": mass_g,
            "n_cycles": 0,
        }

    # Per-cycle metrics
    rows = []
    first_cs = None

    for cyc in cycles:
        disch   = cyc["discharge"]
        chg     = cyc["charge"]
        delta_v = float(np.max(disch["voltage"]) - np.min(disch["voltage"]))
        t_d     = cyc["discharge_time"]
        t_c     = cyc["charge_time"]
        ir_drop = cyc["ir_drop"]

        if data_type == "current_density":
            current_for_esr = cd_value * mass_g if np.isfinite(cd_value) else np.nan
        else:
            current_for_esr = cd_value

        cs  = calc_specific_capacitance(t_d, cd_value, delta_v, data_type, mass_g)
        E   = calc_energy_density(cs, delta_v)
        P   = calc_power_density(E, t_d)
        eta = calc_coulombic_efficiency(t_c, t_d)
        esr = calc_ir_drop_resistance(ir_drop, current_for_esr) if data_type == "current" else np.nan

        if first_cs is None and np.isfinite(cs):
            first_cs = cs

        retention = (cs / first_cs * 100.0) if (first_cs and np.isfinite(cs)) else np.nan

        rows.append({
            "Cycle": cyc["cycle_index"],
            "Charge Time (s)": t_c,
            "Discharge Time (s)": t_d,
            "ΔV_discharge (V)": delta_v,
            "Specific Capacitance (F/g)": cs,
            "Capacitance Retention (%)": retention,
            "Energy Density (Wh/kg)": E,
            "Power Density (W/kg)": P,
            "Coulombic Efficiency (%)": eta,
            "IR Drop (V)": ir_drop,
            "ESR (Ω)": esr,
            "V_top (V)": cyc["v_top"],
            "V_bottom (V)": cyc["v_bottom"],
        })

    per_cycle_df = pd.DataFrame(rows)

    # Extract specific cycle raw data if requested
    extracted_cycle = None
    if extract_cycle_number is not None:
        idx = extract_cycle_number - 1
        if 0 <= idx < len(cycles):
            cyc = cycles[idx]
            t_ch = cyc["charge"]["time"]
            v_ch = cyc["charge"]["voltage"]
            t_dc = cyc["discharge"]["time"]
            v_dc = cyc["discharge"]["voltage"]
            t_all = np.concatenate([t_ch, t_dc[1:]])
            v_all = np.concatenate([v_ch, v_dc[1:]])
            extracted_cycle = pd.DataFrame({
                "Time (s)": t_all,
                "Potential (V)": v_all,
                "Phase": (
                    ["Charge"] * len(t_ch) +
                    ["Discharge"] * (len(t_dc) - 1)
                ),
            })

    return {
        "mode": "stability",
        "label": label,
        "per_cycle": per_cycle_df,
        "extracted_cycle": extracted_cycle,
        "data_type": data_type,
        "mass_g": mass_g,
        "n_cycles": len(cycles),
        "first_cs": first_cs,
        "all_cycles_raw": cycles,   # raw list for plotting
    }
