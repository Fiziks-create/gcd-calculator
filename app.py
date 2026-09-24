"""
gcd_app.py  —  GCD Electrochemical Analyzer — Gradio Interface
===============================================================
Modes
-----
A. Normal GCD   — per current/current-density: Cs, E, P, η_CE
B. Stability    — cycle-by-cycle retention, up to 20 000+ cycles,
                  with optional extraction of any specific cycle

Input format (wide table)
--------------------------
Column 0   : Time (s)
Column 1+  : Potential (V) at each current or current density
Header row : current/density values  e.g. "1 A/g", "2", "5mA/cm2"
"""
from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import gradio as gr

from backend import (
    load_gcd_file,
    parse_gcd_columns,
    detect_cycles,
    select_best_cycle,
    run_normal_gcd_analysis,
    run_stability_gcd_analysis,
    calc_specific_capacitance,
    calc_energy_density,
    calc_power_density,
)

# ── workspace ─────────────────────────────────────────────────
MAX_FILE_SIZE = "50mb"   # stability files can be large
TEMP_ROOT = Path(tempfile.gettempdir()) / "gcd_calculator_work"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _cleanup_later(path: Path, delay: int = 1800):
    def _rm():
        shutil.rmtree(path, ignore_errors=True)
    t = threading.Timer(delay, _rm)
    t.daemon = True
    t.start()


# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS — Normal GCD
# ═══════════════════════════════════════════════════════════════

def _plot_gcd_curves(result: dict, out: Path):
    """
    Plot best-cycle GCD curves (V vs t) for every current/density.
    Each curve is colour-coded. Charge = solid, Discharge = dashed.
    """
    best = result["best_cycles"]
    col_infos = {c["label"]: c for c in result.get("_cd_columns", [])}

    if not any(v is not None for v in best.values()):
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10
    labels = [lbl for lbl, cyc in best.items() if cyc is not None]
    colors = [cmap(i % 10) for i in range(len(labels))]

    for (lbl, cyc), color in zip(
        [(l, best[l]) for l in labels], colors
    ):
        t_chg = cyc["charge"]["time"] - cyc["charge"]["time"][0]
        t_dis = cyc["discharge"]["time"] - cyc["charge"]["time"][0]
        ax.plot(t_chg, cyc["charge"]["voltage"],
                color=color, linewidth=1.4, label=f"{lbl} (chg)")
        ax.plot(t_dis, cyc["discharge"]["voltage"],
                color=color, linewidth=1.4, linestyle="--",
                label=f"{lbl} (dis)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Potential (V)")
    ax.set_title("GCD Curves — Best Cycle per Current/Density\n"
                 "Solid = Charge  |  Dashed = Discharge")
    ax.grid(True, alpha=0.25)
    if len(labels) <= 8:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_all_cycles_one_col(
    time_arr: np.ndarray,
    voltage_arr: np.ndarray,
    cycles: list,
    label: str,
    out: Path,
):
    """
    Plot ALL detected cycles for a single current/density column.
    Best cycle is highlighted in red.
    """
    if not cycles:
        return

    best = select_best_cycle(cycles)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Full raw trace in light grey
    t0 = time_arr[0]
    ax.plot(time_arr - t0, voltage_arr, color="#cccccc",
            linewidth=0.7, label="Full trace", zorder=1)

    cmap = plt.cm.viridis
    n = len(cycles)
    for i, cyc in enumerate(cycles):
        color = cmap(i / max(n - 1, 1))
        t_c = cyc["charge"]["time"] - t0
        v_c = cyc["charge"]["voltage"]
        t_d = cyc["discharge"]["time"] - t0
        v_d = cyc["discharge"]["voltage"]

        lw = 2.2 if cyc is best else 0.9
        zorder = 3 if cyc is best else 2
        clr = "red" if cyc is best else color
        lbl = f"Cycle {cyc['cycle_index']} ★ BEST" if cyc is best else f"Cycle {cyc['cycle_index']}"
        ax.plot(np.concatenate([t_c, t_d]),
                np.concatenate([v_c, v_d]),
                color=clr, linewidth=lw, label=lbl, zorder=zorder)

    ax.set_xlabel("Time from start (s)")
    ax.set_ylabel("Potential (V)")
    ax.set_title(f"All Cycles — {label}\nRed = Best Cycle (longest discharge)")
    ax.grid(True, alpha=0.2)
    if n <= 12:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_capacitance_vs_cd(summary: pd.DataFrame, data_type: str, out: Path):
    """
    Specific capacitance (F/g) vs current density (or current).
    Classic 'rate capability' plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    d = summary.dropna(subset=["Specific Capacitance (F/g)"]).sort_values("Value (A/g or A)")
    if d.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        fig.savefig(out, dpi=150); plt.close(fig); return

    ax.plot(d["Value (A/g or A)"], d["Specific Capacitance (F/g)"],
            marker="o", color="steelblue", linewidth=1.5)
    ax.set_xlabel("Current Density (A/g)" if data_type == "current_density" else "Current (A)")
    ax.set_ylabel("Specific Capacitance (F/g)")
    ax.set_title("Rate Capability: Specific Capacitance vs Current Density")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_ragone(summary: pd.DataFrame, out: Path):
    """
    Ragone plot: Energy density (Wh/kg) vs Power density (W/kg).
    Law: E = ½·Cs·ΔV², P = E/t_d  [Ragone 1968]
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    d = summary.dropna(subset=["Energy Density (Wh/kg)", "Power Density (W/kg)"])
    d = d.sort_values("Power Density (W/kg)")
    if d.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        fig.savefig(out, dpi=150); plt.close(fig); return

    ax.loglog(d["Power Density (W/kg)"], d["Energy Density (Wh/kg)"],
              marker="o", color="darkorange", linewidth=1.5)
    ax.set_xlabel("Power Density (W/kg)")
    ax.set_ylabel("Energy Density (Wh/kg)")
    ax.set_title("Ragone Plot  (E = ½·Cs·ΔV²,  P = E/t_d)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_coulombic_efficiency(summary: pd.DataFrame, data_type: str, out: Path):
    """
    Coulombic efficiency (%) vs current density.
    η_CE = t_discharge / t_charge × 100 % [constant-current GCD]
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    d = summary.dropna(subset=["Coulombic Efficiency (%)"]).sort_values("Value (A/g or A)")
    if d.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        fig.savefig(out, dpi=150); plt.close(fig); return

    ax.plot(d["Value (A/g or A)"], d["Coulombic Efficiency (%)"],
            marker="s", color="seagreen", linewidth=1.5)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Current Density (A/g)" if data_type == "current_density" else "Current (A)")
    ax.set_ylabel("Coulombic Efficiency (%)")
    ax.set_title("Coulombic Efficiency  (η = t_dis / t_chg × 100 %)")
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_ir_drop(summary: pd.DataFrame, data_type: str, out: Path):
    """
    IR drop (V) and ESR (Ω) vs current density.
    R_ESR = ΔV_IR / (2·I)  [Miller 2006]
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    d = summary.sort_values("Value (A/g or A)")

    x_label = "Current Density (A/g)" if data_type == "current_density" else "Current (A)"

    ax = axes[0]
    dd = d.dropna(subset=["IR Drop (V)"])
    if not dd.empty:
        ax.bar(dd["Label"].astype(str), dd["IR Drop (V)"], color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Label")
    ax.set_ylabel("IR Drop (V)")
    ax.set_title("IR Drop vs Current/Density")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    dd = d.dropna(subset=["ESR (Ω)"])
    if not dd.empty:
        ax.bar(dd["Label"].astype(str), dd["ESR (Ω)"], color="#8e44ad", alpha=0.8)
    ax.set_xlabel("Label")
    ax.set_ylabel("ESR (Ω)")
    ax.set_title("ESR vs Current/Density  (R = ΔV_IR / 2I)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS — Stability
# ═══════════════════════════════════════════════════════════════

def _plot_stability_retention(per_cycle: pd.DataFrame, label: str, out: Path):
    """
    Capacitance retention (%) vs cycle number.
    Standard plot for electrochemical stability tests.
    Also overlays coulombic efficiency.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    d = per_cycle.dropna(subset=["Capacitance Retention (%)"])
    if d.empty:
        ax1.text(0.5, 0.5, "No retention data", ha="center", va="center")
        fig.savefig(out, dpi=150); plt.close(fig); return

    ax1.plot(d["Cycle"], d["Capacitance Retention (%)"],
             color="steelblue", linewidth=1.2, label="Capacitance Retention (%)")
    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Capacitance Retention (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, 115)
    ax1.grid(True, alpha=0.2)

    # Coulombic efficiency on secondary axis
    ax2 = ax1.twinx()
    d2 = per_cycle.dropna(subset=["Coulombic Efficiency (%)"])
    if not d2.empty:
        ax2.plot(d2["Cycle"], d2["Coulombic Efficiency (%)"],
                 color="seagreen", linewidth=0.8, alpha=0.7,
                 linestyle="--", label="Coulombic Efficiency (%)")
        ax2.set_ylabel("Coulombic Efficiency (%)", color="seagreen")
        ax2.tick_params(axis="y", labelcolor="seagreen")
        ax2.set_ylim(0, 115)

    ax1.set_title(f"Stability Test — Capacitance Retention\n{label}")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_stability_capacitance(per_cycle: pd.DataFrame, label: str, out: Path):
    """
    Specific capacitance (F/g) vs cycle number — shows absolute fade.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    d = per_cycle.dropna(subset=["Specific Capacitance (F/g)"])
    if d.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out, dpi=150); plt.close(fig); return

    ax.plot(d["Cycle"], d["Specific Capacitance (F/g)"],
            color="darkorange", linewidth=1.2)
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Specific Capacitance (F/g)")
    ax.set_title(f"Specific Capacitance vs Cycle Number\n{label}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_stability_energy_power(per_cycle: pd.DataFrame, label: str, out: Path):
    """
    Energy density and power density over cycles — shows performance evolution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, color, title in [
        (axes[0], "Energy Density (Wh/kg)", "steelblue", "Energy Density vs Cycle"),
        (axes[1], "Power Density (W/kg)",   "tomato",    "Power Density vs Cycle"),
    ]:
        d = per_cycle.dropna(subset=[col])
        if not d.empty:
            ax.plot(d["Cycle"], d[col], color=color, linewidth=1.0)
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel(col)
        ax.set_title(f"{title}\n{label}")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_extracted_cycle(extracted: pd.DataFrame, cycle_num: int,
                          label: str, out: Path):
    """
    Plot the raw voltage–time profile of a single extracted cycle.
    Charge phase in blue, discharge in orange.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    t0 = extracted["Time (s)"].iloc[0]
    chg  = extracted[extracted["Phase"] == "Charge"]
    dis  = extracted[extracted["Phase"] == "Discharge"]

    ax.plot(chg["Time (s)"] - t0, chg["Potential (V)"],
            color="#2196F3", linewidth=1.8, label="Charge")
    ax.plot(dis["Time (s)"] - t0, dis["Potential (V)"],
            color="#FF9800", linewidth=1.8, label="Discharge")

    ax.set_xlabel("Time from cycle start (s)")
    ax.set_ylabel("Potential (V)")
    ax.set_title(f"Extracted Cycle #{cycle_num} — {label}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_discharge_time_vs_cd(summary: pd.DataFrame, data_type: str, out: Path):
    """
    Discharge time (s) vs current density — useful for rate capability insight.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    d = summary.dropna(subset=["Discharge Time (s)"]).sort_values("Value (A/g or A)")
    if d.empty:
        fig.savefig(out, dpi=150); plt.close(fig); return

    ax.bar(d["Label"].astype(str), d["Discharge Time (s)"],
           color="mediumpurple", alpha=0.85)
    ax.set_xlabel("Current / Density")
    ax.set_ylabel("Discharge Time (s)")
    ax.set_title("Discharge Time per Current/Density")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_multi_panel_gcd(result: dict, out: Path):
    """
    Multi-panel summary: one panel per current/density showing the
    best charge–discharge cycle with key metrics annotated.
    """
    best   = result["best_cycles"]
    summ   = result["summary"]
    labels = [lbl for lbl, cyc in best.items() if cyc is not None]
    n      = len(labels)
    if n == 0:
        return

    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                              figsize=(4.5 * cols, 3.5 * rows),
                              squeeze=False)

    row_map = summ.set_index("Label")

    for idx, lbl in enumerate(labels):
        ax   = axes[idx // cols][idx % cols]
        cyc  = best[lbl]

        t0   = cyc["charge"]["time"][0]
        t_c  = cyc["charge"]["time"] - t0
        v_c  = cyc["charge"]["voltage"]
        t_d  = cyc["discharge"]["time"] - t0
        v_d  = cyc["discharge"]["voltage"]

        ax.plot(t_c, v_c, color="#2196F3", linewidth=1.4, label="Chg")
        ax.plot(t_d, v_d, color="#FF9800", linewidth=1.4, linestyle="--", label="Dis")
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.set_ylabel("V (V)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

        # Annotate Cs and η on the plot
        if lbl in row_map.index:
            cs  = row_map.loc[lbl, "Specific Capacitance (F/g)"]
            eta = row_map.loc[lbl, "Coulombic Efficiency (%)"]
            txt = ""
            if np.isfinite(cs):
                txt += f"Cs={cs:.1f} F/g\n"
            if np.isfinite(eta):
                txt += f"η={eta:.1f}%"
            if txt:
                ax.text(0.97, 0.05, txt.strip(), transform=ax.transAxes,
                        fontsize=6.5, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", alpha=0.7))

        if idx == 0:
            ax.legend(fontsize=6)

    # Hide unused panels
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("GCD Best-Cycle Summary per Current/Density\n"
                 "Blue = Charge  |  Orange dashed = Discharge",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# EXPORT PACKAGE
# ═══════════════════════════════════════════════════════════════

def _make_normal_export(result: dict, work: Path, original_name: str,
                        config: dict, df_raw: pd.DataFrame,
                        col_info: dict):
    dirs = {k: work / k for k in ("Results", "Plots", "Metadata", "Validation")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── CSV / Excel ──────────────────────────────────────────
    summ = result["summary"]
    summ.to_csv(dirs["Results"] / "GCD_Summary.csv", index=False)

    # Per-column cycle tables
    all_cyc_rows = []
    for lbl, cycles in result["all_cycles"].items():
        for cyc in cycles:
            all_cyc_rows.append({
                "Label": lbl,
                "Cycle": cyc["cycle_index"],
                "Charge Time (s)": cyc["charge_time"],
                "Discharge Time (s)": cyc["discharge_time"],
                "V_top (V)": cyc["v_top"],
                "V_bottom (V)": cyc["v_bottom"],
                "IR Drop (V)": cyc["ir_drop"],
            })
    pd.DataFrame(all_cyc_rows).to_csv(dirs["Results"] / "All_Cycles_Detected.csv", index=False)

    # Excel workbook
    wb = work / "GCD_Analysis_Results.xlsx"
    with pd.ExcelWriter(wb, engine="openpyxl") as writer:
        summ.to_excel(writer, sheet_name="GCD_Summary", index=False)
        pd.DataFrame(all_cyc_rows).to_excel(writer, sheet_name="All_Cycles", index=False)

    # ── Plots ────────────────────────────────────────────────
    # Inject cd_columns for the curve plotter
    result["_cd_columns"] = col_info["cd_columns"]

    time_arr = pd.to_numeric(df_raw[col_info["time_col"]], errors="coerce").to_numpy()

    _plot_gcd_curves(result, dirs["Plots"] / "GCD_Curves.png")
    _plot_multi_panel_gcd(result, dirs["Plots"] / "GCD_Multi_Panel.png")
    _plot_capacitance_vs_cd(summ, result["data_type"],
                             dirs["Plots"] / "Capacitance_vs_CD.png")
    _plot_ragone(summ, dirs["Plots"] / "Ragone_Plot.png")
    _plot_coulombic_efficiency(summ, result["data_type"],
                                dirs["Plots"] / "Coulombic_Efficiency.png")
    _plot_ir_drop(summ, result["data_type"],
                   dirs["Plots"] / "IR_Drop_ESR.png")
    _plot_discharge_time_vs_cd(summ, result["data_type"],
                                dirs["Plots"] / "Discharge_Time.png")

    # Per-column all-cycles plots
    for info in col_info["cd_columns"]:
        lbl = info["label"]
        cycles = result["all_cycles"].get(lbl, [])
        if cycles:
            voltage_arr = pd.to_numeric(df_raw[info["col"]], errors="coerce").to_numpy()
            safe_lbl = re.sub(r"[^\w]", "_", lbl)
            _plot_all_cycles_one_col(
                time_arr, voltage_arr, cycles, lbl,
                dirs["Plots"] / f"All_Cycles_{safe_lbl}.png"
            )

    # ── Validation ───────────────────────────────────────────
    status_rows = [
        {"Item": "File",        "Value": original_name},
        {"Item": "Mode",        "Value": "Normal GCD"},
        {"Item": "Data type",   "Value": result["data_type"]},
        {"Item": "Mass (g)",    "Value": result["mass_g"]},
        {"Item": "Columns",     "Value": len(col_info["cd_columns"])},
    ]
    for k, v in config.items():
        status_rows.append({"Item": k, "Value": v})
    pd.DataFrame(status_rows).to_csv(dirs["Validation"] / "Analysis_Status.csv", index=False)

    # ── Metadata ─────────────────────────────────────────────
    pd.DataFrame([
        {"Field": "Generated UTC", "Value": datetime.now(timezone.utc).isoformat()},
        {"Field": "File", "Value": original_name},
        *[{"Field": k, "Value": v} for k, v in config.items()],
    ]).to_csv(dirs["Metadata"] / "Metadata.csv", index=False)

    # ── Report ───────────────────────────────────────────────
    lines = [
        "GCD ELECTROCHEMICAL ANALYSIS REPORT — NORMAL MODE",
        "=" * 60,
        f"File          : {original_name}",
        f"Generated UTC : {datetime.now(timezone.utc).isoformat()}",
        f"Data type     : {result['data_type']}",
        f"Mass (g)      : {result['mass_g']}",
        "",
        "SUMMARY TABLE",
        "-" * 60,
        summ.to_string(index=False),
        "",
        "END OF REPORT",
    ]
    (work / "Analysis_Report.txt").write_text("\n".join(lines), encoding="utf-8")

    # ── ZIP ──────────────────────────────────────────────────
    zip_path = work / "GCD_Analysis_Output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in work.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, arcname=str(p.relative_to(work)))

    return zip_path


def _make_stability_export(result: dict, work: Path, original_name: str,
                            config: dict):
    dirs = {k: work / k for k in ("Results", "Plots", "Metadata", "Validation")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    per_cycle = result["per_cycle"]
    label     = result["label"]
    n_cycles  = result["n_cycles"]

    per_cycle.to_csv(dirs["Results"] / "Stability_Per_Cycle.csv", index=False)

    if result.get("extracted_cycle") is not None:
        result["extracted_cycle"].to_csv(
            dirs["Results"] / "Extracted_Cycle.csv", index=False
        )

    wb = work / "GCD_Stability_Results.xlsx"
    with pd.ExcelWriter(wb, engine="openpyxl") as writer:
        per_cycle.to_excel(writer, sheet_name="Per_Cycle", index=False)
        if result.get("extracted_cycle") is not None:
            result["extracted_cycle"].to_excel(
                writer, sheet_name="Extracted_Cycle", index=False
            )

    _plot_stability_retention(per_cycle, label,
                               dirs["Plots"] / "Stability_Retention.png")
    _plot_stability_capacitance(per_cycle, label,
                                 dirs["Plots"] / "Stability_Capacitance.png")
    _plot_stability_energy_power(per_cycle, label,
                                  dirs["Plots"] / "Stability_Energy_Power.png")

    extract_num = config.get("extract_cycle_number")
    if result.get("extracted_cycle") is not None and extract_num:
        _plot_extracted_cycle(result["extracted_cycle"], int(extract_num),
                               label, dirs["Plots"] / "Extracted_Cycle.png")

    pd.DataFrame([
        {"Item": "File",        "Value": original_name},
        {"Item": "Mode",        "Value": "Stability GCD"},
        {"Item": "Column",      "Value": label},
        {"Item": "N Cycles",    "Value": n_cycles},
        {"Item": "Data type",   "Value": result["data_type"]},
        {"Item": "Mass (g)",    "Value": result["mass_g"]},
        *[{"Item": k, "Value": v} for k, v in config.items()],
    ]).to_csv(dirs["Validation"] / "Analysis_Status.csv", index=False)

    pd.DataFrame([
        {"Field": "Generated UTC", "Value": datetime.now(timezone.utc).isoformat()},
        {"Field": "File", "Value": original_name},
        *[{"Field": k, "Value": v} for k, v in config.items()],
    ]).to_csv(dirs["Metadata"] / "Metadata.csv", index=False)

    first_cs  = result.get("first_cs", np.nan)
    last_cs_r = per_cycle["Specific Capacitance (F/g)"].dropna()
    last_cs   = float(last_cs_r.iloc[-1]) if not last_cs_r.empty else np.nan
    final_ret = float(per_cycle["Capacitance Retention (%)"].dropna().iloc[-1]) \
        if not per_cycle["Capacitance Retention (%)"].dropna().empty else np.nan

    lines = [
        "GCD ELECTROCHEMICAL ANALYSIS REPORT — STABILITY MODE",
        "=" * 60,
        f"File           : {original_name}",
        f"Generated UTC  : {datetime.now(timezone.utc).isoformat()}",
        f"Column         : {label}",
        f"Total cycles   : {n_cycles}",
        f"Data type      : {result['data_type']}",
        f"Mass (g)       : {result['mass_g']}",
        "",
        "STABILITY SUMMARY",
        "-" * 60,
        f"  Initial Cs   : {first_cs:.4f} F/g" if np.isfinite(first_cs) else "  Initial Cs   : N/A",
        f"  Final Cs     : {last_cs:.4f} F/g"  if np.isfinite(last_cs)  else "  Final Cs     : N/A",
        f"  Final Retention: {final_ret:.2f} %" if np.isfinite(final_ret) else "  Final Retention: N/A",
        "",
        "END OF REPORT",
    ]
    (work / "Analysis_Report.txt").write_text("\n".join(lines), encoding="utf-8")

    zip_path = work / "GCD_Stability_Output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in work.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, arcname=str(p.relative_to(work)))

    return zip_path


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════




def run_normal_analysis(
    file_path,
    data_type,
    mass_mg,
    v_min_input,
    v_max_input,
    extract_cycle_number_input,
):
    """Called by the Normal GCD Run button."""
    if not file_path:
        raise gr.Error("Please upload a GCD data file.")

    work = TEMP_ROOT / f"job_{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=False)

    try:
        src  = Path(file_path)
        dest = work / src.name
        shutil.copy2(src, dest)

        df       = load_gcd_file(dest)
        col_info = parse_gcd_columns(df, data_type)

        mass_g = float(mass_mg) / 1000.0 if data_type == "current" else 1.0

        v_min = float(v_min_input) if str(v_min_input).strip() not in ("", "None") else None
        v_max = float(v_max_input) if str(v_max_input).strip() not in ("", "None") else None

        ext_cyc = None
        if str(extract_cycle_number_input).strip() not in ("", "0", "None"):
            try:
                ext_cyc = int(extract_cycle_number_input)
            except Exception:
                pass

        config = {
            "data_type":             data_type,
            "mass_mg":               mass_mg if data_type == "current" else "N/A",
            "v_min":                 v_min,
            "v_max":                 v_max,
            "extract_cycle_number":  ext_cyc,
        }

        result = run_normal_gcd_analysis(
            df, col_info, data_type,
            mass_g   = mass_g,
            v_min    = v_min,
            v_max    = v_max,
            extract_cycle_number = ext_cyc,
        )

        zip_path = _make_normal_export(result, work, src.name, config, df, col_info)

        # ── Status text ──────────────────────────────────────
        summ = result["summary"]
        n_ok = summ["Specific Capacitance (F/g)"].notna().sum()

        status = (
            "✅  NORMAL GCD ANALYSIS COMPLETE\n"
            f"\nFile          : {src.name}"
            f"\nData type     : {data_type}"
            f"\nColumns found : {len(col_info['cd_columns'])}"
            f"\nValid columns : {n_ok}"
        )
        if data_type == "current":
            status += f"\nMass used     : {mass_g*1000:.2f} mg"

        best_row = summ.loc[summ["Specific Capacitance (F/g)"] ==
                             summ["Specific Capacitance (F/g)"].max()]
        if not best_row.empty:
            br = best_row.iloc[0]
            status += (
                f"\n\nHighest Cs    : {br['Specific Capacitance (F/g)']:.2f} F/g"
                f"  @ {br['Label']}"
                f"\nCoulombic eff : {br['Coulombic Efficiency (%)']:.1f} %"
                f"\nEnergy density: {br['Energy Density (Wh/kg)']:.4f} Wh/kg"
                f"\nPower density : {br['Power Density (W/kg)']:.2f} W/kg"
            )

        status += (
            "\n\nZIP contains: GCD curves, multi-panel, Ragone, "
            "rate capability, coulombic efficiency, IR drop/ESR, "
            "per-column all-cycle plots, CSV + Excel results."
        )

        _cleanup_later(work)

        return (
            status,          # 0  status
            summ,            # 1  summary table
            str(zip_path),   # 2  download
        )

    except Exception as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise gr.Error(f"{type(exc).__name__}: {exc}")


def run_stability_analysis(
    file_path,
    data_type,
    mass_mg,
    stability_column_label,
    v_min_input,
    v_max_input,
    extract_cycle_number_input,
):
    """Called by the Stability GCD Run button."""
    if not file_path:
        raise gr.Error("Please upload a GCD data file.")

    work = TEMP_ROOT / f"job_{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=False)

    try:
        src  = Path(file_path)
        dest = work / src.name
        shutil.copy2(src, dest)

        df       = load_gcd_file(dest)
        col_info = parse_gcd_columns(df, data_type)

        mass_g = float(mass_mg) / 1000.0 if data_type == "current" else 1.0

        v_min = float(v_min_input) if str(v_min_input).strip() not in ("", "None") else None
        v_max = float(v_max_input) if str(v_max_input).strip() not in ("", "None") else None

        col_label = str(stability_column_label).strip() or None

        ext_cyc = None
        if str(extract_cycle_number_input).strip() not in ("", "0", "None"):
            try:
                ext_cyc = int(extract_cycle_number_input)
            except Exception:
                pass

        config = {
            "data_type":             data_type,
            "mass_mg":               mass_mg if data_type == "current" else "N/A",
            "stability_column":      col_label or "auto (first column)",
            "v_min":                 v_min,
            "v_max":                 v_max,
            "extract_cycle_number":  ext_cyc,
        }

        result = run_stability_gcd_analysis(
            df, col_info, data_type,
            mass_g                  = mass_g,
            v_min                   = v_min,
            v_max                   = v_max,
            extract_cycle_number    = ext_cyc,
            stability_column_label  = col_label,
        )

        zip_path = _make_stability_export(result, work, src.name, config)

        per_cycle = result["per_cycle"]
        n_cyc     = result["n_cycles"]
        first_cs  = result.get("first_cs", np.nan)
        final_ret = float(per_cycle["Capacitance Retention (%)"].dropna().iloc[-1]) \
            if not per_cycle.empty and per_cycle["Capacitance Retention (%)"].notna().any() \
            else np.nan

        status = (
            "✅  STABILITY GCD ANALYSIS COMPLETE\n"
            f"\nFile          : {src.name}"
            f"\nColumn        : {result['label']}"
            f"\nData type     : {data_type}"
            f"\nCycles found  : {n_cyc}"
        )
        if data_type == "current":
            status += f"\nMass used     : {mass_g*1000:.2f} mg"
        if np.isfinite(first_cs):
            status += f"\nInitial Cs    : {first_cs:.4f} F/g"
        if np.isfinite(final_ret):
            status += f"\nFinal retention: {final_ret:.2f} %"
        if ext_cyc:
            if result.get("extracted_cycle") is not None:
                status += f"\nExtracted cycle #{ext_cyc}: ✅ included in ZIP"
            else:
                status += f"\nExtracted cycle #{ext_cyc}: ⚠ not found (only {n_cyc} cycles)"

        _cleanup_later(work)

        extracted_df = result.get("extracted_cycle") or pd.DataFrame(
            columns=["Time (s)", "Potential (V)", "Phase"]
        )

        return (
            status,          # 0  status
            per_cycle,       # 1  per-cycle table
            extracted_df,    # 2  extracted cycle table
            str(zip_path),   # 3  download
        )

    except Exception as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise gr.Error(f"{type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════

def _toggle_mass_input(data_type: str):
    """Show mass input only when current (not current density) is selected."""
    return gr.update(visible=(data_type == "current"))


def _show_normal_tab(choice: str):
    return gr.update(visible=(choice == "Normal GCD"))


def _show_stability_tab(choice: str):
    return gr.update(visible=(choice == "Stability Test"))


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

INTRO_MD = """
# 🔋 GCD Electrochemical Analyzer

**Galvanostatic Charge-Discharge (GCD) analysis** — two modes:

| Mode | Use when |
|------|----------|
| **Normal GCD** | Multiple current/density columns; rate-capability study |
| **Stability Test** | Thousands of cycles at one current/density; retention tracking |

### Input File Format
```
Time (s) │ 1 A/g  │  2 A/g  │  5 A/g  │ 10 A/g
──────────┼────────┼─────────┼─────────┼───────
0.00      │ 0.001  │  0.001  │  0.002  │  0.002
0.05      │ 0.045  │  0.041  │  0.038  │  ...
...
```
- **Column 0**: Time (seconds)
- **Column 1+**: Potential (V) at each current or current density
- Headers: numbers like `1`, `2.5`, or with units `1 A/g`, `5 mA`, `2 mA/cm²`

🔒 Files are processed in temporary storage and deleted after 30 min.
"""

with gr.Blocks(title="GCD Electrochemical Analyzer") as demo:

    gr.Markdown(INTRO_MD)

    # ── Mode selector ──────────────────────────────────────────
    with gr.Row():
        mode_selector = gr.Radio(
            ["Normal GCD", "Stability Test"],
            value="Normal GCD",
            label="🔀 Analysis Mode",
            info="Select mode before uploading. "
                 "Normal = rate capability; Stability = cycle-life test.",
        )

    gr.Markdown("---")

    # ── Shared settings (always visible) ──────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Data Upload & Settings")
            gcd_file = gr.File(
                label="Upload GCD Data (.csv / .xlsx / .xls)",
                file_types=[".csv", ".xlsx", ".xls"],
                type="filepath",
            )

            data_type = gr.Radio(
                ["current_density", "current"],
                value="current_density",
                label="Input Data Type",
                info=(
                    "current_density (A/g, mA/cm², etc.) → mass NOT needed for Cs.\n"
                    "current (A, mA) → mass IS needed; enter it below."
                ),
            )

            # Mass input — only visible when data_type = current
            mass_mg_input = gr.Number(
                value=5.0,
                label="Active Material Mass (mg)",
                info="Used only when data type = 'current'. Ignored for current_density.",
                visible=False,
            )

            gr.Markdown("#### Optional: Potential Window")
            with gr.Row():
                v_min_input = gr.Number(
                    value=None, label="V_min (V)",
                    info="Leave blank to auto-detect from data",
                )
                v_max_input = gr.Number(
                    value=None, label="V_max (V)",
                    info="Leave blank to auto-detect from data",
                )

        with gr.Column(scale=1):
            gr.Markdown("### ℹ️ File Format Reminder")
            gr.Markdown("""
**Wide format** (Time in column 0, one potential column per current/density):

| Time (s) | 1 A/g | 2 A/g | 5 A/g |
|----------|-------|-------|-------|
| 0.0      | 0.001 | 0.001 | 0.002 |
| 0.1      | 0.08  | 0.07  | 0.06  |
| ...      | ...   | ...   | ...   |

**Supported header formats:**
- Bare numbers: `1`, `2`, `5`
- With unit: `1 A/g`, `2 mA/cm2`, `5 mA`, `0.5 A`
- Mixed columns: app will parse what it can

**Cycle detection** is automatic — cells that don't reach the full window
are handled correctly. The cycle with the **longest discharge time** is
selected for metrics calculation.
""")

    # ── Normal GCD panel ──────────────────────────────────────
    with gr.Column(visible=True) as normal_panel:
        gr.Markdown("---")
        gr.Markdown("## 📊 Normal GCD — Rate Capability Analysis")
        gr.Markdown(
            "Calculates **Cs, Energy density, Power density, Coulombic efficiency, "
            "IR drop, ESR** for each current/density column. "
            "Best cycle (longest discharge) is auto-selected per column."
        )

        with gr.Row():
            with gr.Column(scale=1):
                extract_cycle_normal = gr.Number(
                    value=None,
                    label="Extract Specific Cycle (optional)",
                    info="If set, this cycle number is used instead of the "
                         "best (longest discharge) cycle. Leave blank for auto-best.",
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "**Scientific equations used:**\n"
                    "- Cs = J·t_d / ΔV  (F/g)\n"
                    "- E  = ½·Cs·ΔV² / 3.6  (Wh/kg)\n"
                    "- P  = E / t_d  (W/kg)\n"
                    "- η  = t_dis / t_chg × 100 %\n"
                    "- R_ESR = ΔV_IR / (2·I)  (Ω)"
                )

        run_normal_btn = gr.Button("▶ Run Normal GCD Analysis", variant="primary", size="lg")
        normal_status  = gr.Textbox(label="Status", lines=10, interactive=False)

        with gr.Tabs():
            with gr.Tab("📋 Results Summary"):
                normal_summary_output = gr.Dataframe(
                    interactive=False,
                    label="Per Current/Density Results — Cs, E, P, η, IR Drop, ESR",
                )
            with gr.Tab("📦 Download"):
                normal_download = gr.File(
                    label="📦 Download Complete Analysis ZIP"
                )

    # ── Stability Test panel ───────────────────────────────────
    with gr.Column(visible=False) as stability_panel:
        gr.Markdown("---")
        gr.Markdown("## 🔄 Stability Test — Cycle-Life Analysis")
        gr.Markdown(
            "Tracks **capacitance retention** and **coulombic efficiency** "
            "cycle by cycle. Supports up to 20,000+ cycles. "
            "Optional: extract any specific cycle's raw data."
        )

        with gr.Row():
            with gr.Column(scale=1):
                stability_col_label = gr.Textbox(
                    value="",
                    label="Column to Analyse (optional)",
                    info="Enter the exact column header (e.g. '1 A/g'). "
                         "Leave blank to use the first data column automatically.",
                    placeholder="e.g. 1 A/g  or  1  or  leave blank",
                )
                extract_cycle_stability = gr.Number(
                    value=None,
                    label="Extract Specific Cycle Number (optional)",
                    info="Set to any cycle number (1-based) to export its raw "
                         "voltage-time data. Useful for inspecting a single cycle "
                         "at the start, middle, or end of a long test.",
                )

            with gr.Column(scale=1):
                gr.Markdown(
                    "**Stability metrics per cycle:**\n"
                    "- Specific Capacitance (F/g)\n"
                    "- Capacitance Retention (%) = Cs_n / Cs_1 × 100\n"
                    "- Coulombic Efficiency (%) = t_dis / t_chg × 100\n"
                    "- Energy & Power density\n"
                    "- IR drop & ESR (current mode only)"
                )

        run_stability_btn = gr.Button("▶ Run Stability Analysis", variant="primary", size="lg")
        stability_status  = gr.Textbox(label="Status", lines=10, interactive=False)

        with gr.Tabs():
            with gr.Tab("📋 Per-Cycle Results"):
                stability_table = gr.Dataframe(
                    interactive=False,
                    label="Cycle-by-cycle: Cs, Retention, η_CE, E, P, IR Drop",
                )
            with gr.Tab("🔍 Extracted Cycle"):
                extracted_table = gr.Dataframe(
                    interactive=False,
                    label="Raw time-voltage data for the extracted cycle",
                )
            with gr.Tab("📦 Download"):
                stability_download = gr.File(
                    label="📦 Download Stability Analysis ZIP"
                )

    # ── References footer ─────────────────────────────────────
    gr.Markdown(
        "---\n"
        "**Scientific references:** "
        "Conway (1999) • Ragone (1968) • Stoller & Ruoff (2010) • "
        "Miller (2006) • Linden & Reddy (2010)"
    )

    # ── Interactivity ─────────────────────────────────────────
    mode_selector.change(fn=_show_normal_tab,    inputs=mode_selector, outputs=normal_panel)
    mode_selector.change(fn=_show_stability_tab, inputs=mode_selector, outputs=stability_panel)
    data_type.change(fn=_toggle_mass_input,      inputs=data_type,     outputs=mass_mg_input)

    # Normal GCD run
    run_normal_btn.click(
        fn=run_normal_analysis,
        inputs=[
            gcd_file, data_type, mass_mg_input,
            v_min_input, v_max_input, extract_cycle_normal,
        ],
        outputs=[normal_status, normal_summary_output, normal_download],
    )

    # Stability run
    run_stability_btn.click(
        fn=run_stability_analysis,
        inputs=[
            gcd_file, data_type, mass_mg_input,
            stability_col_label, v_min_input, v_max_input,
            extract_cycle_stability,
        ],
        outputs=[stability_status, stability_table, extracted_table, stability_download],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        max_file_size=MAX_FILE_SIZE,
        show_error=True,
    )
