# 🔋 GCD Electrochemical Analyzer

Full Galvanostatic Charge-Discharge (GCD) analysis web application built with Gradio and deployable on Render.

## Modes

| Mode | Description |
|------|-------------|
| **Normal GCD** | Rate-capability study across multiple current / current-density columns |
| **Stability Test** | Cycle-life tracking over thousands of cycles at a single current / density |

## Scientific Equations

| Metric | Equation | Reference |
|--------|----------|-----------|
| Specific Capacitance (current density) | Cs = J · t_d / ΔV (F/g) | Conway (1999) |
| Specific Capacitance (current) | Cs = I · t_d / (m · ΔV) (F/g) | Stoller & Ruoff (2010) |
| Energy Density | E = ½ · Cs · ΔV² / 3.6 (Wh/kg) | Ragone (1968) |
| Power Density | P = E / t_d (W/kg) | Ragone (1968) |
| Coulombic Efficiency | η = t_dis / t_chg × 100 % | Linden & Reddy (2010) |
| IR Drop Resistance | R_ESR = ΔV_IR / (2·I) (Ω) | Miller (2006) |
| Capacitance Retention | Ret = Cs_n / Cs_1 × 100 % | — |

## Input File Format

**Wide format** — Column 0 = Time, Column 1+ = Potential at each current/density:

```
Time (s) | 1 A/g  | 2 A/g  | 5 A/g  | 10 A/g
---------|--------|--------|--------|-------
0.00     | 0.001  | 0.001  | 0.002  | 0.002
0.05     | 0.045  | 0.041  | 0.038  | 0.031
...
```

### Supported Header Formats
- Bare numbers: `1`, `2`, `5`
- With units: `1 A/g`, `2 mA/cm2`, `5 mA`, `0.5 A`
- Any combination — the parser extracts the number and unit automatically

## Cycle Detection

Real GCD data often contains multiple partial or complete cycles. The analyzer:
1. Finds turning points (charge tops, discharge bottoms) using a prominence-based peak finder
2. Pairs them into complete charge + discharge half-cycles
3. **Selects the cycle with the longest discharge time** — most accurate for capacitance integration
4. Handles cells that don't reach the full potential window gracefully

## Data Type Input

| Data Type | Mass needed? | Capacitance formula |
|-----------|-------------|---------------------|
| `current_density` (A/g, mA/g, …) | ❌ No | Cs = J · t / ΔV |
| `current` (A, mA) | ✅ Yes (enter mg) | Cs = I · t / (m · ΔV) |

## Stability Test Features

- Handles up to **20,000+ cycles** efficiently
- Per-cycle: Cs, Retention (%), η_CE, E, P, IR drop, ESR
- **Extract any specific cycle** by number — get its raw voltage–time data as a CSV and plot
- Choose which column to analyse if multiple are present

## Output ZIP Contents

### Normal GCD
```
GCD_Analysis_Output.zip
├── Results/
│   ├── GCD_Summary.csv               ← per current/density metrics
│   └── All_Cycles_Detected.csv       ← all detected cycles across all columns
├── Plots/
│   ├── GCD_Curves.png                ← best-cycle overlay
│   ├── GCD_Multi_Panel.png           ← one panel per current/density
│   ├── Capacitance_vs_CD.png         ← rate capability
│   ├── Ragone_Plot.png
│   ├── Coulombic_Efficiency.png
│   ├── IR_Drop_ESR.png
│   ├── Discharge_Time.png
│   └── All_Cycles_<label>.png        ← per-column cycle overview
├── Metadata/
│   └── Metadata.csv
├── Validation/
│   └── Analysis_Status.csv
├── GCD_Analysis_Results.xlsx
└── Analysis_Report.txt
```

### Stability Test
```
GCD_Stability_Output.zip
├── Results/
│   ├── Stability_Per_Cycle.csv       ← cycle-by-cycle metrics
│   └── Extracted_Cycle.csv           ← raw data for extracted cycle (if requested)
├── Plots/
│   ├── Stability_Retention.png       ← retention + coulombic efficiency
│   ├── Stability_Capacitance.png     ← Cs vs cycle number
│   ├── Stability_Energy_Power.png    ← E and P over cycles
│   └── Extracted_Cycle.png           ← if cycle extraction was requested
├── Metadata/
│   └── Metadata.csv
├── Validation/
│   └── Analysis_Status.csv
├── GCD_Stability_Results.xlsx
└── Analysis_Report.txt
```

## Local Run

```bash
pip install -r requirements.txt
python gcd_app.py
```

## Deploy on Render

1. Push to GitHub
2. Render → New Web Service → connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `python gcd_app.py`
5. Render provides `PORT` automatically

## References

- Conway, B.E. (1999). *Electrochemical Supercapacitors*. Kluwer/Plenum.
- Ragone, D.V. (1968). SAE Technical Paper 680453.
- Stoller, M.D. & Ruoff, R.S. (2010). *Energy Environ. Sci.*, 3, 1294.
- Miller, J.R. (2006). *Electrochim. Acta*, 52, 1703.
- Linden, D. & Reddy, T.B. (2010). *Handbook of Batteries*, 4th ed.
