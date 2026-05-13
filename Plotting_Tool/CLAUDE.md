## Build a light-weight interactive plotting tool to visulize the results of parametric sweep

# Requirements:
- Able to read csv file containing the parametric sweep results (like @/home/why/Desktop/Performance_Simulator/kernel/matmul/full_sweep.csv).
- Allow user to choose the variables to plot on each axis 
- Allow user to select the scale of each axis
- Allow user to choose 2D/3D or stacked bar chart (referring to /home/why/Desktop/Performance_Simulator/kernel/matmul/plot_sweep.py)
- Can be run in the web browser

Build a usable version first.
Create the "README.md" file to document the method of building/using the plotting tool.

Update this file for the current project status after modification.

## Status (v1 — usable)

Implemented as a single-file **Streamlit + Plotly** app reading the long/tidy
CSV schema produced by [`kernel/matmul/full_sweep.py`](../kernel/matmul/full_sweep.py).

### Files
- [app.py](app.py) — Streamlit entry point. Sidebar controls + Plotly figure in the body.
- [requirements.txt](requirements.txt) — `streamlit>=1.30`, `plotly>=5.18`, `pandas>=2.0`.
- [README.md](README.md) — install / run / controls / stacked-bar semantics / example.

### Controls in v1
- CSV path text input (default: `kernel/matmul/full_sweep.csv`, re-read on mtime change via `st.cache_data`).
- Plot type radio: **2D scatter**, **3D scatter**, **Stacked bar**.
- Per-axis column selectors (X / Y / Z) populated from `df.columns`; numeric-only enforced where it matters.
- Per-axis linear/log toggles.

### Stacked-bar
Mirrors `--bar` in [`kernel/matmul/plot_sweep.py`](../kernel/matmul/plot_sweep.py):
bar height = `total_cycles`, segmented by
`mat/vec/dma/scalar/stall_cycle_fraction_pct` with the same five colors used
by the reference matplotlib plotter. Multiple rows sharing an X value are
combined: totals summed, fractions averaged weighted by `total_cycles`.

### Verification
End-to-end exercised against `kernel/matmul/full_sweep.csv` (headless, with
`streamlit` stubbed): all three builders return valid Plotly figures; stacked
segment heights reconstruct each bar's `total_cycles` within rounding error.

### Not in v1 (deferred)
- Group-by encoding (hue / brightness / style / marker) like plot_sweep.py
  `--group-by`.
- Per-column filter widgets like plot_sweep.py `--filter`.
- Multi-CSV overlay, figure export, axis-range presets.

### Run
```bash
cd Plotting_Tool
pip install -r requirements.txt
streamlit run app.py
```

# Requirements for v2
- Add **2D Line Chart** to plot type
- Add legend to the chart that labels the meaning of each curve
- Allow user to edit the name of axis and the title of the plot
- Show the fixed parameters and their values while plotting variables

## Status (v2 — implemented)

All four v2 requirements landed in [app.py](app.py); README updated.

### Added in v2
- **2D Line plot type** (`build_line_2d`): `mode="lines+markers"`, rows
  sorted by X per series so the line is monotonic.
- **Series column selector** (2D scatter + 2D line). `(none)` = single trace
  (v1 behavior); otherwise one trace per unique value of the chosen column.
  Trace names `<col>=<value>` produce the Plotly legend automatically.
- **Editable title and axis labels**: sidebar text inputs `Plot title`,
  `X axis label`, `Y axis label`, `Z axis label` (Z only in 3D). Empty
  string falls back to the column name (and no title).
- **Fixed-parameter caption**: `fixed_params(df, exclude)` returns columns
  whose `nunique() == 1`, excluding any column used as X / Y / Z / Series.
  Reuses the `tile_m/k/n → tile=MxKxN` and `gemm_m/k/n → gemm=MxKxN`
  triple-collapsing from
  [`collect_fixed_dims`](../kernel/matmul/plot_sweep.py) so the annotation
  matches the matplotlib-side convention.

### Still deferred
- Full hue / brightness / style / marker group-by (only single-axis Series
  in v2).
- Per-column filter widgets.
- Multi-CSV overlay, figure export, axis-range presets.
