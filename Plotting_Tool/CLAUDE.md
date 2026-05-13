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

# Requirements for v3
- Allow user to choose color for individual line/point
- Allow user to assign a similar color to groups
- Allow user to choose line style.
- Allow user to set a style of the whole plot 
- Allow editing legend text in-place.
- Allow subplot (multiple small charts on one page)

## Status (v3 — implemented)

All six v3 requirements landed in [app.py](app.py); README updated; the
v2 "Fixed" annotation bug (output/status columns leaking in) is fixed in
the same commit via a `NON_PARAM_COLUMNS` exclusion set.

### Added in v3
- **Plot style** sidebar selector — Plotly templates (`plotly`,
  `plotly_white`, `plotly_dark`, `ggplot2`, `seaborn`, `simple_white`, `none`).
- **Color palette** sidebar selector — qualitative (`Plotly`, `D3`, `Set1`,
  `Set2`, `Pastel`) and sequential (`Viridis`, `Plasma`, `Blues`) palettes.
  Sequential palettes are evenly sampled via `plotly.colors.sample_colorscale`.
  Drives the default per-trace color, including stacked-bar segments
  (preserves historical mat/vec/dma/scalar/stall colors at `palette=Plotly`).
- **Per-series style expander** — above the chart, one row per trace with
  legend-text override, color picker, and (2D line only) dash style.
  Keys include plot-type / series / facet so overrides survive unrelated
  re-renders and reset on context switches.
- **Facet column** — splits the chart into a subplot grid (one panel per
  unique value). Supports 2D scatter / 2D line / Stacked bar; 3D shows an
  `st.info` and falls back to a single chart. Shared X/Y axes; legend
  deduplicated across panels via `legendgroup`.
- **NON_PARAM_COLUMNS** filter applied inside `fixed_params` so the
  `Fixed: …` caption no longer lists `verification_status`,
  `actual_mat_accels`, `actual_vec_accels`, `slowest_worker_tid`,
  `build_ok`, `run_ok`, `wall_seconds`, or any metric/utilization column —
  only true sweep parameters.

### Still deferred
- Full hue / brightness / style / marker group-by (only single-axis Series
  + facet today).
- Multi-CSV overlay, figure export, custom palettes.

### Follow-up patch: composite `gemm` / `tile` columns and constant bar widths
- **Composite columns (additive)**: on load, `add_composite_columns(df)`
  appends string columns `gemm = "MxKxN"` and `tile = "MxKxN"` when the
  underlying components are present. The raw `gemm_m/k/n` and
  `tile_m/k/n` remain in the dataframe and in every dropdown — composites
  are added alongside, not in place. `expand_used_with_composites` keeps
  the composite and its parts in sync inside the `used` set so
  `fixed_params` doesn't double-list them. The synthetic names are also
  skipped inside `fixed_params` directly (the triple-collapse already
  produces `gemm=MxKxN` / `tile=MxKxN`).
- **Constant stacked-bar widths**: stacked-bar X axis is now
  `type="category"`, so bars render at constant visual width regardless
  of X spacing. The log-X toggle is hidden in the sidebar for stacked
  bar (it had no effect on a categorical axis).

### Follow-up patch: log axes use base 2 for power-of-two columns
Plotly's `type="log"` defaults to base 10 (ticks at 1, 10, 100, …), which
is wrong for sweep dims like `threads`, `mat_count`, `vec_count`,
`vec_bytes`, `tile_*`. `_axis_kwargs(title, use_log, values)` now sets
`dtick = log10(2)` whenever the underlying data is a positive integer
power of two, so log axes tick at 1, 2, 4, 8, …. Applied across all
plot types, including facet subplots and the 3D scene axes.

### Follow-up patch: Fix values
After v3 landed, a sidebar **Fix values** section was added: any sweep
parameter column not used as X / Y / Z / Series / Facet that has more
than one CSV value gets a dropdown defaulting to its first concrete
value. This prevents collisions on the chart (multiple rows mapping to
the same X+Series). `(none)` keeps the dimension free. Filters are
applied before plotting; `fixed_params` then naturally shows the pinned
values in the `Fixed:` caption.