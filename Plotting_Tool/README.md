# Sweep Plotter

Interactive, browser-based visualizer for parametric-sweep CSVs produced by
[`kernel/matmul/full_sweep.py`](../kernel/matmul/full_sweep.py) (and any other
sweep script that emits the same long/tidy CSV schema, e.g.
`worker_sweep.csv`, `hardware_sweep.csv`).

Built with [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com/python/).
Re-rendering on widget changes is automatic; Plotly provides interactive zoom,
hover tooltips, and 3D rotation.

## Install

From the repo root:

```bash
cd Plotting_Tool
pip install -r requirements.txt
```

A virtualenv is recommended but not required.

## Run

```bash
streamlit run app.py
```

Streamlit prints a local URL (default `http://localhost:8501`) and opens it in
your browser.

## Controls

All controls live in the left sidebar.

| Control | Effect |
|---|---|
| **CSV path** | Path to the sweep CSV. Defaults to `kernel/matmul/full_sweep.csv` relative to the repo root. Change it to point at any other sweep CSV; dropdowns auto-populate from that file's columns. The file is re-read whenever its modification time changes. |
| **Type** | `2D scatter`, `2D line`, `3D scatter`, or `Stacked bar`. |
| **X / Y / Z column** | Pick any column for each axis. In 2D the Y selector is restricted to numeric columns; in 3D all three are. In stacked-bar mode only X is selectable (bar height and segments are fixed by schema, see below). |
| **log X / log Y / log Z** | Toggle each axis between linear and log scale. When the underlying values are positive integer powers of two (e.g. `threads`, `mat_count`, `vec_bytes`), the log ticks switch to base 2 automatically. Stacked-bar X is categorical (constant bar width), so the log-X toggle is hidden there. |
| **Series column** (2D only) | Pick a column to group rows into separate curves. `(none)` = one curve. Otherwise one trace per unique value of the chosen column, labeled in the legend as `<col>=<value>`. In `2D line` each group is sorted by X before connecting. |
| **Plot title** | Optional figure title. Empty = no title. |
| **X / Y / Z axis label** | Override the default axis label (which is the column name). Empty = use the column name. The Z input only appears in 3D mode. |
| **Plot style** | Overall theme: `plotly`, `plotly_white`, `plotly_dark`, `ggplot2`, `seaborn`, `simple_white`, `none`. |
| **Color palette** | Default colors for every trace. Qualitative: `Plotly`, `D3`, `Set1`, `Set2`, `Pastel`. Sequential (sampled evenly): `Viridis`, `Plasma`, `Blues`. Stacked-bar segments use the palette only when changed from `Plotly` (the default keeps the historical mat/vec/dma/scalar/stall colors). Individual colors can still be overridden in the **Style per series** panel. |
| **Facet column** | Split the chart into one subplot per unique value of the chosen column. Supported in `2D scatter`, `2D line`, and `Stacked bar`. Shared X and Y axes; legend rendered once across all panels. Not supported in `3D scatter`. |
| **Fix values** | One dropdown per sweep parameter that isn't being used as X / Y / Z / Series / Facet and that has more than one value in the CSV. The default is the first concrete value, which pins the dimension so rows don't collide on the chart. Choose `(none)` for a column to let it vary (legacy behavior). |

### Style per series

An expander **"Style per series"** sits above the chart whenever the active
mode has multiple traces (2D scatter / 2D line with a Series column,
2D scatter / 2D line without one, Stacked bar). Each row exposes:

- **Legend** — edit the trace's legend label in place (overrides
  `<series>=<value>` or the stacked-bar segment name).
- **Color** — color picker; default is the palette color for that index.
- **Line** (2D line only) — `solid / dot / dash / longdash / dashdot / longdashdot`.

Overrides are keyed on `(plot type, series column, facet column, trace key)`
so they persist when you toggle unrelated controls but reset when you
switch the series or plot type.

Below the chart, a `Fixed: …` caption auto-lists columns whose values are
constant across the loaded data (excluding the axes / series you're currently
plotting), matching the annotation style of
[`kernel/matmul/plot_sweep.py`](../kernel/matmul/plot_sweep.py).

### Composite `gemm` / `tile` columns

When the CSV contains the three components (`gemm_m`, `gemm_k`, `gemm_n`
or `tile_m`, `tile_k`, `tile_n`), the tool exposes an extra synthetic
column — `gemm` or `tile` — whose values are the concatenated shape
strings (`"128x128x128"`). The raw component columns remain available
in every dropdown too, so you can pick either the whole shape or a
single dimension.

When the composite is used as X / Y / Z / Series / Facet, the `Fixed:`
caption suppresses the corresponding triple-collapse to avoid
duplication, and vice-versa.

### Stacked-bar mode

Mirrors `--bar` in [`kernel/matmul/plot_sweep.py`](../kernel/matmul/plot_sweep.py):

- One bar per unique value of the chosen X column.
- Bar height = `total_cycles`.
- Each bar split into five segments using the cycle-fraction columns:
  `mat_cycle_fraction_pct`, `vec_cycle_fraction_pct`,
  `dma_cycle_fraction_pct`, `scalar_cycle_fraction_pct`,
  `stall_cycle_fraction_pct`.
- If multiple rows share an X value, totals are summed and fractions are
  averaged weighted by `total_cycles` so segment heights still add up.

Hovering a segment shows its absolute cycle count and percentage of the bar.

## Example

```bash
# from repo root, after running a sweep:
python kernel/matmul/full_sweep.py --jobs 4
streamlit run Plotting_Tool/app.py
```

In the browser:
- Set CSV path to `kernel/matmul/full_sweep.csv`.
- Pick `2D line`, X = `threads`, Y = `total_cycles`, Series = `mat_count`, enable log Y → one curve per `mat_count` value with a legend.
- Switch to `Stacked bar`, X = `threads` → cycle-budget breakdown per thread count.
- Switch to `3D scatter`, X = `threads`, Y = `mat_count`, Z = `total_cycles` → rotate to explore.
