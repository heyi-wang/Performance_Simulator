# Plotting Tool

Light-weight interactive plotter for the parametric-sweep CSVs produced by
the kernel and nafblock sweep scripts (e.g.
[`kernel/matmul/full_sweep.py`](../kernel/matmul/full_sweep.py)).

## Run
```bash
cd Plotting_Tool
pip install -r requirements.txt
streamlit run app.py
```

## Files
- [app.py](app.py) — Streamlit entry point. Sidebar controls + Plotly figure.
- [requirements.txt](requirements.txt) — `streamlit>=1.30`, `plotly>=5.18`,
  `pandas>=2.0`, `streamlit-sortables` (drag-reorder legend).
- [README.md](README.md) — install / run / controls.

## Current capabilities (v5)

### Input
- **File uploader**: browse / drag-drop a CSV; falls back to
  `kernel/matmul/full_sweep.csv` when nothing is uploaded.
- **Composite columns**: on load, `add_composite_columns(df)` appends
  string columns from `SYNTHETIC_COMPOSITES` when their three components
  are present — `gemm = "MxKxN"`, `tile = "MxKxN"`, `pool = "CxHxW"`,
  `block = "CxHxW"`. Originals remain selectable.

### Plot types
- 2D scatter, 2D line, 3D scatter, stacked bar.

### Axis / series controls
- Per-axis column selectors with linear / log toggles. Log axes auto-tick
  at base 2 for power-of-two integer columns (`threads`, `mat_count`,
  `vec_count`, `vec_bytes`, `tile_*`).
- **Series** selector (2D scatter + 2D line): `(none)` = single trace;
  otherwise one trace per unique value with auto-named legend entries.
- **Facet** column: subplot grid (one panel per unique value). Supports
  2D scatter / 2D line / stacked bar; 3D shows an info banner and falls
  back to a single chart. Legend dedup via `legendgroup`.

### Per-series styling
- Expander above the chart: legend-text override, color picker, dash
  style (2D line only). Keys include plot-type / series / facet so
  overrides survive unrelated re-renders.
- **Drag-to-reorder legend** (`streamlit-sortables`) stored in
  `st.session_state` and applied via Plotly's `legendrank`. Falls back
  to a static order + warning when the optional dep isn't installed.

### Stacked bar
Bar height = `total_cycles`, segmented by 5 cycle-fraction columns. The
segment schema is **autodetected** from the loaded CSV via
`detect_stack_segments(df)`:

- **Matmul schema** (default): `mat/vec/dma/scalar/stall_cycle_fraction_pct` —
  critical-path-worker cycle fractions from
  [`kernel/matmul/plot_sweep.py`](../kernel/matmul/plot_sweep.py).
- **Nafblock schema**: `layernorm/matmul/dwconv/pooling/vecops_cycle_fraction_pct` —
  per-backend layer cycle fractions from
  [`nafblock/full_sweep.py`](../nafblock/full_sweep.py). Selected when those
  5 columns are present in the loaded CSV.

Both schemas sum to 100 by construction. Multiple rows sharing an X value
are combined (totals summed, fractions weighted-averaged).
- X axis is `type="category"` so bars render at constant visual width.
- Per-segment **percentage labels**: ≥ `STACK_SIDE_LABEL_PCT` (4%) render
  inline at 12 pt; smaller slices get a **side annotation with an arrow**
  pointing at the segment mid-Y, staggered (`ax=45/70`) to avoid overlap.
  `layout.uniformtext = dict(mode="hide", minsize=11)` keeps inline label
  font size constant across bars.
- `Show segment %` checkbox toggles labels off entirely (hover still works).

### Fixed parameters
- `Fix values` sidebar pins any sweep parameter not on an axis to its
  first value (prevents row collisions on X+Series). `(none)` keeps the
  dim free.
- `fixed_params(df, exclude)` emits a `Fixed: …` caption listing pinned
  values. Skips `NON_PARAM_COLUMNS` (`verification_status`,
  `actual_mat_accels`, `actual_vec_accels`, `slowest_worker_tid`,
  `build_ok`, `run_ok`, `wall_seconds`, metric / utilization columns)
  and synthetic composite names.

### Variable dropdown filtering
X / Series / Facet (and 3D X) selectors hide `NON_PARAM_COLUMNS`. Y / Z
stay unrestricted so metrics remain plottable on the value axis.

### Styling
- **Scientific defaults** (`apply_scientific_layout`): template
  `simple_white`, serif font (`Times New Roman / Liberation Serif /
  DejaVu Serif`), inward black tick marks, 18 pt title, mirrored axes.
  3D scenes get the same treatment via `update_scenes`.
- **Plot style** sidebar selector — Plotly templates (`plotly`,
  `plotly_white`, `plotly_dark`, `ggplot2`, `seaborn`, `simple_white`,
  `none`). Scientific layout layers on top.
- **Color palette** sidebar selector — qualitative (`Plotly`, `D3`,
  `Set1`, `Set2`, `Pastel`) + sequential (`Viridis`, `Plasma`, `Blues`)
  via `plotly.colors.sample_colorscale`. Stacked-bar mat/vec/dma/scalar/stall
  colors preserved when `palette=Plotly`.
- **Grid toggles**: `Vertical gridlines` and `Horizontal gridlines`
  checkboxes (default on); propagate through facet subplots and 3D
  scene axes.

## Changelog (high level)
- v1 — initial Streamlit + Plotly app (2D / 3D scatter, stacked bar).
- v2 — 2D line plot, series selector, editable axis labels + title,
  fixed-parameter caption.
- v3 — plot-style + color-palette selectors, per-series style expander,
  facet column, `Fix values`, composite `gemm` / `tile` columns,
  log-axis base-2 ticks.
- v4 — stacked-bar percentage labels, drag-to-reorder legend, relevant-
  only dropdown filtering, scientific defaults.
- v5 — file uploader, grid toggles, show/hide segment % toggle,
  adaptive (side-arrow) percentage placement.

## Still deferred
- Full hue / brightness / style / marker group-by (only single-axis
  series + facet today).
- Multi-CSV overlay, figure export, custom palettes.
