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
| **Type** | `2D scatter`, `3D scatter`, or `Stacked bar`. |
| **X / Y / Z column** | Pick any column for each axis. In 2D the Y selector is restricted to numeric columns; in 3D all three are. In stacked-bar mode only X is selectable (bar height and segments are fixed by schema, see below). |
| **log X / log Y / log Z** | Toggle each axis between linear and log scale. |

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
- Pick `2D scatter`, X = `threads`, Y = `total_cycles`, enable log Y → scaling curve.
- Switch to `Stacked bar`, X = `threads` → cycle-budget breakdown per thread count.
- Switch to `3D scatter`, X = `threads`, Y = `mat_count`, Z = `total_cycles` → rotate to explore.
