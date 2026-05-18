"""Interactive plotting tool for parametric-sweep CSVs.

Run with:
    streamlit run Plotting_Tool/app.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "kernel" / "matmul" / "full_sweep.csv"

STACK_SEGMENTS = [
    ("mat_cycle_fraction_pct",    "Mat",    "#1f77b4"),
    ("vec_cycle_fraction_pct",    "Vec",    "#2ca02c"),
    ("dma_cycle_fraction_pct",    "DMA",    "#ff7f0e"),
    ("scalar_cycle_fraction_pct", "Scalar", "#9467bd"),
    ("stall_cycle_fraction_pct",  "Stall",  "#d62728"),
]

# Columns that are run output / status, not sweep parameters.
# Mirrors the SWEEP_PARAM_COLUMNS / metric split in kernel/matmul/full_sweep.py.
NON_PARAM_COLUMNS: set[str] = {
    "verification_status",
    "actual_mat_accels", "actual_vec_accels",
    "slowest_worker_tid",
    "build_ok", "run_ok",
    "wall_seconds", "total_cycles",
    "mat_util_pct", "vec_util_pct",
    "mat_cycle_fraction_pct", "vec_cycle_fraction_pct",
    "dma_cycle_fraction_pct", "scalar_cycle_fraction_pct",
    "stall_cycle_fraction_pct",
}

SYNTHETIC_COMPOSITES: dict[str, tuple[str, str, str]] = {
    "gemm": ("gemm_m", "gemm_k", "gemm_n"),
    "tile": ("tile_m", "tile_k", "tile_n"),
    "pool": ("pool_channels", "pool_height", "pool_width"),
}

SIZE_LABEL_COMPOSITES = {"gemm", "pool"}
ALIASED_COLUMN_GROUPS = (
    {"threads", "workers"},
)


def add_composite_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append synthetic shape string columns when their three
    components are present. Returns a copy with the extra columns; raw
    component columns are kept (additive)."""
    df = df.copy()
    for synth, parts in SYNTHETIC_COMPOSITES.items():
        if synth in df.columns:
            continue
        if all(p in df.columns for p in parts):
            df[synth] = (
                df[parts[0]].astype(str) + "x"
                + df[parts[1]].astype(str) + "x"
                + df[parts[2]].astype(str)
            )
    return df


def expand_used_with_composites(used: set[str]) -> set[str]:
    """If a composite or alias is used, mark equivalent columns used too.

    Prevents Fix values from pinning duplicate columns such as
    threads/workers or size_label/pool while the equivalent column is already
    driving the chart.
    """
    out = set(used)
    for synth, parts in SYNTHETIC_COMPOSITES.items():
        if synth in used:
            out.update(parts)
        if any(p in used for p in parts):
            out.add(synth)
        if synth in SIZE_LABEL_COMPOSITES and (
            "size_label" in used or synth in out or any(p in out for p in parts)
        ):
            out.add("size_label")
            out.add(synth)
            out.update(parts)
    if any(
        synth in out or any(part in out for part in SYNTHETIC_COMPOSITES[synth])
        for synth in SIZE_LABEL_COMPOSITES
    ):
        for synth in SIZE_LABEL_COMPOSITES:
            out.add(synth)
            out.update(SYNTHETIC_COMPOSITES[synth])
        out.add("size_label")
    for group in ALIASED_COLUMN_GROUPS:
        if out & group:
            out.update(group)
    return out


def duplicates_used_param(df: pd.DataFrame, col: str, used: set[str]) -> bool:
    """True when col carries the same values as an already-used parameter."""
    for used_col in used:
        if used_col == col or used_col not in df.columns:
            continue
        if used_col in NON_PARAM_COLUMNS:
            continue
        try:
            if df[col].astype(str).equals(df[used_col].astype(str)):
                return True
        except (TypeError, ValueError):
            continue
    return False


QUALITATIVE_PALETTES = ["Plotly", "D3", "Set1", "Set2", "Pastel"]
SEQUENTIAL_PALETTES = ["Viridis", "Plasma", "Blues"]
ALL_PALETTES = QUALITATIVE_PALETTES + SEQUENTIAL_PALETTES

TEMPLATES = [
    "plotly", "plotly_white", "plotly_dark", "ggplot2",
    "seaborn", "simple_white", "none",
]

DASH_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

NONE_LABEL = "(none)"


# --------------------------------------------------------------------- helpers


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    del mtime  # cache key only
    return pd.read_csv(path)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _axis_type(use_log: bool) -> str:
    return "log" if use_log else "linear"


LOG2_DTICK = math.log10(2)


def _is_pow2_series(values) -> bool:
    """True if every value is a positive integer power of two."""
    try:
        items = [v for v in values if pd.notna(v)]
    except TypeError:
        return False
    if not items:
        return False
    for v in items:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        if f <= 0 or not f.is_integer():
            return False
        iv = int(f)
        if iv & (iv - 1):
            return False
    return True


def _axis_kwargs(title: str, use_log: bool, values) -> dict:
    """Build a Plotly axis dict, switching log-axis tick spacing to base 2
    when the underlying data is power-of-two (so 'threads' shows 1,2,4,8,…
    instead of 1,10,100,…)."""
    cfg: dict = {"title": title, "type": _axis_type(use_log)}
    if use_log and _is_pow2_series(values):
        cfg["dtick"] = LOG2_DTICK
    return cfg


def _to_hex(color: str) -> str:
    if color.startswith("#"):
        return color.lower()
    if color.startswith("rgb"):
        nums = color[color.find("(") + 1: color.find(")")].split(",")
        r, g, b = (max(0, min(255, int(round(float(n))))) for n in nums[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    return color


def palette_colors(name: str, n: int) -> list[str]:
    if n <= 0:
        return []
    if name in QUALITATIVE_PALETTES:
        seq = getattr(pcolors.qualitative, name)
        return [_to_hex(seq[i % len(seq)]) for i in range(n)]
    if name in SEQUENTIAL_PALETTES:
        if n == 1:
            samples = [0.5]
        else:
            samples = [i / (n - 1) for i in range(n)]
        return [_to_hex(c) for c in pcolors.sample_colorscale(name, samples)]
    return [_to_hex(pcolors.qualitative.Plotly[i % 10]) for i in range(n)]


def _style_for(styles: dict | None, key, fallback_label: str,
               fallback_color: str) -> dict:
    s = (styles or {}).get(key, {})
    return {
        "label": s.get("label") or fallback_label,
        "color": s.get("color") or fallback_color,
        "dash":  s.get("dash") or "solid",
    }


def _apply_2d_layout(fig: go.Figure, title: str, xlabel: str, ylabel: str,
                     x: str, y: str, logx: bool, logy: bool,
                     template: str, df: pd.DataFrame | None = None) -> None:
    x_vals = df[x] if df is not None and x in df else None
    y_vals = df[y] if df is not None and y in df else None
    fig.update_layout(
        template=template,
        title=title or None,
        xaxis=_axis_kwargs(xlabel or x, logx, x_vals),
        yaxis=_axis_kwargs(ylabel or y, logy, y_vals),
        margin=dict(l=40, r=20, t=50 if title else 30, b=40),
    )


# --------------------------------------------------------------------- builders


def build_scatter_2d(df: pd.DataFrame, x: str, y: str, logx: bool, logy: bool,
                     series: str | None, *, styles: dict | None = None,
                     palette: str = "Plotly", template: str = "plotly",
                     title: str = "", xlabel: str = "", ylabel: str = "",
                     ) -> go.Figure:
    fig = go.Figure()
    if series is None:
        st_ = _style_for(styles, "__single__", y, palette_colors(palette, 1)[0])
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], mode="markers",
            name=st_["label"],
            marker=dict(size=8, opacity=0.85, color=st_["color"]),
            hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
            showlegend=False,
        ))
    else:
        values = sorted(df[series].dropna().unique())
        colors = palette_colors(palette, len(values))
        for i, value in enumerate(values):
            sub = df[df[series] == value]
            default_label = f"{series}={value}"
            st_ = _style_for(styles, value, default_label, colors[i])
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="markers",
                name=st_["label"],
                marker=dict(size=8, opacity=0.85, color=st_["color"]),
                hovertemplate=(
                    f"{st_['label']}<br>{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>"
                ),
            ))
    _apply_2d_layout(fig, title, xlabel, ylabel, x, y, logx, logy, template, df)
    return fig


def build_line_2d(df: pd.DataFrame, x: str, y: str, logx: bool, logy: bool,
                  series: str | None, *, styles: dict | None = None,
                  palette: str = "Plotly", template: str = "plotly",
                  title: str = "", xlabel: str = "", ylabel: str = "",
                  ) -> go.Figure:
    fig = go.Figure()
    if series is None:
        sub = df.sort_values(x)
        st_ = _style_for(styles, "__single__", y, palette_colors(palette, 1)[0])
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="lines+markers",
            name=st_["label"],
            line=dict(color=st_["color"], dash=st_["dash"]),
            marker=dict(size=7, color=st_["color"]),
            hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
            showlegend=False,
        ))
    else:
        values = sorted(df[series].dropna().unique())
        colors = palette_colors(palette, len(values))
        for i, value in enumerate(values):
            sub = df[df[series] == value].sort_values(x)
            default_label = f"{series}={value}"
            st_ = _style_for(styles, value, default_label, colors[i])
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="lines+markers",
                name=st_["label"],
                line=dict(color=st_["color"], dash=st_["dash"]),
                marker=dict(size=7, color=st_["color"]),
                hovertemplate=(
                    f"{st_['label']}<br>{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>"
                ),
            ))
    _apply_2d_layout(fig, title, xlabel, ylabel, x, y, logx, logy, template, df)
    return fig


def build_scatter_3d(df: pd.DataFrame, x: str, y: str, z: str,
                     logx: bool, logy: bool, logz: bool, *,
                     template: str = "plotly",
                     title: str = "", xlabel: str = "", ylabel: str = "",
                     zlabel: str = "") -> go.Figure:
    fig = go.Figure(
        data=go.Scatter3d(
            x=df[x], y=df[y], z=df[z], mode="markers",
            marker=dict(size=4, opacity=0.85),
            hovertemplate=(
                f"{x}: %{{x}}<br>{y}: %{{y}}<br>{z}: %{{z}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template=template,
        title=title or None,
        scene=dict(
            xaxis=_axis_kwargs(xlabel or x, logx, df[x]),
            yaxis=_axis_kwargs(ylabel or y, logy, df[y]),
            zaxis=_axis_kwargs(zlabel or z, logz, df[z]),
        ),
        margin=dict(l=0, r=0, t=50 if title else 30, b=0),
    )
    return fig


def build_stacked_bar(df: pd.DataFrame, x: str, logx: bool, *,
                      styles: dict | None = None,
                      palette: str = "Plotly", template: str = "plotly",
                      title: str = "", xlabel: str = "", ylabel: str = "",
                      ) -> go.Figure:
    needed = [col for col, _, _ in STACK_SEGMENTS] + ["total_cycles"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stacked-bar requires columns {needed}; missing: {missing}"
        )

    work = df[[x] + needed].copy()
    grouped_rows = []
    for xv, sub in work.groupby(x, sort=True):
        total = sub["total_cycles"].sum()
        row = {x: xv, "total_cycles": total}
        for col, _, _ in STACK_SEGMENTS:
            row[col] = (
                (sub[col] * sub["total_cycles"]).sum() / total
                if total > 0 else 0.0
            )
        grouped_rows.append(row)
    agg = pd.DataFrame(grouped_rows)

    # Palette overrides the hardcoded segment colors when user changes from Plotly.
    if palette == "Plotly":
        seg_defaults = [c for _, _, c in STACK_SEGMENTS]
    else:
        seg_defaults = palette_colors(palette, len(STACK_SEGMENTS))

    fig = go.Figure()
    for (col, label, _), default_color in zip(STACK_SEGMENTS, seg_defaults):
        st_ = _style_for(styles, label, label, default_color)
        seg_height = agg["total_cycles"] * agg[col] / 100.0
        fig.add_trace(go.Bar(
            x=agg[x], y=seg_height,
            name=st_["label"], marker_color=st_["color"],
            customdata=agg[col],
            hovertemplate=(
                f"{x}: %{{x}}<br>{st_['label']}: %{{y:.0f}} cycles "
                "(%{customdata:.1f}%)<extra></extra>"
            ),
        ))
    fig.update_layout(
        template=template,
        title=title or None,
        barmode="stack",
        # Categorical X so all bars render at constant visual width,
        # regardless of how the underlying X values are spaced.
        xaxis=dict(title=xlabel or x, type="category"),
        yaxis=dict(title=ylabel or "total_cycles"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        margin=dict(l=40, r=20, t=50 if title else 40, b=40),
    )
    return fig


# ---------------------------------------------------------------- fixed params


def fixed_params(df: pd.DataFrame, exclude: set[str]) -> str:
    # Synthetic composites are reported via the triple-collapse below
    # (`tile=MxKxN` / `gemm=MxKxN`), so we never list them by their
    # composite name directly.
    skip = exclude | NON_PARAM_COLUMNS | set(SYNTHETIC_COMPOSITES.keys())
    if any(
        all(part in df.columns for part in SYNTHETIC_COMPOSITES[synth])
        for synth in SIZE_LABEL_COMPOSITES
    ):
        skip.add("size_label")
    fixed: dict[str, object] = {}
    for col in df.columns:
        if col in skip:
            continue
        try:
            if df[col].nunique(dropna=True) == 1:
                vals = df[col].dropna()
                if not vals.empty:
                    fixed[col] = vals.iloc[0]
        except TypeError:
            continue

    parts: list[str] = []
    for synth, composite_cols in SYNTHETIC_COMPOSITES.items():
        if all(c in fixed for c in composite_cols):
            values = [fixed[c] for c in composite_cols]
            if synth == "tile" and all(str(v) in ("0", "0.0") for v in values):
                for c in composite_cols:
                    fixed.pop(c)
                continue
            parts.append(f"{synth}={values[0]}x{values[1]}x{values[2]}")
            for c in composite_cols:
                fixed.pop(c)
    for col, val in fixed.items():
        parts.append(f"{col}={val}")
    return "  ·  ".join(parts)


# ------------------------------------------------------------------- facet wrap


def render_facet(df: pd.DataFrame, facet_col: str, builder_fn,
                 template: str, title: str,
                 xlabel: str, ylabel: str, x: str, y: str | None,
                 logx: bool, logy: bool) -> go.Figure:
    values = sorted(df[facet_col].dropna().unique())
    n = len(values)
    ncols = max(1, math.ceil(math.sqrt(n)))
    nrows = max(1, math.ceil(n / ncols))
    titles = [f"{facet_col}={v}" for v in values]

    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=titles,
        shared_xaxes=True, shared_yaxes=True,
    )
    seen_legend = set()
    for i, v in enumerate(values):
        r = i // ncols + 1
        c = i % ncols + 1
        try:
            sub_fig = builder_fn(df[df[facet_col] == v])
        except ValueError as exc:
            st.warning(f"facet {facet_col}={v}: {exc}")
            continue
        for tr in sub_fig.data:
            name = tr.name or "trace"
            if name in seen_legend:
                tr.showlegend = False
            else:
                seen_legend.add(name)
                tr.showlegend = True
            tr.legendgroup = name
            fig.add_trace(tr, row=r, col=c)
        # Carry barmode forward if we're stacking bars.
        if sub_fig.layout.barmode:
            fig.update_layout(barmode=sub_fig.layout.barmode)

    fig.update_layout(
        template=template,
        title=title or None,
        margin=dict(l=40, r=20, t=70 if title else 50, b=40),
    )
    x_kwargs = _axis_kwargs(xlabel or x, logx, df[x] if x in df else None)
    fig.update_xaxes(**{k: v for k, v in x_kwargs.items() if k != "title"},
                     title_text=x_kwargs["title"])
    if y is not None:
        y_vals = df[y] if y in df else None
        y_kwargs = _axis_kwargs(ylabel or y, logy, y_vals)
        fig.update_yaxes(**{k: v for k, v in y_kwargs.items() if k != "title"},
                         title_text=y_kwargs["title"])
    return fig


# ------------------------------------------------------------ style expander UI


def _style_key_suffix(plot_type: str, series: str | None, facet: str | None) -> str:
    return f"{plot_type}|{series or '-'}|{facet or '-'}"


def render_style_expander(plot_type: str, df: pd.DataFrame,
                          series: str | None, palette: str,
                          facet: str | None,
                          line_dash: bool) -> dict:
    """Render per-trace style overrides above the chart.

    Returns a dict {trace_key: {"label", "color", "dash"}}.
    """
    suffix = _style_key_suffix(plot_type, series, facet)
    styles: dict = {}

    if plot_type == "Stacked bar":
        if palette == "Plotly":
            default_colors = [c for _, _, c in STACK_SEGMENTS]
        else:
            default_colors = palette_colors(palette, len(STACK_SEGMENTS))
        keys = [label for _, label, _ in STACK_SEGMENTS]
        defaults = list(zip(keys, keys, default_colors))
    elif plot_type in ("2D scatter", "2D line"):
        if series is None:
            keys = ["__single__"]
            defaults = [("__single__", "trace", palette_colors(palette, 1)[0])]
        else:
            values = sorted(df[series].dropna().unique())
            colors = palette_colors(palette, len(values))
            keys = list(values)
            defaults = [(v, f"{series}={v}", colors[i])
                        for i, v in enumerate(values)]
    else:
        return styles  # 3D: not supported in v3

    with st.expander("Style per series", expanded=False):
        if line_dash:
            header = st.columns([2, 1, 1])
            header[0].markdown("**Legend**")
            header[1].markdown("**Color**")
            header[2].markdown("**Line**")
        else:
            header = st.columns([2, 1])
            header[0].markdown("**Legend**")
            header[1].markdown("**Color**")
        for key, default_label, default_color in defaults:
            if line_dash:
                cols = st.columns([2, 1, 1])
            else:
                cols = st.columns([2, 1])
            label_in = cols[0].text_input(
                "label", value=default_label, label_visibility="collapsed",
                key=f"label_{suffix}_{key}",
            )
            color_in = cols[1].color_picker(
                "color", value=default_color, label_visibility="collapsed",
                key=f"color_{suffix}_{key}",
            )
            dash_in = "solid"
            if line_dash:
                dash_in = cols[2].selectbox(
                    "dash", DASH_STYLES, index=0,
                    label_visibility="collapsed",
                    key=f"dash_{suffix}_{key}",
                )
            styles[key] = {"label": label_in, "color": color_in, "dash": dash_in}
    return styles


# ----------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="Sweep Plotter", layout="wide")
    st.title("Parametric-Sweep Plotter")

    with st.sidebar:
        st.header("Data")
        csv_path = st.text_input("CSV path", value=str(DEFAULT_CSV))
        if not os.path.exists(csv_path):
            st.error(f"CSV not found: {csv_path}")
            return
        mtime = os.path.getmtime(csv_path)
        try:
            df = load_csv(csv_path, mtime)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to read CSV: {exc}")
            return
        df = add_composite_columns(df)

        st.header("Plot")
        plot_type = st.radio(
            "Type",
            ["2D scatter", "2D line", "3D scatter", "Stacked bar"],
            index=0,
        )

        cols_all = list(df.columns)
        num_cols = numeric_columns(df)

        def _default(name: str, pool: list[str]) -> int:
            return pool.index(name) if name in pool else 0

        st.header("Axes")
        used: set[str] = set()
        series: str | None = None
        x = y = z = ""
        logx = logy = logz = False

        if plot_type in ("2D scatter", "2D line"):
            x = st.selectbox("X column", cols_all, index=_default("threads", cols_all))
            y = st.selectbox("Y column", num_cols,
                             index=_default("total_cycles", num_cols))
            logx = st.checkbox("log X", value=False)
            logy = st.checkbox("log Y", value=False)
            pick = st.selectbox("Series column", [NONE_LABEL] + cols_all, index=0)
            series = None if pick == NONE_LABEL else pick
            used = {x, y}
            if series:
                used.add(series)
        elif plot_type == "3D scatter":
            x = st.selectbox("X column", num_cols,
                             index=_default("threads", num_cols))
            y = st.selectbox("Y column", num_cols,
                             index=_default("mat_count", num_cols))
            z = st.selectbox("Z column", num_cols,
                             index=_default("total_cycles", num_cols))
            logx = st.checkbox("log X", value=False)
            logy = st.checkbox("log Y", value=False)
            logz = st.checkbox("log Z", value=False)
            used = {x, y, z}
        else:  # Stacked bar
            x = st.selectbox("X column", cols_all, index=_default("threads", cols_all))
            # log-X has no visual effect with a categorical X axis, so we
            # omit the toggle here to avoid confusion.
            logx = False
            used = {x, "total_cycles"} | {c for c, _, _ in STACK_SEGMENTS}

        st.header("Labels")
        title = st.text_input("Plot title", value="")
        xlabel = st.text_input("X axis label", value="")
        ylabel = st.text_input("Y axis label", value="")
        zlabel = ""
        if plot_type == "3D scatter":
            zlabel = st.text_input("Z axis label", value="")

        st.header("Style")
        template = st.selectbox("Plot style", TEMPLATES, index=0)
        palette = st.selectbox("Color palette", ALL_PALETTES, index=0)
        facet_pick = st.selectbox("Facet column", [NONE_LABEL] + cols_all, index=0)
        facet = None if facet_pick == NONE_LABEL else facet_pick

        # ------------------------------------------------ Fix values
        # Every sweep parameter not used as X/Y/Z/Series/Facet gets pinned
        # to one of its CSV values; otherwise rows would collide on the chart.
        # Picking "(all)" lets the dimension vary (legacy behavior).
        active_used = set(used)
        if facet:
            active_used.add(facet)
        active_used = expand_used_with_composites(active_used)
        free_cols = [
            c for c in cols_all
            if c not in active_used
            and c not in NON_PARAM_COLUMNS
            and not duplicates_used_param(df, c, active_used)
            and df[c].nunique(dropna=True) > 1
        ]
        fix_filters: dict[str, object] = {}
        if free_cols:
            st.header("Fix values")
            for col in free_cols:
                opts = sorted(df[col].dropna().unique().tolist())
                option_values = [None] + opts
                idx = st.selectbox(
                    col, list(range(len(option_values))),
                    index=1,  # first concrete value
                    format_func=lambda i, _vals=option_values: (
                        NONE_LABEL if _vals[i] is None else str(_vals[i])
                    ),
                    key=f"fix_{col}",
                )
                if option_values[idx] is not None:
                    fix_filters[col] = option_values[idx]

    # Apply fixes outside the sidebar context.
    if fix_filters:
        mask = pd.Series(True, index=df.index)
        for col, val in fix_filters.items():
            mask &= (df[col] == val)
        df = df[mask].copy()
        if df.empty:
            st.warning("No rows match the current Fix values selection.")
            return

    # Style expander (main panel).
    styles: dict = {}
    if plot_type != "3D scatter":
        styles = render_style_expander(
            plot_type, df, series, palette, facet,
            line_dash=(plot_type == "2D line"),
        )

    # Facet caveat for 3D.
    if facet and plot_type == "3D scatter":
        st.info("Facet is not supported for 3D scatter; rendering single chart.")
        facet = None
    if facet:
        used.add(facet)

    # Build figure.
    if plot_type == "2D scatter":
        def _build(d: pd.DataFrame) -> go.Figure:
            return build_scatter_2d(
                d, x, y, logx, logy, series,
                styles=styles, palette=palette, template=template,
                title="", xlabel=xlabel, ylabel=ylabel,
            )
        if facet:
            fig = render_facet(df, facet, _build, template, title,
                               xlabel, ylabel, x, y, logx, logy)
        else:
            fig = build_scatter_2d(
                df, x, y, logx, logy, series,
                styles=styles, palette=palette, template=template,
                title=title, xlabel=xlabel, ylabel=ylabel,
            )
    elif plot_type == "2D line":
        def _build(d: pd.DataFrame) -> go.Figure:
            return build_line_2d(
                d, x, y, logx, logy, series,
                styles=styles, palette=palette, template=template,
                title="", xlabel=xlabel, ylabel=ylabel,
            )
        if facet:
            fig = render_facet(df, facet, _build, template, title,
                               xlabel, ylabel, x, y, logx, logy)
        else:
            fig = build_line_2d(
                df, x, y, logx, logy, series,
                styles=styles, palette=palette, template=template,
                title=title, xlabel=xlabel, ylabel=ylabel,
            )
    elif plot_type == "3D scatter":
        fig = build_scatter_3d(
            df, x, y, z, logx, logy, logz,
            template=template, title=title,
            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
        )
    else:  # Stacked bar
        def _build(d: pd.DataFrame) -> go.Figure:
            return build_stacked_bar(
                d, x, logx, styles=styles, palette=palette,
                template=template, title="",
                xlabel=xlabel, ylabel=ylabel,
            )
        if facet:
            fig = render_facet(df, facet, _build, template, title,
                               xlabel, ylabel, x, "total_cycles", logx, False)
        else:
            try:
                fig = build_stacked_bar(
                    df, x, logx, styles=styles, palette=palette,
                    template=template, title=title,
                    xlabel=xlabel, ylabel=ylabel,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

    st.plotly_chart(fig, use_container_width=True)
    fixed_text = fixed_params(df, expand_used_with_composites(used))
    if fixed_text:
        st.caption(f"Fixed: {fixed_text}")
    st.caption(f"{len(df)} rows · {csv_path}")


if __name__ == "__main__":
    main()
