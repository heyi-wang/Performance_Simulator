"""Interactive plotting tool for parametric-sweep CSVs.

Run with:
    streamlit run Plotting_Tool/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "kernel" / "matmul" / "full_sweep.csv"

STACK_SEGMENTS = [
    ("mat_cycle_fraction_pct",    "Mat",    "#1f77b4"),
    ("vec_cycle_fraction_pct",    "Vec",    "#2ca02c"),
    ("dma_cycle_fraction_pct",    "DMA",    "#ff7f0e"),
    ("scalar_cycle_fraction_pct", "Scalar", "#9467bd"),
    ("stall_cycle_fraction_pct",  "Stall",  "#d62728"),
]


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    del mtime  # cache key only
    return pd.read_csv(path)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _axis_type(use_log: bool) -> str:
    return "log" if use_log else "linear"


def build_scatter_2d(df: pd.DataFrame, x: str, y: str,
                     logx: bool, logy: bool) -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(
            x=df[x], y=df[y], mode="markers",
            marker=dict(size=8, opacity=0.8),
            hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title=x, type=_axis_type(logx)),
        yaxis=dict(title=y, type=_axis_type(logy)),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def build_scatter_3d(df: pd.DataFrame, x: str, y: str, z: str,
                     logx: bool, logy: bool, logz: bool) -> go.Figure:
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
        scene=dict(
            xaxis=dict(title=x, type=_axis_type(logx)),
            yaxis=dict(title=y, type=_axis_type(logy)),
            zaxis=dict(title=z, type=_axis_type(logz)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def build_stacked_bar(df: pd.DataFrame, x: str, logx: bool) -> go.Figure:
    needed = [col for col, _, _ in STACK_SEGMENTS] + ["total_cycles"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Stacked-bar requires columns {needed}; missing: {missing}"
        )

    work = df[[x] + needed].copy()
    # If multiple rows share an X value, sum total_cycles and average fractions
    # weighted by total_cycles so the segment heights still add up.
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

    fig = go.Figure()
    for col, label, color in STACK_SEGMENTS:
        seg_height = agg["total_cycles"] * agg[col] / 100.0
        fig.add_trace(go.Bar(
            x=agg[x], y=seg_height, name=label, marker_color=color,
            customdata=agg[col],
            hovertemplate=(
                f"{x}: %{{x}}<br>{label}: %{{y:.0f}} cycles "
                "(%{customdata:.1f}%)<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title=x, type=_axis_type(logx)),
        yaxis=dict(title="total_cycles"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


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

        st.header("Plot")
        plot_type = st.radio(
            "Type", ["2D scatter", "3D scatter", "Stacked bar"], index=0,
        )

        cols = list(df.columns)
        num_cols = numeric_columns(df)

        def _default(name: str, pool: list[str]) -> int:
            return pool.index(name) if name in pool else 0

        if plot_type == "2D scatter":
            x = st.selectbox("X column", cols, index=_default("threads", cols))
            y = st.selectbox("Y column", num_cols,
                             index=_default("total_cycles", num_cols))
            logx = st.checkbox("log X", value=False)
            logy = st.checkbox("log Y", value=False)
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
        else:  # Stacked bar
            x = st.selectbox("X column", cols, index=_default("threads", cols))
            logx = st.checkbox("log X", value=False)

    if plot_type == "2D scatter":
        fig = build_scatter_2d(df, x, y, logx, logy)
    elif plot_type == "3D scatter":
        fig = build_scatter_3d(df, x, y, z, logx, logy, logz)
    else:
        try:
            fig = build_stacked_bar(df, x, logx)
        except ValueError as exc:
            st.error(str(exc))
            return

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(df)} rows · {csv_path}")


if __name__ == "__main__":
    main()
