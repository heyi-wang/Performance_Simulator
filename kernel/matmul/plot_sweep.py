#!/usr/bin/env python3
"""Plot arbitrary slices of the full_sweep.csv dataset.

2D mode (default):
    python3 kernel/matmul/plot_sweep.py \\
        --input kernel/matmul/full_sweep.csv \\
        --x threads --y total_cycles \\
        --group-by gemm_m,mat_count \\
        --filter tile_m=16,tile_k=32,tile_n=64,mat_latency=4 \\
        --log-x --log-y \\
        --output /tmp/demo.png

In 2D mode, --group-by columns are mapped to visual channels in order:
    1st column -> hue (distinct base colors)
    2nd column -> brightness within that hue (light -> dark)
    3rd column -> line style (solid, dashed, dash-dot, dotted)
    4th+ column -> marker shape

3D mode (--3d):
    python3 kernel/matmul/plot_sweep.py \\
        --input kernel/matmul/full_sweep.csv \\
        --x threads --y mat_count --z total_cycles --3d \\
        --3d-style surface \\
        --filter tile_m=16,tile_k=32,tile_n=64,mat_latency=4,gemm_m=1024,vec_count=1,vec_bytes=64 \\
        --log-z --output /tmp/demo3d.png

Filters accept comma-separated key=val pairs; values can be lists with '|'
(e.g. mat_count=1|2|4). Sweep parameters that are not on an axis or in
--group-by but have a single unique value across the filtered rows are
auto-annotated below the plot as fixed dimensions.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
except ImportError:
    sys.exit("matplotlib is required. Install with: pip install matplotlib")


NUMERIC_COLUMNS = {
    "tile_m", "tile_k", "tile_n",
    "mat_latency", "mat_count", "vec_count", "vec_bytes",
    "gemm_m", "gemm_k", "gemm_n", "threads",
    "total_cycles", "actual_mat_accels", "actual_vec_accels",
    "wall_seconds", "build_ok", "run_ok",
}

SWEEP_PARAM_COLUMNS = [
    "tile_m", "tile_k", "tile_n",
    "mat_latency", "mat_count", "vec_count", "vec_bytes",
    "gemm_m", "gemm_k", "gemm_n", "threads",
]

AXIS_LABELS = {
    "tile_m": "Tile M",
    "tile_k": "Tile K",
    "tile_n": "Tile N",
    "mat_latency": "Matrix latency [cycles]",
    "mat_count": "Matrix accelerator count",
    "vec_count": "Vector accelerator count",
    "vec_bytes": "Vector datapath width [B]",
    "gemm_m": "GEMM M",
    "gemm_k": "GEMM K",
    "gemm_n": "GEMM N",
    "threads": "Worker threads",
    "total_cycles": "Total cycles",
    "actual_mat_accels": "Active matrix accelerators",
    "actual_vec_accels": "Active vector accelerators",
    "wall_seconds": "Sim wall time [s]",
}

HUE_PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # teal
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
]
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def parse_filter(spec: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"bad --filter entry: '{part}'")
        key, value = part.split("=", 1)
        out[key.strip()] = [v.strip() for v in value.split("|") if v.strip()]
    return out


def parse_group_by(spec: str) -> list[str]:
    return [p.strip() for p in spec.split(",") if p.strip()]


def row_matches(row: dict[str, str], filters: dict[str, list[str]]) -> bool:
    for key, values in filters.items():
        if key not in row:
            return False
        if row[key] not in values:
            return False
    return True


def coerce(col: str, value: str):
    if col in NUMERIC_COLUMNS:
        if value == "":
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return None
    return value


def label_for(col: str, log: bool = False) -> str:
    base = AXIS_LABELS.get(col, col)
    if log:
        return f"{base} (log)"
    return base


def adjust_lightness(hex_color: str, lightness: float) -> tuple[float, float, float]:
    """Return RGB with the target lightness in HLS space, hue/sat preserved."""
    rgb = matplotlib.colors.to_rgb(hex_color)
    h, _, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, lightness)), s)


def assign_styles(
    series_keys: list[tuple],
    group_cols: list[str],
    flat: bool,
) -> dict[tuple, dict]:
    """Map each series key to a {color, linestyle, marker} dict.

    Hierarchical encoding (default):
        col[0] -> hue, col[1] -> brightness, col[2] -> linestyle,
        col[3+] -> marker. Values are sorted so brightness is monotonic
        and hue assignment is stable across runs.

    Flat encoding (--style flat):
        one color per series from HUE_PALETTE, marker cycle, solid lines.
    """
    if flat or not group_cols:
        styles: dict[tuple, dict] = {}
        for i, key in enumerate(series_keys):
            styles[key] = {
                "color": HUE_PALETTE[i % len(HUE_PALETTE)],
                "linestyle": "-",
                "marker": MARKERS[i % len(MARKERS)],
            }
        return styles

    levels: list[list] = []
    for depth in range(len(group_cols)):
        seen: list = []
        for key in series_keys:
            v = key[depth]
            if v not in seen:
                seen.append(v)
        try:
            seen.sort(key=lambda x: (float(x), str(x)))
        except (TypeError, ValueError):
            seen.sort(key=str)
        levels.append(seen)

    hues = {v: HUE_PALETTE[i % len(HUE_PALETTE)] for i, v in enumerate(levels[0])}

    if len(levels) >= 2:
        n_shade = max(1, len(levels[1]))
        if n_shade == 1:
            shade_pos = {levels[1][0]: 0.5}
        else:
            shade_pos = {v: i / (n_shade - 1) for i, v in enumerate(levels[1])}
    else:
        shade_pos = {None: 0.5}

    if len(levels) >= 3:
        ls_map = {v: LINE_STYLES[i % len(LINE_STYLES)] for i, v in enumerate(levels[2])}
    else:
        ls_map = {}

    if len(levels) >= 4:
        marker_index = {
            tuple(key[3:]): MARKERS[i % len(MARKERS)]
            for i, key in enumerate(
                sorted({tuple(k[3:]) for k in series_keys}, key=str)
            )
        }
    else:
        marker_index = {}

    L_LOW, L_HIGH = 0.30, 0.72
    styles = {}
    for key in series_keys:
        hue_hex = hues[key[0]]
        if len(group_cols) >= 2:
            t = shade_pos[key[1]]
            lightness = L_LOW + t * (L_HIGH - L_LOW)
            color = adjust_lightness(hue_hex, lightness)
        else:
            color = matplotlib.colors.to_rgb(hue_hex)

        linestyle = ls_map.get(key[2], "-") if len(group_cols) >= 3 else "-"
        marker = marker_index.get(tuple(key[3:]), "o") if len(group_cols) >= 4 else "o"

        styles[key] = {"color": color, "linestyle": linestyle, "marker": marker}
    return styles


def format_fixed_value(col: str, value) -> str:
    return f"{col}={value}"


def collect_fixed_dims(
    rows: list[dict],
    used_cols: set[str],
    explicit_filter: dict[str, list[str]],
) -> str:
    """Build the bottom-margin annotation listing dimensions held constant."""
    fixed: dict[str, object] = {}
    for col in SWEEP_PARAM_COLUMNS:
        if col in used_cols:
            continue
        values = {row.get(col, "") for row in rows}
        values.discard("")
        if len(values) == 1:
            v = next(iter(values))
            fixed[col] = coerce(col, v) if v else v

    parts: list[str] = []

    if all(c in fixed for c in ("tile_m", "tile_k", "tile_n")):
        parts.append(f"tile={fixed['tile_m']}x{fixed['tile_k']}x{fixed['tile_n']}")
        for c in ("tile_m", "tile_k", "tile_n"):
            fixed.pop(c)

    if all(c in fixed for c in ("gemm_m", "gemm_k", "gemm_n")):
        parts.append(f"gemm={fixed['gemm_m']}x{fixed['gemm_k']}x{fixed['gemm_n']}")
        for c in ("gemm_m", "gemm_k", "gemm_n"):
            fixed.pop(c)

    for col in SWEEP_PARAM_COLUMNS:
        if col in fixed:
            parts.append(f"{col}={fixed[col]}")

    return "  ·  ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot slices of full_sweep.csv (2D or 3D)."
    )
    p.add_argument("--input", required=True, help="Path to full_sweep.csv.")
    p.add_argument("--x", required=True, help="Column to put on the x-axis.")
    p.add_argument(
        "--y", default="total_cycles",
        help="2D: metric column. 3D: second horizontal axis. Default: total_cycles.",
    )
    p.add_argument(
        "--z", default="total_cycles",
        help="3D mode only: metric column on the vertical axis. Default: total_cycles.",
    )
    p.add_argument(
        "--group-by", default="",
        help="Comma-separated columns; one series per unique tuple.",
    )
    p.add_argument(
        "--filter", default="",
        help="Comma-separated key=val; use key=a|b to accept multiple values.",
    )
    p.add_argument("--log-x", action="store_true", help="Log2-scale x-axis.")
    p.add_argument("--log-y", action="store_true", help="Log-scale y-axis.")
    p.add_argument("--log-z", action="store_true", help="3D mode: log-scale z-axis.")
    p.add_argument("--output", required=True, help="Output PNG path.")
    p.add_argument("--title", default="", help="Optional plot title.")
    p.add_argument(
        "--require-pass", action="store_true",
        help="Drop rows where verification_status != PASS.",
    )
    p.add_argument(
        "--style", choices=["hierarchical", "flat"], default="hierarchical",
        help="2D color scheme: 'hierarchical' (hue/brightness/linestyle) or 'flat'.",
    )
    p.add_argument("--3d", dest="three_d", action="store_true", help="Render in 3D.")
    p.add_argument(
        "--3d-style", dest="three_d_style",
        choices=["scatter", "surface", "bar"], default="scatter",
        help="3D rendering style.",
    )
    p.add_argument(
        "--no-fixed-annotation", action="store_true",
        help="Suppress the auto-generated 'Fixed: ...' line below the plot.",
    )
    return p


def apply_paper_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-v0_8-paper"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 110,
    })


def render_2d(
    series: dict[tuple, list[tuple]],
    args: argparse.Namespace,
    group_cols: list[str],
    fixed_text: str,
) -> "plt.Figure":
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    series_keys = sorted(series.keys(), key=lambda k: tuple(str(x) for x in k))
    styles = assign_styles(series_keys, group_cols, flat=(args.style == "flat"))

    for key in series_keys:
        pts = sorted(series[key])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = (
            ", ".join(f"{c}={v}" for c, v in zip(group_cols, key))
            if group_cols else "all"
        )
        st = styles[key]
        ax.plot(
            xs, ys,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markersize=5.5,
            linewidth=1.7,
            label=label,
        )

    if args.log_x:
        ax.set_xscale("log", base=2)
        xs_all = sorted({p[0] for pts in series.values() for p in pts})
        try:
            ax.set_xticks(xs_all)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        except Exception:
            pass
    if args.log_y:
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))

    ax.set_xlabel(label_for(args.x, args.log_x))
    ax.set_ylabel(label_for(args.y, args.log_y))
    set_title(ax, args, group_cols, three_d=False)

    ax.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if group_cols:
        legend_title = " / ".join(group_cols)
        ncol = 2 if len(series_keys) > 6 else 1
        ax.legend(title=legend_title, frameon=False, ncol=ncol)

    finish_figure(fig, args, fixed_text)
    return fig


def render_3d(
    rows: list[dict],
    args: argparse.Namespace,
    group_cols: list[str],
    fixed_text: str,
) -> "plt.Figure":
    fig = plt.figure(figsize=(11.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_box_aspect((1.5, 1.2, 1.0))
    except (AttributeError, TypeError):
        pass

    pts: list[tuple] = []
    for row in rows:
        x = coerce(args.x, row.get(args.x, ""))
        y = coerce(args.y, row.get(args.y, ""))
        z = coerce(args.z, row.get(args.z, ""))
        if x is None or y is None or z is None:
            continue
        gkey = tuple(row.get(c, "") for c in group_cols) if group_cols else ("all",)
        pts.append((x, y, z, gkey))

    if not pts:
        sys.exit("No valid (x, y, z) points after filtering.")

    style = args.three_d_style

    if style == "surface":
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        try:
            surf = ax.plot_trisurf(
                xs, ys, zs, cmap="viridis", linewidth=0.2, antialiased=True, alpha=0.9
            )
            fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label=label_for(args.z, args.log_z))
        except Exception as exc:
            sys.exit(
                f"trisurf failed ({exc}); try --3d-style scatter or bar with this data."
            )

    elif style == "bar":
        keys_seen: list = []
        for _, _, _, gkey in pts:
            if gkey not in keys_seen:
                keys_seen.append(gkey)
        styles = assign_styles(keys_seen, group_cols, flat=(args.style == "flat"))
        x_vals = sorted({p[0] for p in pts})
        y_vals = sorted({p[1] for p in pts})
        dx = (x_vals[1] - x_vals[0]) * 0.6 if len(x_vals) > 1 else 0.5
        dy = (y_vals[1] - y_vals[0]) * 0.6 if len(y_vals) > 1 else 0.5
        for x, y, z, gkey in pts:
            ax.bar3d(
                x - dx / 2, y - dy / 2, 0, dx, dy, z,
                color=styles[gkey]["color"], alpha=0.85, edgecolor="black", linewidth=0.3,
            )
        if group_cols:
            for gkey in keys_seen:
                ax.scatter([], [], [], color=styles[gkey]["color"],
                           label=", ".join(f"{c}={v}" for c, v in zip(group_cols, gkey)))
            ax.legend(title=" / ".join(group_cols), frameon=False, fontsize=9)

    else:  # scatter
        keys_seen: list = []
        for _, _, _, gkey in pts:
            if gkey not in keys_seen:
                keys_seen.append(gkey)
        try:
            keys_seen.sort(key=lambda k: tuple(float(v) if v not in ("", None) else 0 for v in k))
        except (TypeError, ValueError):
            keys_seen.sort(key=str)
        styles = assign_styles(keys_seen, group_cols, flat=(args.style == "flat"))
        for gkey in keys_seen:
            gx = [p[0] for p in pts if p[3] == gkey]
            gy = [p[1] for p in pts if p[3] == gkey]
            gz = [p[2] for p in pts if p[3] == gkey]
            label = (
                ", ".join(f"{c}={v}" for c, v in zip(group_cols, gkey))
                if group_cols else "all"
            )
            ax.scatter(
                gx, gy, gz,
                color=styles[gkey]["color"],
                marker=styles[gkey]["marker"],
                s=42, depthshade=True, edgecolor="black", linewidth=0.4,
                label=label,
            )
        if group_cols:
            ax.legend(title=" / ".join(group_cols), frameon=False, fontsize=9)

    xs_all = [p[0] for p in pts]
    ys_all = [p[1] for p in pts]
    zs_all = [p[2] for p in pts]

    def _set_log(axis_setter, vals):
        positive = [v for v in vals if isinstance(v, (int, float)) and v > 0]
        if not positive:
            return
        lo, hi = min(positive), max(positive)
        try:
            axis_setter(lo * 0.9, hi * 1.1)
        except Exception:
            pass

    if args.log_x:
        _set_log(ax.set_xlim, xs_all)
        try:
            ax.set_xscale("log")
        except Exception:
            pass
    if args.log_y:
        _set_log(ax.set_ylim, ys_all)
        try:
            ax.set_yscale("log")
        except Exception:
            pass
    if args.log_z:
        _set_log(ax.set_zlim, zs_all)
        try:
            ax.set_zscale("log")
        except Exception:
            pass

    ax.set_xlabel(label_for(args.x, args.log_x), labelpad=10)
    ax.set_ylabel(label_for(args.y, args.log_y), labelpad=10)
    ax.set_zlabel(label_for(args.z, args.log_z), labelpad=10)
    set_title(ax, args, group_cols, three_d=True)

    ax.view_init(elev=22, azim=-58)
    finish_figure(fig, args, fixed_text, three_d=True)
    return fig


def set_title(ax, args: argparse.Namespace, group_cols: list[str], three_d: bool) -> None:
    if args.title:
        ax.set_title(args.title)
        return
    metric = args.z if three_d else args.y
    metric_h = AXIS_LABELS.get(metric, metric)
    if three_d:
        x_h = AXIS_LABELS.get(args.x, args.x)
        y_h = AXIS_LABELS.get(args.y, args.y)
        title = f"{metric_h}  vs  {x_h} × {y_h}"
    else:
        x_h = AXIS_LABELS.get(args.x, args.x)
        title = f"{metric_h}  vs  {x_h}"
    if group_cols:
        title += f"  (grouped by {', '.join(group_cols)})"
    ax.set_title(title)


def finish_figure(fig, args: argparse.Namespace, fixed_text: str, three_d: bool = False) -> None:
    if three_d:
        fig.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.10)
    else:
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    if fixed_text and not args.no_fixed_annotation:
        fig.text(
            0.5, 0.015, f"Fixed: {fixed_text}",
            ha="center", va="bottom",
            fontsize=9, color="0.30", style="italic",
        )


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    apply_paper_style()

    filters = parse_filter(args.filter)
    group_cols = parse_group_by(args.group_by)

    rows: list[dict] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.require_pass and row.get("verification_status") != "PASS":
                continue
            if not row_matches(row, filters):
                continue
            rows.append(row)

    if not rows:
        sys.exit("No rows match the filter.")

    used_cols = {args.x, *group_cols}
    if args.three_d:
        used_cols |= {args.y, args.z}
    else:
        used_cols.add(args.y)
    fixed_text = collect_fixed_dims(rows, used_cols, filters)

    if args.three_d:
        fig = render_3d(rows, args, group_cols, fixed_text)
    else:
        series: dict[tuple, list[tuple]] = defaultdict(list)
        for row in rows:
            key = tuple(row[c] for c in group_cols) if group_cols else ("all",)
            xv = coerce(args.x, row[args.x])
            yv = coerce(args.y, row[args.y])
            if xv is None or yv is None:
                continue
            series[key].append((xv, yv))
        if not series:
            sys.exit("No (x, y) pairs to plot after filtering.")
        fig = render_2d(series, args, group_cols, fixed_text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 200}
    if not args.three_d:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
