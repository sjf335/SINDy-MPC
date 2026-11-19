"""Selected-reference validation: tracking paths (Fig. 2) and errors (Fig. 3).

Read only the frozen formal run; do not rerun the plant or fit any model.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from section4b_data import load_validation

from plot_section4a_predictions import COLORS as PREDICTION_COLORS
from plot_section4a_predictions import STYLES as PREDICTION_STYLES


LATEX = Path(__file__).resolve().parents[1]
METHOD = LATEX.parent / "method"
RUN = METHOD / "runs/pelican_trajectory_search_v1"
OUT = LATEX / "figures"
DATA = OUT / "data"
WIDTH_IN = 521.30 / 72
KINDS = ("circle", "figure8", "helix")
TITLES = ("(a) Circle", "(b) Figure eight", "(c) Helix")
ORDER = ("pd", "sindy_current", "linear_history", "motor_sindy", "sindy_history")
SEEDS = tuple(range(24301, 24306))
PATH_SEED = SEEDS[0]
COLORS = {**PREDICTION_COLORS, "pd": "#777777", "reference": "black"}
STYLES = {**PREDICTION_STYLES, "pd": ":", "reference": (0, (7, 4))}
LABELS = {
    "reference": "Reference", "pd": "PD", "sindy_current": "SINDy-MPC",
    "linear_history": "Linear history MPC", "motor_sindy": "SINDy-MPC with Motor Dynamics",
    "sindy_history": "History SINDy-MPC",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified_data():
    data, _, _, _, sources, report = load_validation()
    return data, sources, report


def configure_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": .6, "grid.linewidth": .4, "grid.alpha": .55,
        "grid.color": "#D8D8D8", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "path.simplify": False,
    })


def line_width(name):
    return 1.3 if name == "sindy_history" else (1.0 if name == "reference" else .95)


def add_legend(fig, include_reference):
    names = (["reference"] if include_reference else []) + list(ORDER)
    # Fill columns so the displayed row order ends with the proposed controller.
    rows = [names[:3], names[3:]]
    legend_names = [rows[row][col] for col in range(3) for row in range(2) if col < len(rows[row])]
    handles = [Line2D([], [], color=COLORS[n], ls=STYLES[n], lw=line_width(n), label=LABELS[n])
               for n in legend_names]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
                        ncol=3, frameon=True, fancybox=False, edgecolor="black",
                        facecolor="white", framealpha=1, borderpad=.5,
                        handlelength=3.0, columnspacing=1.7, labelspacing=.5)
    legend.get_frame().set_linewidth(.6)


def path_figure(data):
    fig = plt.figure(figsize=(WIDTH_IN, 4.45))
    column_labels = ("PD", "SINDy-MPC", "Linear history\nMPC",
                     "SINDy-MPC with\nMotor Dynamics", "History\nSINDy-MPC")
    column_left = [.53 + 1.32 * col for col in range(5)]
    panel_width = 1.09
    row_bottom = (2.90, 1.85, .45)
    row_height = (1.10, .60, 1.10)
    row_limits = {}
    for col, label in enumerate(column_labels):
        fig.text((column_left[col] + panel_width/2) / WIDTH_IN, .962, label,
                 ha="center", va="center", fontsize=8, linespacing=1.15)
    for row, kind in enumerate(KINDS):
        is3d = kind == "helix"
        reference = data[(kind, PATH_SEED, "pd")]["reference"][:, :3]
        positions = [data[(kind, PATH_SEED, name)]["physical"][:, :3] for name in ORDER]
        p = np.concatenate([reference] + positions)
        span = np.ptp(p, axis=0)
        lower, upper = p.min(0) - .08 * span, p.max(0) + .08 * span
        row_limits[kind] = {"lower_m": lower.tolist(), "upper_m": upper.tolist()}
        center_y = (row_bottom[row] + row_height[row]/2) / 4.45
        fig.text(.015, center_y, TITLES[row], ha="center", va="center", fontsize=8, rotation=90)
        row_axes = []
        for col, (name, position) in enumerate(zip(ORDER, positions)):
            rect = [column_left[col]/WIDTH_IN, row_bottom[row]/4.45,
                    panel_width/WIDTH_IN, row_height[row]/4.45]
            kwargs = {"projection": "3d", "computed_zorder": False} if is3d else {}
            ax = fig.add_axes(rect, **kwargs)
            row_axes.append(ax)
            ncoords = 3 if is3d else 2
            # Every cell contains exactly one controller and the same reference.
            ax.plot(*position[:, :ncoords].T, color=COLORS[name], ls="-", lw=1.15, zorder=3)
            ax.plot(*reference[:, :ncoords].T, color="black", ls=(0, (5, 3)), lw=.8, zorder=4)
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            ax.tick_params(labelsize=8, pad=1, length=2.2, width=.5)
            if is3d:
                ax.set_zlim(lower[2], upper[2])
                ax.set_box_aspect(upper-lower, zoom=.88)
                ax.set_proj_type("ortho")
                ax.view_init(elev=27, azim=-58)
                ax.set_xticks([-2, 0])
                ax.set_yticks([-1, 1])
                ax.set_zticks([1.0, 1.6])
                ax.set_xlabel("x (m)", fontsize=8, labelpad=-8)
                ax.set_ylabel("y (m)", fontsize=8, labelpad=-8)
                # A compact horizontal label above the z axis avoids spilling
                # into the next controller's panel at conference column scale.
                ax.set_zlabel("")
                ax.text2D(.98, .80, "z (m)", transform=ax.transAxes,
                          fontsize=8, ha="center", va="bottom")
                ax.tick_params(pad=-2)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.set_pane_color((1, 1, 1, 0))
                    axis._axinfo["grid"].update(color=(.86, .86, .86, 1), linewidth=.35)
            else:
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([-2, -1, 0] if kind == "circle" else [-1, 0, 1])
                ax.set_yticks([-1, 0, 1] if kind == "circle" else [-.5, 0, .5])
                ax.set_xlabel("x (m)", fontsize=8, labelpad=2)
                if col == 0:
                    ax.set_ylabel("y (m)", fontsize=8, labelpad=2)
                else:
                    ax.tick_params(labelleft=False)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(True)
            assert len(ax.lines) == 2
            assert all(len(line.get_xdata()) == 2001 for line in ax.lines)
        for ax in row_axes[1:]:
            np.testing.assert_array_equal(ax.get_xlim(), row_axes[0].get_xlim())
            np.testing.assert_array_equal(ax.get_ylim(), row_axes[0].get_ylim())
            if is3d:
                np.testing.assert_array_equal(ax.get_zlim(), row_axes[0].get_zlim())
    handles = [Line2D([], [], color="black", ls=(0, (5, 3)), lw=.8, label="Reference"),
               Line2D([], [], color="black", ls="-", lw=1.15, label="Tracked trajectory")]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
                        ncol=2, frameon=True, fancybox=False, edgecolor="black", framealpha=1,
                        borderpad=.45, handlelength=3.0, columnspacing=2.0, fontsize=8)
    legend.get_frame().set_linewidth(.6)
    layout = {"rows": list(KINDS), "columns": list(ORDER), "panel_count": 15,
              "lines_per_panel": 2, "samples_per_line": 2001, "shared_limits_by_row": row_limits,
              "tracking_style": "Solid lines in controller colors; black dashed references.",
              "projection": "Equal-scale XY for circle/figure-eight; equal-scale orthographic 3D for helix.",
              "insets": False}
    return fig, layout


def error_figure(data):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 3.10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.35, top=.95, wspace=.16)
    for index, (ax, kind) in enumerate(zip(axes, KINDS)):
        for name in ORDER:
            group = [data[(kind, seed, name)] for seed in SEEDS]
            errors = np.stack([d["error_cm"] for d in group])
            assert errors.min() >= 0 and errors.max() < 15, "Error plot limits would hide samples."
            t = group[0]["t"]
            ax.fill_between(t, errors.min(axis=0), errors.max(axis=0),
                            color=COLORS[name], alpha=.10, lw=0, zorder=1)
            ax.plot(t, errors.mean(axis=0), color=COLORS[name], ls=STYLES[name],
                    lw=line_width(name), zorder=4 if name == "sindy_history" else 3)
        ax.set(xlim=(0, 20), ylim=(0, 15), xlabel="Time (s)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks([0, 3, 6, 9, 12, 15])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(pad=2, length=2.5, width=.6)
        ax.grid(True)
        rect = ax.get_position()
        fig.text(rect.x0 + rect.width/2, .20, TITLES[index], ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Position error (cm)")
    add_legend(fig, include_reference=False)
    return fig


def export_data(data, selection="all"):
    path_frames, error_frames, summary_frames = [], [], []
    for kind in KINDS:
        ref = data[(kind, PATH_SEED, "pd")]
        for name in ("reference",) + ORDER:
            p = ref["reference"][:, :3] if name == "reference" else data[(kind, PATH_SEED, name)]["physical"][:, :3]
            path_frames.append(pd.DataFrame({"trajectory": kind, "seed": PATH_SEED, "controller": name,
                "time_s": ref["t"], "x_m": p[:, 0], "y_m": p[:, 1], "z_m": p[:, 2]}))
        for name in ORDER:
            for seed in SEEDS:
                d = data[(kind, seed, name)]
                error_frames.append(pd.DataFrame({"trajectory": kind, "controller": name, "seed": seed,
                                                  "time_s": d["t"], "error_cm": d["error_cm"]}))
            e = np.stack([data[(kind, seed, name)]["error_cm"] for seed in SEEDS])
            summary_frames.append(pd.DataFrame({"trajectory": kind, "controller": name, "time_s": ref["t"],
                "mean_error_cm": e.mean(0), "min_error_cm": e.min(0), "max_error_cm": e.max(0), "n": 5}))
    for name, frames in [("tracking_paths", path_frames), ("position_error_per_run", error_frames),
                         ("position_error_summary", summary_frames)]:
        if selection == "paths" and name != "tracking_paths":
            continue
        if selection == "errors" and name == "tracking_paths":
            continue
        pd.concat(frames, ignore_index=True).to_csv(DATA / f"section4b_{name}.csv", index=False)


def save_figure(fig, stem):
    pdf, png = OUT / f"{stem}.pdf", OUT / f"{stem}.png"
    fig.savefig(pdf, metadata={"Title": stem.replace("_", " "), "Subject": "Tracking from test initial conditions on references selected during development"})
    fig.savefig(png, dpi=600)
    dimensions = list(fig.get_size_inches())
    plt.close(fig)
    with Image.open(png) as picture:
        picture.convert("L").save(OUT / f"{stem}_grayscale.png", dpi=(600, 600))
        pixels = list(picture.size)
    return {"size_inches": dimensions, "png_dpi": 600, "png_pixels": pixels,
            "sha256": {p.name: digest(p) for p in (pdf, png)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "paths", "errors"), default="all")
    args = parser.parse_args()
    report_path = OUT / "section4b_tracking_verification.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    data, sources, report = read_verified_data()
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    configure_style()
    export_data(data, args.figure)
    report["figures"] = previous.get("figures", {})
    if args.figure in ("all", "paths"):
        fig, report["trajectory_grid"] = path_figure(data)
        report["figures"]["figure_2"] = save_figure(fig, "section4b_tracking_paths")
    elif "trajectory_grid" in previous:
        report["trajectory_grid"] = previous["trajectory_grid"]
    if args.figure in ("all", "errors"):
        report["figures"]["figure_3"] = save_figure(error_figure(data), "section4b_position_error")
    # Reused figures remain byte-for-byte identical to the previously verified output.
    for figure in report["figures"].values():
        for name, expected_hash in figure["sha256"].items():
            assert digest(OUT / name) == expected_hash
    for path, before in sources.items():
        assert digest(path) == before, f"Source changed during plotting: {path}"
    report["source_hashes_unchanged"] = True
    report["source_sha256"] = {p.relative_to(METHOD).as_posix(): value for p, value in sources.items()}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
