"""Show the saved representative prediction window as a 3-D position path."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from plot_section4a_predictions import (
    COLORS, LABELS, MODELS, OUT, STYLES, WIDTH_IN, LATEX,
    digest, read_verified_example,
)


STEM = "section4a_prediction_3d"
COLUMN_WIDTH_IN = 251.65 / 72
COLUMN_HEIGHT_IN = 2.60


def single_column_legend(fig, intermediate_s, duration):
    """Six model keys and three time markers in exactly two compact rows."""
    def entry(label, color="black", linestyle="none", marker=None):
        symbol = DrawingArea(11 if marker is None else 6, 8, 0, 0)
        width = symbol.width
        symbol.add_artist(Line2D([1, width-1] if marker is None else [width/2],
                                 [4, 4] if marker is None else [4],
                                 color=color, linestyle=linestyle, linewidth=1.2,
                                 marker=marker, markersize=5 if marker == "*" else 3.5,
                                 markerfacecolor="white" if marker == "o" else color))
        return HPacker(children=[symbol, TextArea(label, textprops={"size": 8})],
                       align="center", pad=0, sep=2)

    rows = []
    for names in [("truth", "sindy_current", "motor_sindy"),
                  ("linear_history", "constant_twist", "sindy_history")]:
        rows.append([entry(
            LABELS[name], "black" if name == "truth" else COLORS[name],
            "-" if name == "truth" else STYLES[name]) for name in names])
    markers = HPacker(children=[entry(f"{intermediate_s:g} s", marker="o"),
                               entry(f"{duration:g} s", marker="s")],
                      align="center", pad=0, sep=4)
    rows[0].append(entry("Start", marker="*"))
    rows[1].append(markers)
    content = VPacker(children=[HPacker(children=row, align="center", pad=0, sep=3)
                               for row in rows], align="center", pad=0, sep=3)
    box = AnchoredOffsetbox(loc="lower center", child=content, frameon=True,
                            bbox_to_anchor=(0.5, 0.015), bbox_transform=fig.transFigure,
                            pad=0.45, borderpad=0, prop={"size": 8})
    box.patch.set(edgecolor="black", facecolor="white", linewidth=0.7)
    fig.add_artist(box)
    fig.canvas.draw()
    bounds = box.get_window_extent(fig.canvas.get_renderer())
    assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width, "Legend exceeds column width."


def draw(saved, time=None, origin_time=333.01, intermediate_s=0.5, height_in=3.65,
         width_in=WIDTH_IN, single_column=False):
    if time is None:
        time = np.arange(len(saved["truth"]))*0.01
    intermediate_index = int(np.argmin(abs(time-intermediate_s)))
    assert np.isclose(time[intermediate_index], intermediate_s)
    duration = float(time[-1])
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    if single_column:
        plt.rcParams.update({"axes.labelsize": 8, "xtick.labelsize": 8,
                             "ytick.labelsize": 8})
    fig = plt.figure(figsize=(width_in, height_in))
    axes_rect = [-0.025, 0.10, 0.87, 1.06] if single_column else [0.04, 0.24, 0.90, 0.80]
    ax = fig.add_axes(axes_rect, projection="3d", computed_zorder=False)
    all_positions = np.concatenate([states[:, :3] for states in saved.values()])
    minimum, maximum = all_positions.min(axis=0), all_positions.max(axis=0)
    padding = 0.09 * (maximum-minimum)
    bounds = np.column_stack([minimum-padding, maximum+padding])
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    # Equal distance in metres corresponds to equal scale along each spatial axis.
    ax.set_box_aspect(bounds[:, 1]-bounds[:, 0], zoom=1.0 if single_column else 1.30)
    ax.set_proj_type("ortho")
    ax.view_init(elev=28, azim=-58)
    ax.set_xlabel("x (m)", labelpad=8)
    ax.set_ylabel("y (m)", labelpad=3 if single_column else 8)
    ax.set_zlabel("z (m)", labelpad=0 if single_column else 6)
    ax.xaxis.set_major_locator(MaxNLocator(4 if single_column else 5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=(0.86, 0.86, 0.86, 1), linewidth=0.45)
    for name in ["truth"] + MODELS:
        position = saved[name][:, :3]
        color = "black" if name == "truth" else COLORS[name]
        linewidth = 2.0 if name == "truth" else (1.6 if name == "sindy_history" else 1.25)
        style = "-" if name == "truth" else STYLES[name]
        ax.plot(*position.T, color=color, linestyle=style, linewidth=linewidth,
                zorder=4 if name == "truth" else 3, clip_on=False)
        ax.scatter(*position[intermediate_index], color=color, facecolors="white", edgecolors=color,
                   marker="o", s=20, linewidths=1, depthshade=False, zorder=5, clip_on=False)
        ax.scatter(*position[-1], color=color, marker="s", s=20, linewidths=0.5,
                   edgecolors="white", depthshade=False, zorder=5, clip_on=False)
    start = saved["truth"][0, :3]
    ax.scatter(*start, color="black", marker="*", s=72, edgecolors="white",
               linewidths=0.5, depthshade=False, zorder=6, clip_on=False)
    ax.text(*start, "  Start", color="black", fontsize=8, ha="left", va="bottom", zorder=7)

    if single_column:
        single_column_legend(fig, intermediate_s, duration)
        return fig, bounds

    legend_order = ["truth", "motor_sindy", "sindy_current", "sindy_history", "linear_history", "constant_twist"]
    handles = [Line2D([], [], color="black" if name == "truth" else COLORS[name],
                      linestyle="-" if name == "truth" else STYLES[name],
                      linewidth=1.8 if name == "truth" else 1.4, label=LABELS[name])
               for name in legend_order]
    point_handles = [
        Line2D([], [], linestyle="none", color="black", marker="*", markersize=7, label="Common start"),
        Line2D([], [], linestyle="none", color="black", marker="o", markerfacecolor="white", markersize=5, label=f"{intermediate_s:g} s"),
        Line2D([], [], linestyle="none", color="black", marker="s", markersize=5, label=f"{duration:g} s"),
    ]
    # Matplotlib fills columns first: two models and one time marker per column.
    combined_handles = [handle for column in range(3)
                        for handle in [*handles[2*column:2*column+2], point_handles[column]]]
    legend = fig.legend(handles=combined_handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
                        ncol=3, frameon=True, fancybox=False, framealpha=1,
                        edgecolor="black", facecolor="white", borderpad=0.65,
                        handlelength=3.2, columnspacing=1.8, labelspacing=0.6)
    legend.get_frame().set_linewidth(0.7)
    return fig, bounds


def main():
    time, saved, _, sources, report = read_verified_example()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"data").mkdir(exist_ok=True)
    rows = []
    for name, states in saved.items():
        for index, elapsed in enumerate(time):
            rows.append({"model": "observed" if name == "truth" else name,
                         "flight_id": 26, "elapsed_s": elapsed, "log_time_s": 333.01+elapsed,
                         "x_m": states[index, 0], "y_m": states[index, 1], "z_m": states[index, 2]})
    pd.DataFrame(rows).to_csv(OUT/"data/section4a_prediction_3d.csv", index=False)
    fig, bounds = draw(saved, width_in=COLUMN_WIDTH_IN, height_in=COLUMN_HEIGHT_IN,
                       single_column=True)
    pdf_path, png_path = OUT/f"{STEM}.pdf", OUT/f"{STEM}.png"
    fig.savefig(pdf_path, metadata={"Title": "Three-dimensional observed and predicted position trajectories", "Subject": "Pelican flight 26, 333.01-334.01 s"})
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    with Image.open(png_path) as image:
        image.convert("L").save(OUT/f"{STEM}_grayscale.png", dpi=(600, 600))
        report["png_pixels"] = list(image.size)
    for path, before in sources.items():
        assert digest(path) == before, f"Frozen source changed: {path}"
    report.update({
        "source_hashes_unchanged": True,
        "source_sha256": {str(p.relative_to(LATEX.parent)).replace("\\", "/"): value for p, value in sources.items()},
        "plot_transform": "Original world-frame position components, no smoothing or coordinate transformation.",
        "spatial_scale": "Equal metres per unit length on all three axes; orthographic camera.",
        "axis_limits_m": bounds.tolist(), "marker_indices": {"start": 0, "0.5_s": 50, "1_s": 100},
        "figure_size_inches": [COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN], "png_dpi": 600,
        "latex_width": "251.65 bp (MEAE text column)",
        "legend_layout": "One compact rectangular frame below the plot: exactly two balanced rows, three model entries and time markers per row; 8 pt Times New Roman text.",
        "in_plot_title": False,
        "output_sha256": {path.name: digest(path) for path in [pdf_path, png_path]},
    })
    (OUT/f"{STEM}_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ["source_sha256", "output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
