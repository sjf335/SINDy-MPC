"""Plot observed states and saved free-running predictions for Section IV-A."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation


LATEX = Path(__file__).resolve().parents[1]
RUN = LATEX.parent / "method/runs/pelican_e2_v1"
OUT = LATEX / "figures"
STEM = "section4a_prediction_trajectories"
WIDTH_IN = 43 * 12 / 72.27
MODELS = ["sindy_current", "linear_history", "motor_sindy", "sindy_history", "constant_twist"]
COLORS = {"sindy_current": "#D55E00", "linear_history": "#0072B2", "motor_sindy": "#CC79A7", "sindy_history": "#009E73", "constant_twist": "#777777"}
LABELS = {"truth": "Observed", "sindy_current": "SINDy", "linear_history": "Linear history", "motor_sindy": "SINDy with Motor Dynamics", "sindy_history": "History SINDy", "constant_twist": "Constant twist"}
STYLES = {"sindy_current": "--", "linear_history": "-.", "motor_sindy": (0, (5, 1, 1, 1)), "sindy_history": "-", "constant_twist": ":"}
MARKERS = {"sindy_current": "s", "linear_history": "^", "motor_sindy": "D", "sindy_history": "o", "constant_twist": "x"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified_example():
    selection_path = RUN / "analysis/representative_selection.json"
    frozen_path = RUN / "FROZEN.json"
    perflight_path = RUN / "test/prediction_per_flight.csv"
    perwindow_path = RUN / "test/prediction_per_window.csv"
    sources = {p: digest(p) for p in [selection_path, frozen_path, perflight_path, perwindow_path]}
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert selection["rule"] == frozen["predeclared_plot_selection"]
    meta = next(item for item in selection["examples"] if item["designation"] == "median")
    flight_id, origin = meta["flight_id"], meta["origin_index"]
    assert (flight_id, origin) == (26, 33300)
    perflight = pd.read_csv(perflight_path)
    baseline = perflight[(perflight.model == "sindy_current") & (perflight.horizon_s == 0.5)].sort_values(["velocity_rmse", "flight_id"])
    assert int(baseline.iloc[len(baseline)//2].flight_id) == flight_id
    windows = pd.read_csv(perwindow_path)
    candidates = windows[(windows.model == "sindy_current") & (windows.flight_id == flight_id) & (windows.horizon_s == 0.5)].sort_values(["velocity_error", "origin_index"])
    assert int(candidates.iloc[len(candidates)//2].origin_index) == origin

    saved_path = RUN / f"analysis/example_median_flight_{flight_id:02d}.npz"
    sources[saved_path] = digest(saved_path)
    with np.load(saved_path) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    time = saved.pop("time")
    np.testing.assert_allclose(time, np.arange(101)*0.01, rtol=0, atol=1e-14)
    assert set(saved) == set(["truth"] + MODELS)
    maximum_endpoint_delta = 0.0
    for name, path in saved.items():
        assert path.shape == (101, 13) and np.isfinite(path).all()
        np.testing.assert_allclose(np.linalg.norm(path[:, 6:10], axis=1), 1, atol=1e-12)
        if name == "truth":
            continue
        np.testing.assert_allclose(path[0], saved["truth"][0], rtol=0, atol=1e-12)
        endpoint_path = RUN / f"test/{name}_endpoints.npz"
        sources[endpoint_path] = digest(endpoint_path)
        with np.load(endpoint_path) as endpoints:
            for step in [1, 10, 25, 50, 100]:
                prefix = f"f{flight_id:02d}_s{step:03d}_"
                idx = np.flatnonzero(endpoints[prefix+"origins"] == origin)
                assert len(idx) == 1
                idx = int(idx[0])
                predicted = endpoints[prefix+"prediction"][idx]
                truth = endpoints[prefix+"truth"][idx]
                assert endpoints[prefix+"failed_at_step"][idx] > step
                np.testing.assert_allclose(path[step], predicted, rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(saved["truth"][step], truth, rtol=1e-10, atol=1e-11)
                maximum_endpoint_delta = max(maximum_endpoint_delta, float(np.max(abs(path[step]-predicted))))

    angles = {}
    angle_roundtrip_error = 0.0
    for name, states in saved.items():
        rotations = Rotation.from_quat(states[:, 6:10])
        euler = np.unwrap(rotations.as_euler("xyz"), axis=0)
        angles[name] = euler
        angle_roundtrip_error = max(angle_roundtrip_error, float(np.max((Rotation.from_euler("xyz", euler).inv()*rotations).magnitude())))
    assert angle_roundtrip_error < 1e-12
    report = {
        "status": "PASS", "selection": meta, "selection_rule": selection["rule"],
        "selection_rule_recomputed": True, "prediction_steps": 100, "dt_s": 0.01,
        "models": MODELS, "saved_endpoints_checked": 25,
        "max_endpoint_absolute_difference": maximum_endpoint_delta,
        "euler_quaternion_roundtrip_max_rad": angle_roundtrip_error,
        "all_predictions_finite": True,
        "state_source": "Saved E2 conditional free-running predictions and observed/reconstructed states.",
        "future_inputs": "Recorded motor commands; no repeated measured-state reset within this 1 s window.",
        "plot_transform": "Quaternion to xyz Euler angles for display only; no smoothing, refitting or time alignment.",
        "prediction_recomputed": False,
    }
    return time, saved, angles, sources, report


def draw(time, saved, angles):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH_IN, 5.55), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.085, top=0.865, hspace=0.46, wspace=0.42)
    column_titles = ["Position x", "Position y", "Position z", "Velocity x", "Velocity y", "Velocity z", "Roll", "Pitch", "Yaw"]
    labels = [r"$x$ (m)", r"$y$ (m)", r"$z$ (m)", r"$v_x$ (m/s)", r"$v_y$ (m/s)", r"$v_z$ (m/s)", r"$\phi$ (rad)", r"$\theta$ (rad)", r"$\psi$ (rad)"]
    for index, ax in enumerate(axes.flat):
        component = index if index < 6 else index-6
        series = saved if index < 6 else angles
        ax.axvline(0.5, color="0.65", linewidth=0.7, linestyle=(0, (3, 3)), zorder=0)
        ax.plot(time, series["truth"][:, component], color="black", linewidth=1.7, zorder=4)
        for name in MODELS:
            ax.plot(time, series[name][:, component], color=COLORS[name], linestyle=STYLES[name],
                    linewidth=1.45 if name == "sindy_history" else 1.05,
                    marker=MARKERS[name], markersize=2.5, markeredgewidth=0.5,
                    markevery=(2*MODELS.index(name), 15), zorder=3 if name == "sindy_history" else 2)
        ax.set_title(f"({chr(97+index)}) {column_titles[index]}", loc="left", pad=4)
        ax.set_ylabel(labels[index], labelpad=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.margins(y=0.09)
        if index >= 6:
            ax.set_xlabel("Prediction time (s)")
    # Arrange the row-wise legend as Observed / Current / Linear, then Motor / History / Constant.
    legend_order = ["truth", "motor_sindy", "sindy_current", "sindy_history", "linear_history", "constant_twist"]
    handles = []
    for name in legend_order:
        if name == "truth":
            handles.append(Line2D([], [], color="black", linewidth=1.7, label=LABELS[name]))
        else:
            handles.append(Line2D([], [], color=COLORS[name], linestyle=STYLES[name], marker=MARKERS[name], markersize=3,
                                  linewidth=1.3, label=LABELS[name]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=3,
               frameon=False, columnspacing=1.8, handlelength=2.7, labelspacing=0.6)
    fig.text(0.53, 0.902, "Test flight 26 | start 333.01 s | dashed vertical line: 0.5 s", fontsize=7.5, ha="center")
    return fig


def main():
    time, saved, angles, sources, report = read_verified_example()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"data").mkdir(exist_ok=True)
    records = []
    for name, states in saved.items():
        for k, timestamp in enumerate(time):
            records.append({
                "model": "observed" if name == "truth" else name,
                "flight_id": 26, "origin_time_s": 333.01, "prediction_time_s": timestamp,
                **dict(zip(["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"], states[k, :6])),
                **dict(zip(["roll_rad", "pitch_rad", "yaw_rad"], angles[name][k])),
            })
    pd.DataFrame(records).to_csv(OUT/"data/section4a_prediction_trajectories.csv", index=False)
    fig = draw(time, saved, angles)
    pdf_path, png_path = OUT/f"{STEM}.pdf", OUT/f"{STEM}.png"
    fig.savefig(pdf_path, metadata={"Title": "Observed states and free-running model predictions", "Subject": "Pelican test flight 26; frozen representative example"})
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    with Image.open(png_path) as preview:
        preview.convert("L").save(OUT/f"{STEM}_grayscale.png", dpi=(600,600))
        report["png_pixels"] = list(preview.size)
    for path, before in sources.items():
        assert digest(path) == before, f"Frozen source changed: {path}"
    report["source_hashes_unchanged"] = True
    report["source_sha256"] = {str(p.relative_to(LATEX.parent)).replace("\\", "/"): h for p, h in sources.items()}
    report["figure_size_inches"] = [WIDTH_IN, 5.55]
    report["png_dpi"] = 600
    report["output_sha256"] = {path.name: digest(path) for path in [pdf_path, png_path]}
    (OUT/f"{STEM}_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ["source_sha256", "output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
