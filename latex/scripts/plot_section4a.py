"""Draw Section IV-A from frozen results; no fitting or control simulations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import pandas as pd
from PIL import Image


LATEX = Path(__file__).resolve().parents[1]
RUNS = LATEX.parent / "method" / "runs"
E2 = RUNS / "pelican_e2_v1"
E3 = RUNS / "pelican_e3_corrected_v1"
OUT = LATEX / "figures"
DATA = OUT / "data"
STEM = "section4a_prediction_ablation"
# IEEEtran's existing 43 pc text width, converted from TeX points to inches.
WIDTH_IN = 43 * 12 / 72.27
MODELS = ["sindy_current", "linear_history", "motor_sindy", "sindy_history", "constant_twist"]
LABELS = {
    "sindy_current": "SINDy",
    "linear_history": "Linear history",
    "motor_sindy": "SINDy with Motor Dynamics",
    "sindy_history": "History SINDy",
    "constant_twist": "Constant twist",
}
COLORS = {
    "sindy_current": "#D55E00", "linear_history": "#0072B2",
    "motor_sindy": "#CC79A7", "sindy_history": "#009E73",
    "constant_twist": "#666666",
}
STYLES = {
    "sindy_current": "--", "linear_history": "-.",
    "motor_sindy": (0, (5, 1, 1, 1)), "sindy_history": "-",
    "constant_twist": ":",
}
MARKERS = dict(zip(MODELS, ["s", "^", "D", "o", "x"]))
VARIANTS = ["original", "equilibrium", "online", "combined"]
VARIANT_LABELS = ["Original", "Calibration\nonly", "Online\nonly", "Combined"]
HORIZONS = np.array([0.01, 0.1, 0.25, 0.5, 1.0])
METRICS = ["position_rmse_m", "late_rmse_m"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate():
    source_paths = [
        E2 / "test/prediction_per_flight.csv",
        E2 / "test/prediction_summary.csv",
        E2 / "analysis/metric_intervals.csv",
        E2 / "analysis/bootstrap_draws.npz",
        E3 / "tracking_per_run.csv", E3 / "tracking_summary.csv",
    ]
    hashes = {p: digest(p) for p in source_paths}
    intervals = pd.read_csv(source_paths[2])
    prediction = intervals[(intervals.metric == "velocity") & intervals.model.isin(MODELS)].copy()
    flights = pd.read_csv(source_paths[0])
    summary = pd.read_csv(source_paths[1]).set_index(["model", "horizon_s"])
    with np.load(source_paths[3]) as saved:
        flight_ids, draws = saved["flight_ids"], saved["draws"]
    assert len(flight_ids) == 11 and draws.shape == (2000, 11)
    assert len(prediction) == 25
    max_prediction_delta = 0.0
    for name in MODELS:
        group = prediction[prediction.model == name].sort_values("horizon_s")
        np.testing.assert_allclose(group.horizon_s, HORIZONS)
        for row in group.itertuples():
            perflight = flights[(flights.model == name) & (flights.horizon_s == row.horizon_s)]
            perflight = perflight.set_index("flight_id").loc[flight_ids]
            values = perflight.velocity_rmse.to_numpy()
            ci = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
            recomputed = np.r_[values.mean(), ci]
            stored = np.array([row.mean, row.lower_95, row.upper_95])
            np.testing.assert_allclose(recomputed, stored, rtol=1e-11, atol=1e-13)
            max_prediction_delta = max(max_prediction_delta, float(np.max(abs(recomputed-stored))))
            aggregate = summary.loc[(name, row.horizon_s)]
            np.testing.assert_allclose(row.mean, aggregate.velocity_rmse_mean, rtol=1e-11)
            assert perflight.windows.sum() == aggregate.windows == 2515
            assert perflight.failed_windows.sum() == row.failed_windows == aggregate.failed_windows
            expected_failures = {"sindy_current": 2, "motor_sindy": 48}.get(name, 0) if row.horizon_s == 1 else 0
            assert row.failed_windows == expected_failures
            assert bool(row.conditional_on_success) == bool(expected_failures)

    runs = pd.read_csv(source_paths[4])
    runs = runs[runs.scenario == "nominal"].copy()
    assert len(runs) == 32 and runs.complete.eq(True).all()
    assert not runs.duplicated(["variant", "trajectory", "seed"]).any()
    assert set(runs.trajectory) == {"hover", "circle", "figure8", "helix"}
    assert set(runs.seed) == {11301, 11302}
    max_tracking_delta = 0.0
    for row in runs.itertuples():
        path = E3 / "cases" / f"nominal_{row.trajectory}_{row.seed}_{row.variant}" / "trajectory.npz"
        hashes[path] = digest(path)
        with np.load(path) as raw:
            times = raw["t"]
            displacement = raw["physical"][:, :3] - raw["reference"][:, :3]
        np.testing.assert_allclose(times, np.arange(2001) * 0.01, rtol=0, atol=1e-10)
        squared_error = np.sum(displacement**2, axis=1)
        recovered = np.array([np.sqrt(squared_error.mean()), np.sqrt(squared_error[times >= 10.0].mean())])
        recorded = np.array([row.position_rmse_m, row.late_rmse_m])
        np.testing.assert_allclose(recovered, recorded, rtol=1e-10, atol=1e-12)
        max_tracking_delta = max(max_tracking_delta, float(np.max(abs(recovered-recorded))))

    counts = runs.groupby(["variant", "seed"]).size()
    assert counts.eq(4).all() and len(counts) == 8
    by_seed = runs.groupby(["variant", "seed"])[METRICS].mean().reset_index()
    group_means = runs.groupby("variant")[METRICS].mean()
    stored_tracking = pd.read_csv(source_paths[5])
    stored_tracking = stored_tracking[stored_tracking.scenario == "nominal"].set_index("variant")
    for variant in VARIANTS:
        seed_mean = by_seed[by_seed.variant == variant][METRICS].mean().to_numpy()
        np.testing.assert_allclose(seed_mean, group_means.loc[variant].to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(seed_mean, stored_tracking.loc[variant, METRICS].to_numpy(dtype=float), rtol=1e-12)

    ablation_rows = []
    for variant in VARIANTS:
        for metric, period in zip(METRICS, ["0-20 s", "10-20 s"]):
            values = by_seed.loc[by_seed.variant == variant, metric].to_numpy() * 100
            ablation_rows.append({
                "variant": variant, "period": period, "mean_cm": values.mean(),
                "min_seed_mean_cm": values.min(), "max_seed_mean_cm": values.max(),
                "paired_initial_states": 2, "references_per_initial_state": 4, "trials": 8,
            })
    report = {
        "status": "PASS", "prediction_points": 25, "test_flights": 11,
        "common_prediction_origins": 2515, "bootstrap_draws_reused": 2000,
        "prediction_max_absolute_check_difference": max_prediction_delta,
        "ablation_trajectories_recomputed": 32,
        "ablation_max_absolute_check_difference_m": max_tracking_delta,
        "ablation_aggregation": "Arithmetic mean of per-trial RMSE; four references per initial state, then two initial states with equal weight.",
        "ablation_range": "Range of two reference-averaged initial-state results, not a confidence interval.",
        "regression_or_simulation_rerun": False,
    }
    return prediction, by_seed, pd.DataFrame(ablation_rows), runs, hashes, report


def make_figure(prediction, by_seed):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 8, "legend.fontsize": 7.4, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig = plt.figure(figsize=(WIDTH_IN, 3.15))
    ax_prediction = fig.add_axes([0.078, 0.205, 0.405, 0.54])
    ax_ablation = fig.add_axes([0.67, 0.205, 0.31, 0.54])
    fig.text(0.078, 0.965, "(a) Multi-step prediction", fontsize=9, weight="bold", va="top")
    fig.text(0.575, 0.965, "(b) Calibration and bias compensation", fontsize=9, weight="bold", va="top")

    for name in MODELS:
        group = prediction[prediction.model == name].sort_values("horizon_s")
        x = group.horizon_s.to_numpy()
        y = group["mean"].to_numpy()
        ax_prediction.fill_between(x, group.lower_95.to_numpy(), group.upper_95.to_numpy(), color=COLORS[name], alpha=0.10, linewidth=0)
        ax_prediction.plot(x, y, color=COLORS[name], linestyle=STYLES[name], linewidth=1.25)
        for row in group.itertuples():
            ax_prediction.plot(row.horizon_s, row.mean, marker=MARKERS[name], markersize=4.0,
                               color=COLORS[name], markerfacecolor="white" if row.failed_windows else COLORS[name],
                               markeredgewidth=0.9, linestyle="none", zorder=4)
    handles = [Line2D([], [], color=COLORS[name], linestyle=STYLES[name], marker=MARKERS[name],
                      markersize=4, linewidth=1.2, label=LABELS[name]) for name in MODELS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.078, 0.925), ncol=2,
               frameon=False, borderaxespad=0, columnspacing=1.0, handlelength=2.1, labelspacing=0.4)
    ax_prediction.set(xscale="log", yscale="log", xlim=(0.008, 1.18), ylim=(0.001, 6),
                      xlabel="Prediction horizon (s)", ylabel="Velocity endpoint RMSE (m/s)")
    ax_prediction.set_xticks(HORIZONS, labels=["0.01", "0.1", "0.25", "0.5", "1"])
    ax_prediction.xaxis.set_minor_locator(NullLocator())
    ax_prediction.yaxis.set_minor_locator(NullLocator())
    ax_prediction.grid(axis="y", color="0.88", linewidth=0.5)

    period_specs = [
        ("position_rmse_m", "0-20 s", "#0072B2", "o", 0.17),
        ("late_rmse_m", "10-20 s", "#D55E00", "s", -0.17),
    ]
    for j, variant in enumerate(VARIANTS):
        for metric, label, color, marker, offset in period_specs:
            values = by_seed.loc[by_seed.variant == variant, metric].to_numpy() * 100
            center = values.mean()
            y = 3 - j + offset
            ax_ablation.plot([values.min(), values.max()], [y, y], color=color, linewidth=1)
            ax_ablation.plot(values, [y, y], marker=marker, markersize=4.2, markerfacecolor="white",
                             markeredgecolor=color, markeredgewidth=0.9, linestyle="none", zorder=3)
            ax_ablation.plot(center, y, marker=marker, markersize=5.1, color=color,
                             markeredgecolor="white", markeredgewidth=0.3, linestyle="none", zorder=4)
    period_handles = [Line2D([], [], color=color, marker=marker, linestyle="-", linewidth=1,
                             markersize=4.5, label=label) for _, label, color, marker, _ in period_specs]
    fig.legend(handles=period_handles, loc="upper left", bbox_to_anchor=(0.67, 0.918), ncol=2,
               frameon=False, borderaxespad=0, columnspacing=0.9, handlelength=1.3)
    fig.text(0.67, 0.825, "4 references, 2 paired initial states", fontsize=7.2)
    ax_ablation.set(xscale="log", xlim=(0.35, 22), ylim=(-0.6, 3.6),
                    xlabel="Position RMSE (cm, log scale)")
    ax_ablation.set_xticks([0.5, 1, 2, 5, 10, 20], labels=["0.5", "1", "2", "5", "10", "20"])
    ax_ablation.xaxis.set_minor_locator(NullLocator())
    ax_ablation.xaxis.set_minor_formatter(NullFormatter())
    ax_ablation.set_yticks([3, 2, 1, 0], labels=VARIANT_LABELS)
    ax_ablation.tick_params(axis="y", length=0, pad=7)
    ax_ablation.spines["left"].set_visible(False)
    ax_ablation.grid(axis="x", color="0.88", linewidth=0.5)
    for boundary in [0.5, 1.5, 2.5]:
        ax_ablation.axhline(boundary, color="0.93", linewidth=0.6, zorder=0)

    fig.text(0.078, 0.045, "Shading: 95% flight bootstrap. At 1 s, hollow markers exclude failed windows: Current 2, Motor 48.", fontsize=7.0)
    fig.text(0.67, 0.78, "Filled: mean; open: initial-state means", fontsize=6.8)
    return fig


def main():
    prediction, by_seed, ablation, runs, hashes, report = load_and_validate()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    prediction.to_csv(DATA / "section4a_prediction.csv", index=False)
    by_seed.to_csv(DATA / "section4a_ablation_per_initial_state_m.csv", index=False)
    ablation.to_csv(DATA / "section4a_ablation_summary_cm.csv", index=False)
    runs[["variant", "trajectory", "seed"] + METRICS].to_csv(DATA / "section4a_ablation_per_trial_m.csv", index=False)
    fig = make_figure(prediction, by_seed)
    pdf_path, png_path = OUT / f"{STEM}.pdf", OUT / f"{STEM}.png"
    fig.savefig(pdf_path, metadata={"Title": "Prediction and compensation ablation", "Subject": "Frozen Pelican results for Section IV-A"})
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    with Image.open(png_path) as im:
        im.convert("L").save(OUT / f"{STEM}_grayscale.png", dpi=(600, 600))
        report["png_pixels"] = list(im.size)
    for path, before in hashes.items():
        assert digest(path) == before, f"Frozen source changed: {path}"
    report.update({"source_hashes_unchanged": True, "source_sha256": {
        str(p.relative_to(LATEX.parent)).replace("\\", "/"): value for p, value in hashes.items()
    }, "figure_size_inches": [WIDTH_IN, 3.15], "png_dpi": 600})
    report["output_sha256"] = {p.name: digest(p) for p in [pdf_path, png_path]}
    (OUT / f"{STEM}_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ["source_sha256", "output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
