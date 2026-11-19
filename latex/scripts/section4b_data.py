"""Read and independently aggregate the frozen trajectory-search validation.

No source experiment is modified. All displayed results use untouched
validation initial states, after development-only reference selection.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

LATEX = Path(__file__).resolve().parents[1]
METHOD = LATEX.parent / "method"
RUN = METHOD / "runs/pelican_trajectory_search_v1"
KINDS = ("circle", "figure8", "helix")
ORDER = ("pd", "sindy_current", "linear_history", "motor_sindy", "sindy_history")
SEEDS = tuple(range(24301, 24306))
PATH_SEED = SEEDS[0]
LABELS = {
    "pd": "PD", "sindy_current": "SINDy-MPC",
    "linear_history": "Linear history MPC", "motor_sindy": "SINDy-MPC with Motor Dynamics",
    "sindy_history": "History SINDy-MPC",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validation():
    manifest_path = RUN / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    sources = {manifest_path: digest(manifest_path)}

    def register(path):
        key = path.relative_to(METHOD).as_posix()
        actual = digest(path)
        assert key in manifest, f"Missing frozen hash: {key}"
        assert actual == manifest[key]["sha256"], f"Frozen source changed: {key}"
        sources[path] = actual

    for name in ("config.json", "selection.json", "selected_parameters.csv",
                 "validation_summary.csv", "validation_comparisons.csv",
                 "validate/verification.json", "validate/tracking_per_run.csv"):
        register(RUN / name)
    settings = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    selection = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))
    audit = json.loads((RUN / "validate/verification.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["selection_provenance_verified"]
    assert audit["tracking_sha256"] == sources[RUN / "validate/tracking_per_run.csv"]
    assert selection["selection_stage"] == "development_only"
    assert tuple(selection["validation_seeds"]) == tuple(settings["validation_seeds"]) == SEEDS
    assert tuple(settings["controllers"]) == ORDER
    assert tuple(settings["trajectories"]) == KINDS
    assert settings["duration_s"] == 20.0
    assert set(SEEDS).isdisjoint(settings["development_seeds"] + settings["confirmation_seeds"])
    candidates = {c["trajectory"]: c for c in selection["candidates"]}
    for kind in KINDS:
        register(RUN / "selected_configs" / f"{kind}.json")
    saved = pd.read_csv(RUN / "validate/tracking_per_run.csv")
    expected = set(itertools.product(KINDS, SEEDS, ORDER))
    assert len(saved) == 75 and set(zip(saved.trajectory, saved.seed, saved.controller)) == expected
    assert saved.complete.all() and saved.tracking_success.all()
    for field in ("fallbacks", "solver_failures", "input_bound_violations", "input_slew_violations"):
        assert (saved[field] == 0).all()

    data, rows, position_diffs, variation_diffs = {}, [], [], []
    for row in saved.itertuples(index=False):
        assert row.candidate_id == candidates[row.trajectory]["candidate_id"]
        base = RUN / "validate/cases" / row.case_id
        for filename in ("trajectory.npz", "control.csv", "metrics.json"):
            register(base / filename)
        with np.load(base / "trajectory.npz", allow_pickle=False) as archive:
            d = {key: archive[key].copy() for key in
                 ("t", "physical", "reference", "command", "initial_history_commands")}
        np.testing.assert_allclose(d["t"], np.arange(2001) * .01, rtol=0, atol=1e-12)
        assert d["physical"].shape == (2001, 17) and d["reference"].shape == (2001, 13)
        assert all(np.isfinite(value).all() for value in d.values())
        error = np.linalg.norm(d["physical"][:, :3] - d["reference"][:, :3], axis=1)
        d["error_cm"] = 100 * error
        derived = dict(position_rmse_m=float(np.sqrt(np.mean(error**2))),
                       late_rmse_m=float(np.sqrt(np.mean(error[d["t"] >= 10]**2))),
                       max_position_error_m=float(error.max()),
                       final_position_error_m=float(error[-1]))
        metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
        for key, value in derived.items():
            for original in (getattr(row, key), metrics[key]):
                np.testing.assert_allclose(value, original, rtol=1e-10, atol=1e-12)
            position_diffs.append(abs(value - getattr(row, key)))
        controls = pd.read_csv(base / "control.csv")
        assert len(controls) == 400 and controls.success.all() and not controls.fallback.any()
        np.testing.assert_array_equal(controls.step, np.arange(0, 2000, 5))
        u = controls[[f"u{i}" for i in range(1, 5)]].to_numpy()
        np.testing.assert_allclose(u, d["command"][controls.step], rtol=1e-12, atol=1e-12)
        changes = np.diff(np.vstack([d["initial_history_commands"][-1], u]), axis=0)
        variation = float(np.sum(changes**2))
        np.testing.assert_allclose(variation, row.command_variation_squared, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(variation, metrics["command_variation_squared"], rtol=1e-10, atol=1e-10)
        variation_diffs.append(abs(variation - row.command_variation_squared))
        rows.append(dict(trajectory=row.trajectory, seed=row.seed, controller=row.controller,
                         candidate_id=row.candidate_id, complete=row.complete,
                         tracking_success=row.tracking_success, **derived,
                         command_variation_squared=variation, fallbacks=row.fallbacks,
                         solver_failures=row.solver_failures, control_calls=len(controls),
                         input_bound_violations=row.input_bound_violations,
                         input_slew_violations=row.input_slew_violations,
                         input_bound_active_fraction=row.input_bound_active_fraction,
                         input_slew_active_fraction=row.input_slew_active_fraction))
        data[(row.trajectory, row.seed, row.controller)] = d
    for kind in KINDS:
        ref = data[(kind, PATH_SEED, "pd")]["reference"]
        for seed in SEEDS:
            paired = data[(kind, seed, "pd")]
            for name in ORDER:
                d = data[(kind, seed, name)]
                np.testing.assert_array_equal(d["reference"], ref)
                np.testing.assert_array_equal(d["physical"][0], paired["physical"][0])
                np.testing.assert_array_equal(d["initial_history_commands"], paired["initial_history_commands"])

    per_run = pd.DataFrame(rows)
    group_rows, overall_rows = [], []
    original_summary = pd.read_csv(RUN / "validation_summary.csv").set_index(["trajectory", "controller"])
    for kind, name in itertools.product(KINDS, ORDER):
        g = per_run[(per_run.trajectory == kind) & (per_run.controller == name)]
        rec = dict(trajectory=kind, controller=name, n=len(g),
                   rmse_cm=g.position_rmse_m.mean() * 100,
                   rmse_std_cm=g.position_rmse_m.std(ddof=1) * 100,
                   late_rmse_cm=g.late_rmse_m.mean() * 100,
                   late_rmse_std_cm=g.late_rmse_m.std(ddof=1) * 100,
                   command_variation_mean_cu2=g.command_variation_squared.mean(),
                   command_variation_std_cu2=g.command_variation_squared.std(ddof=1))
        for key in ("rmse_cm", "rmse_std_cm", "late_rmse_cm"):
            np.testing.assert_allclose(rec[key], original_summary.loc[(kind, name), key], rtol=1e-10, atol=1e-10)
        group_rows.append(rec)
    by_trajectory = pd.DataFrame(group_rows)
    for name in ORDER:
        g = per_run[per_run.controller == name]
        rec = dict(controller=name, label=LABELS[name], n=len(g),
                   full_rmse_cm=g.position_rmse_m.mean() * 100,
                   late_rmse_cm=g.late_rmse_m.mean() * 100,
                   command_variation_mean_cu2=g.command_variation_squared.mean(),
                   completed=int(g.complete.sum()), successful=int(g.tracking_success.sum()),
                   fallbacks=int(g.fallbacks.sum()), solver_failures=int(g.solver_failures.sum()))
        for kind in KINDS:
            match = by_trajectory[(by_trajectory.trajectory == kind) & (by_trajectory.controller == name)].iloc[0]
            rec[f"{kind}_rmse_cm"] = match.rmse_cm
            rec[f"{kind}_rmse_std_cm"] = match.rmse_std_cm
        overall_rows.append(rec)
    overall = pd.DataFrame(overall_rows)
    report = {
        "status": "PASS", "source_run": RUN.relative_to(METHOD).as_posix(),
        "cases_checked": len(per_run), "samples_per_case": 2001,
        "position_metrics_checked": len(position_diffs),
        "command_variation_metrics_checked": len(variation_diffs),
        "max_position_metric_difference_m": float(max(position_diffs)),
        "max_command_variation_difference_cu2": float(max(variation_diffs)),
        "source_manifest_hashes_verified": len(sources) - 1,
        "selection_stage": "development_only", "selected_references": selection["candidates"],
        "paired_initial_states_verified": True, "references_identical_across_controllers": True,
        "trajectory_selection": "First frozen held-out validation seed; no outcome-based run selection.",
        "trajectory_selection_seed": PATH_SEED, "seeds": list(SEEDS), "time_range_s": [0, 20],
        "controller_order": list(ORDER), "reference_order": list(KINDS),
        "position_source": "Simulated physical positions.",
        "path_processing": "All 2001 samples; no smoothing, resampling or averaging of paths.",
        "error_statistic": "Mean of five per-run 3D position-error norms at each sample, in cm.",
        "error_band": "Minimum-maximum over five paired initial states; not a confidence interval.",
        "table_dispersion": "Sample standard deviation (ddof=1) across five initial states per trajectory.",
        "overall_weighting": "Equal mean of 15 run-level metrics per controller (three trajectories, five states).",
        "command_variation": "Sum of squared 4D control-command increments; includes final warm-up to first command.",
        "timing": "Excluded: four concurrent workers; no runtime comparisons from this batch.",
    }
    return data, per_run, by_trajectory, overall, sources, report


def verify_sources(sources, report):
    for path, before in sources.items():
        assert digest(path) == before, f"Source changed: {path}"
    report["source_hashes_unchanged"] = True
    report["source_sha256"] = {p.relative_to(METHOD).as_posix(): value for p, value in sources.items()}
