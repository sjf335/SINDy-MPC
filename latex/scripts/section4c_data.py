"""Verify and aggregate frozen E4 records for manuscript Section IV-C.

All time windows, failures, and recovery criteria follow the original protocol.
The original experiment, models, and controller settings are read-only.
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
RUN = METHOD / "runs/pelican_e4_v1"
KINDS = ("circle", "figure8")
ORDER = ("pd", "sindy_current", "linear_history", "motor_sindy", "sindy_history")
SCENARIOS = ("nominal", "tau_1p5", "tau_2", "force_pulse")
SEEDS = tuple(range(15301, 15306))
METRICS = ("position_rmse_m", "late_rmse_m", "event_peak_error_m",
           "post_event_rmse_m", "recovery_time_s")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    manifest_path = RUN / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    sources = {manifest_path: digest(manifest_path)}

    def register(path):
        key = path.relative_to(METHOD).as_posix()
        actual = digest(path)
        assert key in manifest and actual == manifest[key]["sha256"], f"Changed source: {key}"
        sources[path] = actual

    for name in ("config.json", "e3_config.json", "verification.json", "robustness_per_run.csv",
                 "robustness_summary.csv", "robustness_by_trajectory.csv"):
        register(RUN / name)
    settings = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    cfg = json.loads((RUN / "e3_config.json").read_text(encoding="utf-8"))
    audit = json.loads((RUN / "verification.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["tracking_sha256"] == sources[RUN / "robustness_per_run.csv"]
    assert tuple(settings["formal"]["seeds"]) == SEEDS
    assert tuple(settings["formal"]["trajectories"]) == KINDS
    assert tuple(settings["controllers"]) == ORDER
    assert cfg["dt"] == .01 and cfg["duration_s"] == 20
    assert (settings["force_start_s"], settings["force_end_s"], settings["post_event_start_s"],
            settings["recovery_error_m"], settings["recovery_dwell_s"]) == (8, 12, 14, .03, 1)
    saved = pd.read_csv(RUN / "robustness_per_run.csv")
    assert len(saved) == 250
    frame = saved[saved.scenario.isin(SCENARIOS)].copy()
    expected = set(itertools.product(SCENARIOS, KINDS, SEEDS, ORDER))
    actual = set(zip(frame.scenario, frame.trajectory, frame.seed, frame.controller))
    assert len(frame) == 200 and actual == expected
    data, rows, differences = {}, [], []
    recovery_checks = 0
    for row in frame.itertuples(index=False):
        path = RUN / "cases" / f"{row.scenario}_{row.trajectory}_{row.seed}_{row.controller}"
        for filename in ("trajectory.npz", "metrics.json"):
            register(path / filename)
        with np.load(path / "trajectory.npz", allow_pickle=False) as z:
            d = {key: z[key].copy() for key in ("t", "physical", "reference", "world_acceleration")}
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        t = d["t"]
        np.testing.assert_allclose(t, np.arange(len(t)) * .01, rtol=0, atol=1e-12)
        assert all(np.isfinite(value).all() for value in d.values())
        assert d["physical"].shape == (len(t), 17)
        assert d["reference"].shape == (len(t), 13)
        complete = len(t) == 2001
        assert complete == bool(row.complete) == metrics["complete"]
        np.testing.assert_allclose(t[-1], row.duration_completed_s, rtol=0, atol=1e-12)
        error = np.linalg.norm(d["physical"][:, :3] - d["reference"][:, :3], axis=1)
        d["error_cm"] = error * 100
        d["complete"] = complete
        event = (t >= 8) & (t <= 12)  # Original metric includes the removal endpoint.
        after = t >= 14
        derived = {
            "prefix_position_rmse_m": float(np.sqrt(np.mean(error**2))),
            "position_rmse_m": float(np.sqrt(np.mean(error**2))) if complete else np.nan,
            "late_rmse_m": float(np.sqrt(np.mean(error[t >= 10]**2))) if complete else np.nan,
            "event_peak_error_m": float(error[event].max()) if event.any() else np.nan,
            "post_event_rmse_m": float(np.sqrt(np.mean(error[after]**2))) if complete else np.nan,
            "max_position_error_m": float(error.max()),
            "final_position_error_m": float(error[-1]),
        }
        recovery = np.nan
        for k in np.flatnonzero(t >= 12):
            if k + 101 > len(t):
                break
            if np.all(error[k:k+101] <= .03):
                recovery = float(t[k] - 12)
                break
        recovered = np.isfinite(recovery)
        assert recovered == bool(row.recovered) == metrics["recovered"]
        derived["recovery_time_s"] = recovery
        for key, value in derived.items():
            np.testing.assert_allclose(value, getattr(row, key), rtol=1e-10, atol=1e-12, equal_nan=True)
            original = metrics[key] if metrics[key] is not None else np.nan
            np.testing.assert_allclose(value, original, rtol=1e-10, atol=1e-12, equal_nan=True)
            if np.isfinite(value):
                differences.append(abs(value - getattr(row, key)))
        if row.scenario == "force_pulse":
            recovery_checks += 1
            applied = np.zeros_like(d["world_acceleration"])
            assert len(applied) == len(t) - 1
            applied[(t[:-1] >= 8) & (t[:-1] < 12)] = [.4, -.25, .15]
            np.testing.assert_allclose(d["world_acceleration"], applied, rtol=0, atol=1e-12)
        else:
            np.testing.assert_allclose(d["world_acceleration"], 0, rtol=0, atol=1e-12)
        rows.append(dict(scenario=row.scenario, trajectory=row.trajectory, seed=row.seed,
                         controller=row.controller, complete=complete,
                         tracking_success=row.tracking_success, recovered=recovered,
                         termination_reason=row.termination_reason, duration_s=float(t[-1]),
                         fallbacks=row.fallbacks, solver_failures=row.solver_failures,
                         input_bound_violations=row.input_bound_violations,
                         input_slew_violations=row.input_slew_violations, **derived))
        data[(row.scenario, row.trajectory, row.seed, row.controller)] = d

    for kind, seed in itertools.product(KINDS, SEEDS):
        paired = data[("nominal", kind, seed, "pd")]
        for scenario, name in itertools.product(SCENARIOS, ORDER):
            d = data[(scenario, kind, seed, name)]
            # Equality across scenarios is guaranteed per controller; feedback initialization
            # in the original runner can produce controller-specific t=0 roundoff.
            nominal = data[("nominal", kind, seed, name)]
            np.testing.assert_array_equal(d["physical"][0], nominal["physical"][0])
            np.testing.assert_allclose(d["physical"][0], paired["physical"][0], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(d["reference"], paired["reference"][:len(d["t"])])
            if scenario == "force_pulse":
                before = d["t"] <= 8
                np.testing.assert_allclose(d["physical"][before], nominal["physical"][before],
                                           rtol=0, atol=1e-12)
    per_run = pd.DataFrame(rows)
    summary = aggregate(per_run, ("scenario", "controller"))
    by_trajectory = aggregate(per_run, ("scenario", "trajectory", "controller"))
    for original_name, new, keys in (
            ("robustness_summary.csv", summary, ["scenario", "controller"]),
            ("robustness_by_trajectory.csv", by_trajectory, ["scenario", "trajectory", "controller"])):
        old = pd.read_csv(RUN / original_name).set_index(keys)
        for row in new.to_dict("records"):
            key = tuple(row[x] for x in keys)
            assert row["completed"] == old.loc[key, "completed"]
            assert row["recovered_cases"] == old.loc[key, "recovered_cases"]
            for metric in METRICS:
                np.testing.assert_allclose(row[metric + "_mean"], old.loc[key, metric],
                                           rtol=1e-10, atol=1e-12, equal_nan=True)
    report = dict(
        status="PASS", source_run=RUN.relative_to(METHOD).as_posix(), cases_checked=len(per_run),
        completed_cases=int(per_run.complete.sum()), source_manifest_hashes_verified=len(sources)-1,
        finite_metrics_checked=len(differences), max_metric_absolute_difference=float(max(differences)),
        pulse_recovery_criteria_checked=recovery_checks, paired_initial_states_verified=True,
        references_identical_across_controllers=True, pre_pulse_trajectories_match_nominal=True,
        seeds=list(SEEDS), scenarios=list(SCENARIOS), controller_order=list(ORDER),
        full_interval_s=[0, 20], late_interval_s=[10, 20],
        pulse_applied_interval="8 <= t < 12 seconds",
        peak_metric_interval="8 <= t <= 12 seconds, matching the frozen E4 definition",
        post_event_interval_s=[14, 20],
        recovery="Time after 12 s to first 101 consecutive samples <= 0.03 m (one second).",
        missing_metrics="No full/late RMSE for incomplete groups; no recovery mean unless all runs recover.",
        aggregation="Equal weight per run: five states per trajectory or ten runs across both trajectories.",
        dispersion="Sample standard deviation (ddof=1); figures use minimum-maximum ranges.",
        primary_rows_excluded="pose_noise: summarized separately in the manuscript from the paired noise and estimation diagnostics.",
        source_e3_config=cfg,
    )
    return data, per_run, summary, by_trajectory, sources, report


def aggregate(frame, keys):
    rows = []
    for key, g in frame.groupby(list(keys), sort=False):
        r = dict(zip(keys, key))
        r.update(cases=len(g), completed=int(g.complete.sum()), successes=int(g.tracking_success.sum()),
                 recovered_cases=int(g.recovered.sum()), fallbacks=int(g.fallbacks.sum()),
                 solver_failures=int(g.solver_failures.sum()),
                 input_bound_violations=int(g.input_bound_violations.sum()),
                 input_slew_violations=int(g.input_slew_violations.sum()))
        for metric in METRICS:
            valid = g.complete.all() and (metric != "recovery_time_s" or g.recovered.all())
            for suffix, value in (("mean", g[metric].mean()), ("std", g[metric].std(ddof=1)),
                                  ("min", g[metric].min()), ("max", g[metric].max())):
                r[metric + "_" + suffix] = float(value) if valid else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def finalize_report(sources, report, path):
    for source, expected in sources.items():
        assert digest(source) == expected, f"Source changed: {source}"
    report["source_hashes_unchanged"] = True
    report["source_sha256"] = {p.relative_to(METHOD).as_posix(): v for p, v in sources.items()}
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
