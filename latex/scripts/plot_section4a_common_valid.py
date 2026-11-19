"""Plot a two-second continuous prediction from the fixed E2 example.

All plotted paths share the requested two-second horizon. The original E2
termination guards remain active and can shorten the common interval.
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from plot_section4a_3d import draw
from plot_section4a_predictions import (
    LATEX, MODELS, OUT, RUN, WIDTH_IN, digest, read_verified_example,
)

METHOD = LATEX.parent / "method"
sys.path[:0] = [str(METHOD), str(METHOD / "scripts")]
from pelican_e2.model import context_for_episode, rollout
from run_pelican_e2 import restore_model
from verify_pelican_e2 import independent_rollout, load_episode, verify_freeze

STEM = "section4a_prediction_common_valid"
MAX_SECONDS = 2.0
HEIGHT_IN = 3.65


def main():
    _, old, _, sources, old_report = read_verified_example()
    freeze, splits = verify_freeze(RUN)
    meta = old_report["selection"]
    flight_id, origin = meta["flight_id"], meta["origin_index"]
    origin_time = meta["origin_t_s"]
    assert flight_id in splits["test"]
    config_path = RUN / "config.json"
    model_path = RUN / "models/selected_models.json"
    motor_path = RUN / "motor_model.json"
    episode_path = RUN / "data" / f"flight_{flight_id:02d}.npz"
    code_paths = [METHOD / p for p in freeze["code_sha256"]]
    for path in [config_path, model_path, motor_path, episode_path, *code_paths]:
        sources[path] = digest(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dt = config["dt"]
    steps = round(MAX_SECONDS / dt)
    extended = {**config, "horizons_s": sorted(set(config["horizons_s"] + [MAX_SECONDS]))}
    motor = json.loads(motor_path.read_text(encoding="utf-8"))
    models = {m["name"]: restore_model(m)
              for m in json.loads(model_path.read_text(encoding="utf-8"))}
    episode = load_episode(RUN, flight_id)
    assert origin + steps < len(episode["x"])
    paths = {"truth": episode["x"][origin:origin+steps+1].copy()}
    failures, checks = {}, {}
    for name in MODELS:
        initial, history, prior = context_for_episode(
            episode, np.array([origin]), models[name], extended)
        _, failure, predicted = rollout(
            initial, history, episode["u"][origin:origin+steps][None],
            prior, models[name], motor, extended, keep_all=True)
        states = predicted[0]
        reference, reference_failure = independent_rollout(
            episode, origin, models[name], motor, extended)
        # Quaternions q and -q represent the same attitude.
        finite = np.isfinite(reference).all(axis=1) & np.isfinite(states).all(axis=1)
        signs = np.where(np.sum(reference[finite, 6:10]*states[finite, 6:10], axis=1) < 0, -1, 1)
        reference[finite, 6:10] *= signs[:, None]
        assert reference_failure == int(failure[0]), name
        np.testing.assert_allclose(states, reference, rtol=2e-8, atol=2e-8, equal_nan=True)
        np.testing.assert_allclose(states[:101], old[name], rtol=1e-10, atol=1e-11)
        failures[name] = int(failure[0])
        if failures[name] <= steps:
            assert np.isnan(states[failures[name]:]).all()
        checks[name] = {
            "first_invalid_step": failures[name] if failures[name] <= steps else None,
            "first_invalid_time_s": round(failures[name]*dt, 8) if failures[name] <= steps else None,
            "independent_rollout_max_absolute_difference": float(np.nanmax(abs(states-reference))),
            "original_first_second_max_absolute_difference": float(np.max(abs(states[:101]-old[name]))),
        }
        paths[name] = states
    np.testing.assert_allclose(paths["truth"][:101], old["truth"], rtol=0, atol=0)

    last_step = min(steps, min(failures.values())-1)
    assert last_step >= 100
    time = np.arange(last_step+1)*dt
    saved = {name: states[:last_step+1].copy() for name, states in paths.items()}
    for name, states in saved.items():
        assert np.isfinite(states).all(), name
        if name != "truth":
            assert (np.linalg.norm(states[:, 3:6], axis=1) <= config["max_speed_m_s"]).all()
            assert (np.linalg.norm(states[:, 10:13], axis=1) <= config["max_body_rate_rad_s"]).all()
        np.testing.assert_allclose(np.linalg.norm(states[:, 6:10], axis=1), 1, atol=1e-12)
    limiting = [name for name, step in failures.items() if step == last_step+1 and step <= steps]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    npz_path = OUT / "data" / f"{STEM}.npz"
    csv_path = OUT / "data" / f"{STEM}.csv"
    np.savez_compressed(npz_path, time=time, **saved)
    rows = []
    for name, states in saved.items():
        for k, elapsed in enumerate(time):
            rows.append({"model": "observed" if name == "truth" else name,
                         "flight_id": flight_id, "elapsed_s": elapsed,
                         "log_time_s": origin_time+elapsed,
                         "x_m": states[k, 0], "y_m": states[k, 1], "z_m": states[k, 2]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    fig, bounds = draw(saved, time=time, origin_time=origin_time, intermediate_s=1.0, height_in=HEIGHT_IN)
    pdf_path, png_path = OUT / f"{STEM}.pdf", OUT / f"{STEM}.png"
    fig.savefig(pdf_path, metadata={
        "Title": "Observed and predicted positions over two seconds",
        "Subject": f"Pelican flight {flight_id}, {origin_time:.2f}-{origin_time+time[-1]:.2f} s",
    })
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    with Image.open(png_path) as preview:
        preview.convert("L").save(OUT / f"{STEM}_grayscale.png", dpi=(600, 600))
        pixels = list(preview.size)
    for path, before in sources.items():
        assert digest(path) == before, f"Source changed: {path}"
    report = {
        "status": "PASS", "selection": meta,
        "selection_rule": old_report["selection_rule"], "selection_rule_recomputed": True,
        "frozen_sources_verified": True, "source_hashes_unchanged": True,
        "original_saved_endpoints_checked": old_report["saved_endpoints_checked"],
        "extension_search_cap_s": MAX_SECONDS, "dt_s": dt,
        "plotted_duration_s": round(time[-1], 8), "plotted_steps": last_step,
        "samples_per_curve": len(time), "limiting_models": limiting,
        "log_start_s": origin_time, "log_end_s": round(origin_time+time[-1], 8),
        "termination_rule": "Nonfinite state, speed > 50 m/s, or body rate > 100 rad/s; unchanged from frozen E2.",
        "plot_cutoff_rule": f"One shared horizon of {MAX_SECONDS:g} s, shortened only if a model triggers the original termination guard.",
        "models": checks,
        "scope": "Illustrative extension of the fixed median example beyond the original 1 s E2 maximum horizon; original aggregate metrics unchanged.",
        "prediction_source": "Frozen raw E2 models; observed state/history initialized once; recorded commands supplied throughout. No measured-state resets or refitting.",
        "plot_transform": "Original world-frame positions, no smoothing, clipping or coordinate transformation.",
        "spatial_scale": "Equal metres per unit length on all axes; orthographic camera.",
        "axis_limits_m": bounds.tolist(),
        "marker_indices": {"start": 0, "1_s": round(1/dt), "end": last_step},
        "figure_size_inches": [WIDTH_IN, HEIGHT_IN], "png_dpi": 600, "png_pixels": pixels,
        "source_sha256": {str(p.relative_to(LATEX.parent)).replace("\\", "/"): h for p, h in sources.items()},
        "output_sha256": {p.name: digest(p) for p in [npz_path, csv_path, pdf_path, png_path]},
    }
    (OUT / f"{STEM}_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ["source_sha256", "output_sha256"]}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
