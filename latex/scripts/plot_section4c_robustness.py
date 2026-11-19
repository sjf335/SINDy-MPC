"""Generate Figure 4, Table IV, and verified data from frozen E4 records."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from plot_section4b_tracking import COLORS, STYLES, LABELS, WIDTH_IN, configure_style, add_legend, line_width
from section4c_data import KINDS, ORDER, SEEDS, LATEX, digest, load_data, finalize_report

FIG = LATEX / "figures"
DATA = FIG / "data"
TABLE = LATEX / "tables"
TAU_SCENARIOS = ("nominal", "tau_1p5", "tau_2")


def axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(pad=2, length=2.5, width=.6)
    ax.grid(True)


def motor_figure(groups):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 3.85), sharex=True, sharey="row")
    fig.subplots_adjust(left=.095, right=.985, top=.92, bottom=.27, hspace=.29, wspace=.16)
    indexed = groups.set_index(["scenario", "trajectory", "controller"])
    limits = []
    for row, metric in enumerate(("position_rmse_m", "late_rmse_m")):
        ymax = 100 * groups.loc[groups.scenario.isin(TAU_SCENARIOS), metric + "_max"].max()
        upper = float(np.ceil(ymax * (1.20 if row == 0 else 1.08) * 2) / 2)
        limits.append([0, upper])
        for col, kind in enumerate(KINDS):
            ax = axes[row, col]
            for name in ORDER:
                g = [indexed.loc[(scenario, kind, name)] for scenario in TAU_SCENARIOS]
                mean = np.array([v[metric + "_mean"] for v in g]) * 100
                low = np.array([v[metric + "_min"] for v in g]) * 100
                high = np.array([v[metric + "_max"] for v in g]) * 100
                # Missing RMSE at x2 remains missing; no invented failure-height value.
                ax.fill_between([1, 1.5, 2], low, high, color=COLORS[name], alpha=.09, lw=0)
                ax.plot([1, 1.5, 2], mean, color=COLORS[name], ls=STYLES[name],
                        marker="o", ms=3.3, lw=line_width(name),
                        zorder=5 if name == "sindy_history" else 3)
            ax.set(xlim=(.95, 2.05), ylim=(0, upper))
            ax.set_xticks([1, 1.5, 2], ["1", "1.5", "2"])
            axes_style(ax)
            if row == 0:
                ax.set_title(("Circle", "Figure eight")[col], pad=8)
                ax.text(.98, .97, "SINDy with Motor Dynamics: 0/5 completed (2×)",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        color=COLORS["motor_sindy"],
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5))
            else:
                ax.set_xlabel("Multiplier of motor time constants", labelpad=3)
        axes[row, 0].set_ylabel("Position RMSE (cm)\n" + ("0–20 s" if row == 0 else "10–20 s"))
    add_legend(fig, include_reference=False)
    return fig, dict(layout="2 rows x 2 columns; full/late RMSE, circle/figure-eight",
                     y_limits_cm=limits, x_values=[1, 1.5, 2],
                     band="Minimum-maximum of five run-level RMSEs",
                     failure_annotation="SINDy with Motor Dynamics: 0/5 completed at x2 in each trajectory.",
                     failed_rmse="Missing; not replaced by a value or connected through.")


def pulse_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.05), sharey="row")
    fig.subplots_adjust(left=.09, right=.985, top=.92, bottom=.245, hspace=.46, wspace=.16)
    ymax = max(data[("force_pulse", kind, seed, name)]["error_cm"].max()
               for kind in KINDS for seed in SEEDS for name in ORDER)
    upper = float(np.ceil(ymax * 1.08 / 2) * 2)
    clipped = []
    for col, kind in enumerate(KINDS):
        ax, zoom = axes[:, col]
        title = ("Circle", "Figure eight")[col]
        ax.set_title(title, pad=7)
        zoom.set_title(f"{title} (14–20 s)", pad=7)
        ax.axvspan(8, 12, color="#D9D9D9", alpha=.5, lw=0, zorder=0)
        ax.add_patch(Rectangle((14, 0), 6, 4, fill=False, ec="#555555", lw=.7, ls=(0, (3, 2)), zorder=7))
        for name in ORDER:
            group = [data[("force_pulse", kind, seed, name)] for seed in SEEDS]
            assert all(d["complete"] for d in group)
            t = group[0]["t"]
            errors = np.stack([d["error_cm"] for d in group])
            mean, lo, hi = errors.mean(0), errors.min(0), errors.max(0)
            for a in (ax, zoom):
                a.fill_between(t, lo, hi, color=COLORS[name], alpha=.09, lw=0, zorder=1)
                a.plot(t, mean, color=COLORS[name], ls=STYLES[name], lw=line_width(name),
                       zorder=5 if name == "sindy_history" else 3)
            count = int(np.sum(errors[:, t >= 14] > 4))
            if count:
                clipped.append(dict(trajectory=kind, controller=name, samples_above_zoom=count,
                                    samples_in_zoom_interval=int(errors[:, t >= 14].size)))
        for a in (ax, zoom):
            a.axhline(3, color="black", ls=(0, (5, 4)), lw=.7, zorder=2)
            axes_style(a)
        ax.set(xlim=(0, 20), ylim=(0, upper))
        ax.set_xticks([0, 4, 8, 12, 16, 20])
        zoom.set(xlim=(14, 20), ylim=(0, 4), xlabel="Time (s)")
        zoom.set_xticks([14, 16, 18, 20])
        zoom.set_yticks([0, 1, 2, 3, 4])
        # Panel identifiers remain outside the plotting area beside the titles.
        for panel, letter in ((ax, "(a)" if col == 0 else "(b)"),
                              (zoom, "(c)" if col == 0 else "(d)")):
            panel.text(0, 1.035, letter, transform=panel.transAxes,
                       ha="left", va="bottom", fontsize=8, clip_on=False)
    axes[0, 0].set_ylabel("Position error (cm)")
    axes[1, 0].set_ylabel("Position error (cm)")
    add_legend(fig, include_reference=False)
    return fig, dict(layout="2 columns: circle/figure-eight; full response above and declared zoom below",
                     panel_labels="(a)-(d) above the axes, outside the plotting area",
                     panel_titles=["Circle", "Figure eight", "Circle (14–20 s)", "Figure eight (14–20 s)"],
                     disturbance_annotation="Shaded interval; defined in caption",
                     main_xlim_s=[0, 20], main_ylim_cm=[0, upper],
                     zoom_xlim_s=[14, 20], zoom_ylim_cm=[0, 4],
                     zoom_clipped_samples=clipped,
                     zoom_policy="Same zoom for all controllers; all values retained in main panels and CSV.",
                     mean="Mean of five 3D error norms at each sample; no smoothing.",
                     band="Minimum-maximum over five paired initial states; not a confidence interval.",
                     threshold_cm=3, disturbance_interval_s=[8, 12])


def save_figure(fig, stem):
    pdf, png = FIG / f"{stem}.pdf", FIG / f"{stem}.png"
    fig.savefig(pdf, metadata={"Title": stem, "Subject": "Frozen E4 physical simulation results"})
    fig.savefig(png, dpi=600)
    size = fig.get_size_inches().tolist()
    plt.close(fig)
    with Image.open(png) as im:
        im.convert("L").save(FIG / f"{stem}_grayscale.png", dpi=(600, 600))
        pixels = list(im.size)
    return dict(size_inches=size, png_pixels=pixels, png_dpi=600,
                sha256={p.name: digest(p) for p in (pdf, png)})


def table_tex(summary):
    index = summary.set_index(["scenario", "controller"])

    def cell(scenario, name, metric):
        r = index.loc[(scenario, name)]
        v = r[metric + "_mean"]
        if not np.isfinite(v):
            return r"---"
        unit = 1 if metric == "recovery_time_s" else 100
        txt = f"{unit*v:.2f}" + r"\pm" + f"{unit*r[metric + '_std']:.2f}"
        best = summary.loc[summary.scenario == scenario, metric + "_mean"].min()
        if np.isclose(v, best, rtol=0, atol=1e-12):
            txt = r"\boldsymbol{" + txt + "}"
        return "$" + txt + "$"

    lines = [
        r"% Generated by scripts/plot_section4c_robustness.py from frozen E4 records.",
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{Tracking under Motor Mismatch and External Disturbance}",
        r"  \label{tab:robustness}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{1.10}",
        r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}",
        r"    \toprule",
        r"    \multicolumn{8}{c}{(a) Mismatch in motor time constants: position RMSE (cm)} \\",
        r"    \midrule",
        r"    & \multicolumn{2}{c}{Nominal} & \multicolumn{2}{c}{$1.5\times$} & \multicolumn{2}{c}{$2\times$} & Completed \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"    Controller & 0--20 s & 10--20 s & 0--20 s & 10--20 s & 0--20 s & 10--20 s & ($2\times$) \\",
        r"    \midrule",
    ]
    for name in ORDER:
        label = ("SINDy-MPC (MD)"
                 if name == "motor_sindy" else LABELS[name])
        values = [label]
        for scenario in TAU_SCENARIOS:
            values += [cell(scenario, name, key) for key in ("position_rmse_m", "late_rmse_m")]
        values += [f"{int(index.loc[('tau_2', name), 'completed'])}/10"]
        lines.append("    " + " & ".join(values) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular*}",
        r"  \par\medskip",
        r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"    \toprule",
        r"    \multicolumn{6}{c}{(b) External disturbance over $[8,12)$ s} \\",
        r"    \midrule",
        r"    Controller & RMSE, 0--20 s & Peak, 8--12 s & RMSE, 14--20 s & Recovery time & Recovered \\",
        r"    & (cm) & (cm) & (cm) & (s) & \\",
        r"    \midrule",
    ]
    for name in ORDER:
        label = ("SINDy-MPC (MD)"
                 if name == "motor_sindy" else LABELS[name])
        values = [label] + [cell("force_pulse", name, key) for key in
                                  ("position_rmse_m", "event_peak_error_m", "post_event_rmse_m", "recovery_time_s")]
        values += [f"{int(index.loc[('force_pulse', name), 'recovered_cases'])}/10"]
        lines.append("    " + " & ".join(values) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular*}",
        r"  \par\smallskip",
        r"  \parbox{\textwidth}{\footnotesize",
        r"    Values are mean $\pm$ sample standard deviation over ten trials",
        r"    (circle and figure eight, five paired initial conditions each).",
        r"    MD denotes motor dynamics.",
        r"    Bold denotes the lowest mean in each column.",
        r"    All nominal, $1.5\times$, and disturbance trials complete.",
        r"    A dash indicates early termination in panel (a) or no sustained recovery in panel (b).",
        r"    Recovery is measured from 12~s to the start of the first continuous",
        r"    1~s interval with position error at most 3~cm.}",
        r"\end{table*}",
    ]
    (TABLE / "section4c_robustness.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    motor_rows, pulse_rows = [], []
    for name in ORDER:
        m = dict(controller=name, label=LABELS[name], trials_per_condition=10)
        for scenario in TAU_SCENARIOS:
            r = index.loc[(scenario, name)]
            m[scenario + "_completed"] = int(r.completed)
            for metric, short in (("position_rmse_m", "full_rmse"), ("late_rmse_m", "late_rmse")):
                for stat in ("mean", "std"):
                    m[f"{scenario}_{short}_{stat}_cm"] = r[metric + "_" + stat] * 100
        motor_rows.append(m)
        r = index.loc[("force_pulse", name)]
        p = dict(controller=name, label=LABELS[name], trials=10, completed=int(r.completed),
                 recovered=int(r.recovered_cases))
        for metric, short in (("position_rmse_m", "full_rmse"), ("event_peak_error_m", "pulse_peak"),
                              ("post_event_rmse_m", "post_event_rmse"), ("recovery_time_s", "recovery_time")):
            unit, multiplier = ("s", 1) if metric == "recovery_time_s" else ("cm", 100)
            for stat in ("mean", "std"):
                p[f"{short}_{stat}_{unit}"] = r[metric + "_" + stat] * multiplier
        pulse_rows.append(p)
    pd.DataFrame(motor_rows).to_csv(TABLE / "section4c_table_iv_motor.csv", index=False)
    pd.DataFrame(pulse_rows).to_csv(TABLE / "section4c_table_iv_disturbance.csv", index=False)


def paper_units(frame):
    result = frame.copy()
    for col in frame.columns:
        if col.endswith("_m") or "_m_" in col:
            new = col.replace("_m_", "_cm_")
            if new.endswith("_m"):
                new = new[:-2] + "_cm"
            result[new] = result.pop(col) * 100
    return result


def main():
    for directory in (FIG, DATA, TABLE):
        directory.mkdir(exist_ok=True)
    data, per_run, summary, by_trajectory, sources, report = load_data()
    paper_units(per_run).to_csv(TABLE / "section4c_per_run.csv", index=False)
    paper_units(summary).to_csv(TABLE / "section4c_summary.csv", index=False)
    paper_units(by_trajectory).to_csv(TABLE / "section4c_by_trajectory.csv", index=False)
    failures = per_run[~per_run.complete].copy()
    paper_units(failures).to_csv(TABLE / "section4c_terminated_runs.csv", index=False)
    paper_units(by_trajectory[by_trajectory.scenario.isin(TAU_SCENARIOS)]).to_csv(
        DATA / "section4c_motor_sensitivity.csv", index=False)
    curves, per_run_curves = [], []
    for kind in KINDS:
        for name in ORDER:
            group = [data[("force_pulse", kind, seed, name)] for seed in SEEDS]
            errors = np.stack([d["error_cm"] for d in group])
            curves.append(pd.DataFrame(dict(trajectory=kind, controller=name, time_s=group[0]["t"],
                mean_error_cm=errors.mean(0), min_error_cm=errors.min(0), max_error_cm=errors.max(0), n=5)))
            for seed, d in zip(SEEDS, group):
                per_run_curves.append(pd.DataFrame(dict(trajectory=kind, controller=name, seed=seed,
                                                       time_s=d["t"], error_cm=d["error_cm"])))
    pd.concat(curves, ignore_index=True).to_csv(DATA / "section4c_disturbance_summary.csv", index=False)
    pd.concat(per_run_curves, ignore_index=True).to_csv(DATA / "section4c_disturbance_per_run.csv", index=False)
    configure_style()
    fig, report["pulse_figure"] = pulse_figure(data)
    report["figure_4"] = save_figure(fig, "section4c_disturbance_response")
    report["excluded_from_manuscript"] = ["figures/section4c_motor_mismatch.pdf"]
    table_tex(summary)
    outputs = list(TABLE.glob("section4c_*.csv")) + [TABLE / "section4c_robustness.tex"]
    outputs += list(DATA.glob("section4c_*.csv"))
    report["data_and_table_sha256"] = {p.relative_to(LATEX).as_posix(): digest(p) for p in outputs}
    report["report_script_sha256"] = digest(Path(__file__))
    report["loader_script_sha256"] = digest(Path(__file__).with_name("section4c_data.py"))
    finalize_report(sources, report, FIG / "section4c_verification.json")
    print(json.dumps({k: report[k] for k in ("status", "cases_checked", "completed_cases",
        "source_manifest_hashes_verified", "finite_metrics_checked", "max_metric_absolute_difference",
        "pulse_recovery_criteria_checked")}, indent=2))
    print(paper_units(summary)[["scenario", "controller", "completed", "position_rmse_cm_mean",
                               "late_rmse_cm_mean", "post_event_rmse_cm_mean", "recovery_time_s_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
