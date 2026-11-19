"""Disclose the complete search and show only frozen held-out-state validation."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
from run_pelican_e2 import checksum, write_json
from pelican_trajectory_search_common import read_json, rank, load_stages, ORDER
from report_pelican_speed_sweep import COLORS, STYLES, LABELS, HEADERS
from pelican_e3.reference import translation

KINDS = ['circle', 'figure8', 'helix']
TITLES = ['Circle', 'Figure-eight', 'Helix']


def save(fig, path):
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=600)
    plt.close(fig)
    with Image.open(path.with_suffix('.png')) as im:
        im.convert('L').save(path.with_name(path.name + '_grayscale').with_suffix('.png'))


def path_plot(data, seed):
    width, height = 7.14, 4.45
    fig = plt.figure(figsize=(width, height))
    left = [.53 + 1.32 * col for col in range(5)]
    bottom, rh = [2.90, 1.85, .45], [1.10, .60, 1.10]
    for j, label in enumerate(HEADERS):
        fig.text((left[j] + 1.09 / 2) / width, .96, label, ha='center', va='center', fontsize=8)
    for i, kind in enumerate(KINDS):
        positions = [data[(kind, seed, name)]['physical'][:, :3] for name in ORDER]
        references = [data[(kind, seed, name)]['reference'][:, :3] for name in ORDER]
        p = np.concatenate(positions + references)
        span = np.maximum(np.ptp(p, axis=0), .01)
        lo, hi = p.min(0) - .08 * span, p.max(0) + .08 * span
        fig.text(.015, (bottom[i] + rh[i] / 2) / height, TITLES[i], rotation=90,
                 ha='center', va='center', fontsize=8)
        for j, name in enumerate(ORDER):
            is3d = kind == 'helix'
            kw = dict(projection='3d', computed_zorder=False) if is3d else {}
            ax = fig.add_axes([left[j] / width, bottom[i] / height, 1.09 / width, rh[i] / height], **kw)
            dims = 3 if is3d else 2
            ax.plot(*positions[j][:, :dims].T, color=COLORS[j], lw=1.1)
            ax.plot(*references[j][:, :dims].T, color='black', lw=.75, ls=(0, (5, 3)))
            if len(positions[j]) < 2001:
                ax.plot(*positions[j][-1:, :dims].T, marker='x', color=COLORS[j], ms=4)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.tick_params(labelsize=6.5, pad=1, length=2.2, width=.5)
            ax.set_xlabel('x (m)', fontsize=6.5, labelpad=2)
            if is3d:
                ax.set_zlim(lo[2], hi[2])
                ax.set_box_aspect(hi - lo, zoom=.88)
                ax.set_proj_type('ortho')
                ax.view_init(elev=27, azim=-58)
                ax.set_xticks([float(np.ceil(lo[0])), 0])
                ax.set_yticks([float(np.ceil(lo[1])), float(np.floor(hi[1]))])
                ax.set_zticks([1., round(p[:, 2].max(), 1)])
                ax.set_xlabel('x (m)', fontsize=6.5, labelpad=-8)
                ax.set_ylabel('y (m)', fontsize=6.5, labelpad=-8)
                ax.text2D(.98, .80, 'z (m)', transform=ax.transAxes, fontsize=6.5, ha='center', va='bottom')
                ax.tick_params(pad=-2)
                for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis.set_pane_color((1, 1, 1, 0))
                    axis._axinfo['grid'].update(color=(.86, .86, .86, 1), linewidth=.35)
            else:
                ax.set_aspect('equal', adjustable='box')
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                if j == 0:
                    ax.set_ylabel('y (m)', fontsize=6.5, labelpad=2)
                else:
                    ax.tick_params(labelleft=False)
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(alpha=.3, lw=.4)
    fig.legend([Line2D([], [], color='black', lw=.8, ls=(0, (5, 3))), Line2D([], [], color='black', lw=1.1)],
        ['Reference', 'Tracked trajectory'], loc='lower center', bbox_to_anchor=(.5, .015), ncol=2,
        frameon=True, fancybox=False, edgecolor='black', fontsize=7.5)
    return fig


def error_plot(data, seeds, rows):
    fig, axes = plt.subplots(1, 3, figsize=(7.14, 2.9), sharey=True, sharex=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.32, top=.90, wspace=.14)
    ymax = max(d['error_cm'].max() for d in data.values())
    for kind, title, ax in zip(KINDS, TITLES, axes):
        for j, name in enumerate(ORDER):
            group = [data[(kind, s, name)] for s in seeds]
            complete = rows[(rows.trajectory == kind) & (rows.controller == name)].complete.all()
            if complete:
                e = np.stack([d['error_cm'] for d in group])
                ax.fill_between(group[0]['t'], e.min(0), e.max(0), color=COLORS[j], alpha=.10, lw=0)
                ax.plot(group[0]['t'], e.mean(0), color=COLORS[j], ls=STYLES[j], lw=1.)
            else:
                for d in group:
                    ax.plot(d['t'], d['error_cm'], color=COLORS[j], ls=STYLES[j], lw=.8)
        ax.set(xlim=(0, 20), ylim=(0, ymax * 1.05), xlabel='Time (s)', title=title)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.25, lw=.4)
    axes[0].set_ylabel('Position error (cm)')
    fig.legend([Line2D([], [], color=c, ls=s, lw=1.) for c, s in zip(COLORS, STYLES)], LABELS,
        loc='lower center', bbox_to_anchor=(.53, .01), ncol=3, frameon=True, fancybox=False, edgecolor='black', fontsize=7)
    return fig


def report(root):
    root = root.resolve()
    settings = read_json(root / 'config.json')
    for stage in ['grid', 'refine', 'confirm', 'validate']:
        audit = read_json(root / stage / 'verification.json')
        assert audit['status'] == 'PASS'
        assert audit['tracking_sha256'] == checksum(root / stage / 'tracking_per_run.csv')
    screening = load_stages(root, ['grid', 'refine'])
    rank(screening).to_csv(root / 'screening_ranking.csv', index=False)
    development = load_stages(root, ['grid', 'refine', 'confirm'])
    ids = {c['candidate_id'] for c in read_json(root / 'confirm/candidates.json')}
    finalists = development[development.candidate_id.isin(ids)]
    devrank = rank(finalists)
    devrank.to_csv(root / 'finalists_ranking.csv', index=False)
    rows = pd.read_csv(root / 'validate/tracking_per_run.csv')
    chosen = read_json(root / 'selection.json')['candidates']
    validation_rank = rank(rows).set_index('candidate_id')
    summaries, comparisons = [], []
    for kind in KINDS:
        for name in ORDER:
            g = rows[(rows.trajectory == kind) & (rows.controller == name)]
            ok = bool(g.complete.all())
            summaries.append(dict(trajectory=kind, controller=name, n=len(g), completed=int(g.complete.sum()),
                rmse_cm=float(g.position_rmse_m.mean() * 100) if ok else np.nan,
                rmse_std_cm=float(g.position_rmse_m.std(ddof=1) * 100) if ok else np.nan,
                late_rmse_cm=float(g.late_rmse_m.mean() * 100) if ok else np.nan,
                max_error_cm=float(g.max_position_error_m.max() * 100), fallbacks=int(g.fallbacks.sum()),
                input_bound_active_percent=float(g.input_bound_active_fraction.mean() * 100),
                input_slew_active_percent=float(g.input_slew_active_fraction.mean() * 100)))
        h = rows[(rows.trajectory == kind) & (rows.controller == 'sindy_history')].set_index('seed')
        for name in ORDER[:-1]:
            b = rows[(rows.trajectory == kind) & (rows.controller == name)].set_index('seed')
            paired = h[['position_rmse_m', 'late_rmse_m']].join(b[['position_rmse_m', 'late_rmse_m']], lsuffix='_h', rsuffix='_b')
            ok = h.complete.all() and b.complete.all()
            comparisons.append(dict(trajectory=kind, baseline=name, paired_cases=len(paired),
                full_gain_percent=float(100 * (1 - h.position_rmse_m.mean() / b.position_rmse_m.mean())) if ok else np.nan,
                late_gain_percent=float(100 * (1 - h.late_rmse_m.mean() / b.late_rmse_m.mean())) if ok else np.nan,
                full_history_wins=int((paired.position_rmse_m_h < paired.position_rmse_m_b).sum()),
                late_history_wins=int((paired.late_rmse_m_h < paired.late_rmse_m_b).sum())))
    summary, comparison = pd.DataFrame(summaries), pd.DataFrame(comparisons)
    summary.to_csv(root / 'validation_summary.csv', index=False)
    comparison.to_csv(root / 'validation_comparisons.csv', index=False)
    selected_rows = []
    for c in chosen:
        dr = devrank[devrank.candidate_id == c['candidate_id']].iloc[0]
        vr = validation_rank.loc[c['candidate_id']]
        cfg = read_json(root / 'validate/reference_configs' / (c['candidate_id'] + '.json'))
        reference_samples = [translation(t, c['trajectory'], cfg) for t in np.arange(2001) * cfg['dt']]
        selected_rows.append(dict(**c, development_score_percent=dr.score_percent,
            validation_score_percent=vr.score_percent, validation_learned_score_percent=vr.learned_score_percent,
            peak_reference_speed_m_s=float(max(np.linalg.norm(v) for _, v, _ in reference_samples)),
            peak_reference_acceleration_m_s2=float(max(np.linalg.norm(a) for _, _, a in reference_samples))))
    pd.DataFrame(selected_rows).to_csv(root / 'selected_parameters.csv', index=False)
    (root / 'selected_configs').mkdir(exist_ok=True)
    for c in chosen:
        (root / 'selected_configs' / (c['trajectory'] + '.json')).write_bytes(
            (root / 'validate/reference_configs' / (c['candidate_id'] + '.json')).read_bytes())
    seeds = settings['validation_seeds']
    data, frames = {}, []
    for row in rows.itertuples(index=False):
        with np.load(root / 'validate/cases' / row.case_id / 'trajectory.npz') as f:
            d = {k: f[k] for k in ['t', 'physical', 'reference']}
        d['error_cm'] = np.linalg.norm(d['physical'][:, :3] - d['reference'][:, :3], axis=1) * 100
        data[(row.trajectory, row.seed, row.controller)] = d
        frames.append(pd.DataFrame(dict(trajectory=row.trajectory, seed=row.seed, controller=row.controller,
            time_s=d['t'], position_error_cm=d['error_cm'])))
    pd.concat(frames, ignore_index=True).to_csv(root / 'validation_position_error.csv', index=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    (root / 'figures').mkdir(exist_ok=True)
    save(path_plot(data, seeds[0]), root / 'figures/selected_tracking_paths')
    save(error_plot(data, seeds, rows), root / 'figures/selected_position_error')
    all_rows = load_stages(root, ['grid', 'refine', 'confirm', 'validate'])
    failures = all_rows[(~all_rows.complete) | (all_rows.fallbacks > 0)].copy()
    failures.to_csv(root / 'search_failures_and_fallbacks.csv', index=False)
    lines = ['# 三类轨迹参数搜索与留出初态验证', '', '## Material Passport', '',
        f'- 共 {len(screening.candidate_id.unique())} 个参数候选，{len(all_rows)} 个算例记录，其中 {int(all_rows.reused.sum())} 个复用已核验结果。',
        '- 有意按 History 的开发优势选择工况；最佳限于记录的候选集合，不是全局最优或随机代表性工况。',
        '- 先用三个开发初态确认，再冻结每类一个设置；五个留出初态仅用于最终验证，未再调参。',
        '- 模型、控制器、物理对象及补偿保持冻结；本轮仅改变参考参数；全部阶段独立核验通过。',
        '- 仿真并发执行，本报告不作控制运行时间或硬实时比较。', '', '## 冻结的选择', '',
        '| 轨迹 | 半径/横向振幅 (m) | 角频率 (rad/s) | 上升速度 (m/s) | 开发最小改善 (%) | 留出最小改善 (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for c in selected_rows:
        climb = f"{c['climb_m_s']:g}" if c['trajectory'] == 'helix' else '—'
        validation_score = f"{c['validation_score_percent']:.2f}" if pd.notna(c['validation_score_percent']) else '—（未满足合格条件）'
        lines.append(f"| {c['trajectory']} | {c['radius_m']:g} | {c['frequency_rad_s']:g} | {climb} | {c['development_score_percent']:.2f} | {validation_score} |")
    lines += ['', '最小改善同时包含 PD、三个学习 MPC 和全程/后半程两个窗口；负数表示至少一项未领先。',
        '', '## 留出初态全程位置 RMSE（cm，均值 ± 样本标准差）', '',
        '| 方法 | 圆形 | 八字 | 螺旋 |', '| --- | ---: | ---: | ---: |']
    for name, label in zip(ORDER, LABELS):
        vals = []
        for kind in KINDS:
            r = summary[(summary.trajectory == kind) & (summary.controller == name)].iloc[0]
            vals.append(f'{r.rmse_cm:.3f} ± {r.rmse_std_cm:.3f}' if pd.notna(r.rmse_cm)
                        else f'—（{int(r.completed)}/{int(r.n)} 完成）')
        lines.append('| ' + label + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '五个初态对应同一名义物理对象，不是五架独立无人机；本报告作描述性比较，不作显著性声明。',
        '', '## 搜索中的提前终止或回退', '',
        '| 轨迹 | 半径 (m) | 角频率 (rad/s) | 爬升 (m/s) | 方法 | 初态 | 完成 | 回退 | 终止原因 |',
        '| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |']
    for r in failures.itertuples(index=False):
        lines.append(f'| {r.trajectory} | {r.radius_m:g} | {r.frequency_rad_s:g} | {r.climb_m_s:g} | {r.controller} | {r.seed} | {r.complete} | {r.fallbacks} | {r.termination_reason} |')
    lines += ['', '这些工况完整保留，未将失败前缀误差与完整 20 s 误差混合排名。',
        '', '## 数据与复现', '',
        '- configs/pelican_trajectory_search.json 与 docs/PELICAN_TRAJECTORY_SEARCH_PROTOCOL.md：预定范围和选择规则。',
        '- screening_ranking.csv：全部单初态候选；finalists_ranking.csv：两个候选/轨迹的三开发初态复核。',
        '- selection.json：留出运行前冻结的选择；selected_configs/：三类最终配置。',
        '- validation_summary.csv / validation_comparisons.csv：留出均值、离散性和每个基线的配对胜出次数。',
        '- grid/、refine/、confirm/、validate/：全部计划、初态、原始轨迹、约束和求解记录，包括失败。',
        f'- figures/selected_tracking_paths.pdf：固定首个留出初态 {seeds[0]}；圆、八字、螺旋三行，History 最后一列。',
        '- figures/selected_position_error.pdf：五初态的均值与最小–最大范围；范围不是置信区间。', '']
    (root / 'SEARCH_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(root / 'artifact_hashes.json', dict(report_script_sha256=checksum(Path(__file__)), files={
        p.relative_to(ROOT).as_posix(): {'sha256': checksum(p), 'bytes': p.stat().st_size}
        for p in root.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}))
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    report(parser.parse_args().run_dir)
