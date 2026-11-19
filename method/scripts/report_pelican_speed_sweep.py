"""Report every prespecified speed, with first-seed paths and independent errors."""
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
from PIL import Image
from run_pelican_e2 import checksum, write_json
from run_pelican_speed_sweep import read_json

ORDER = ['pd', 'sindy_current', 'linear_history', 'motor_sindy', 'sindy_history']
LABELS = ['PD', 'Current SINDy-MPC', 'Linear history MPC', 'Motor + SINDy-MPC', 'History SINDy-MPC']
HEADERS = ['PD', 'Current\nSINDy-MPC', 'Linear history\nMPC', 'Motor +\nSINDy-MPC', 'History\nSINDy-MPC']
COLORS = ['#888888', '#D55E00', '#0072B2', '#CC79A7', '#009E73']
STYLES = [':', '--', '-.', (0, (3, 1, 1, 1)), '-']


def save(fig, path):
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=600)
    with Image.open(path.with_suffix('.png')) as im:
        im.convert('L').save(path.with_name(path.name + '_grayscale').with_suffix('.png'))
    plt.close(fig)


def report(out):
    out = out.resolve()
    audit = read_json(out / 'verification.json')
    assert audit['status'] == 'PASS'
    assert audit['tracking_sha256'] == checksum(out / 'tracking_per_run.csv')
    frozen, settings = read_json(out / 'FROZEN.json'), read_json(out / 'config.json')
    stage = settings[frozen['stage']]
    rows = pd.read_csv(out / 'tracking_per_run.csv')
    frequencies, seed = settings['frequency_rad_s'], stage['seeds'][0]
    summary = []
    for w in frequencies:
        for name in ORDER:
            group = rows[(rows.frequency_rad_s == w) & (rows.controller == name)]
            complete = bool(group.complete.all())
            summary.append(dict(frequency_rad_s=w, controller=name, cases=len(group),
                completed=int(group.complete.sum()), tracking_successes=int(group.tracking_success.sum()),
                position_rmse_cm=float(group.position_rmse_m.mean() * 100) if complete else np.nan,
                position_rmse_std_cm=float(group.position_rmse_m.std(ddof=1) * 100) if complete and len(group) > 1 else np.nan,
                late_rmse_cm=float(group.late_rmse_m.mean() * 100) if complete else np.nan,
                max_error_cm=float(group.max_position_error_m.max() * 100),
                fallbacks=int(group.fallbacks.sum()),
                input_bound_active_percent=float(group.input_bound_active_fraction.mean() * 100),
                input_slew_active_percent=float(group.input_slew_active_fraction.mean() * 100)))
    stats = pd.DataFrame(summary)
    stats.to_csv(out / 'tracking_summary.csv', index=False)
    data = {}
    for row in rows.itertuples(index=False):
        with np.load(out / 'cases' / row.case_id / 'trajectory.npz') as f:
            data[(row.frequency_rad_s, row.seed, row.controller)] = {k: f[k] for k in ['t', 'physical', 'reference']}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    folder = out / 'figures'
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(frequencies), 5, figsize=(7.14, 3.65), squeeze=False)
    fig.subplots_adjust(left=.10, right=.99, bottom=.17, top=.84, wspace=.24, hspace=.58)
    all_xy = np.concatenate([d[k][:, :2] for (w, s, name), d in data.items() if s == seed
                             for k in ['physical', 'reference']])
    low, high = all_xy.min(0), all_xy.max(0)
    pad = np.maximum((high - low) * .06, .02)
    for i, w in enumerate(frequencies):
        for j, name in enumerate(ORDER):
            ax, d = axes[i, j], data[(w, seed, name)]
            p, ref = d['physical'], d['reference']
            ax.plot(p[:, 0], p[:, 1], color=COLORS[j], lw=1.0)
            ax.plot(ref[:, 0], ref[:, 1], color='black', lw=.75, ls=(0, (5, 3)))
            if not bool(rows[(rows.frequency_rad_s == w) & (rows.seed == seed) & (rows.controller == name)].iloc[0].complete):
                ax.plot(p[-1, 0], p[-1, 1], marker='x', color=COLORS[j], ms=4)
            ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
            ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([-.7, 0, .7])
            ax.set_yticks([-.3, 0, .3])
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.25, lw=.4)
            ax.set_xlabel('x (m)', labelpad=1)
            if j == 0:
                ax.set_ylabel('y (m)', labelpad=1)
            else:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.set_title(HEADERS[j], fontsize=8, pad=9)
        position = axes[i, 0].get_position()
        fig.text(.016, (position.y0 + position.y1) / 2, f'{w:g} rad/s', rotation=90,
                 ha='center', va='center', fontsize=8)
    legend = fig.legend([Line2D([], [], color='black', ls=(0, (5, 3)), lw=.8),
                         Line2D([], [], color='black', lw=1.)],
        ['Reference', 'Tracked trajectory'], loc='lower center', bbox_to_anchor=(.53, .013),
        ncol=2, frameon=True, fancybox=False, edgecolor='black', fontsize=8)
    legend.get_frame().set_linewidth(.6)
    save(fig, folder / 'trajectory_speed_grid')
    fig, axes = plt.subplots(1, len(frequencies), figsize=(7.14, 2.65), sharey=True, squeeze=False)
    fig.subplots_adjust(left=.085, right=.99, bottom=.34, top=.88, wspace=.13)
    error_rows = []
    ymax = max(np.linalg.norm(d['physical'][:, :3] - d['reference'][:, :3], axis=1).max() * 100
               for d in data.values())
    for i, w in enumerate(frequencies):
        ax = axes[0, i]
        for j, name in enumerate(ORDER):
            subset = [data[(w, s, name)] for s in stage['seeds']]
            lengths = [np.linalg.norm(d['physical'][:, :3] - d['reference'][:, :3], axis=1) * 100 for d in subset]
            complete = rows[(rows.frequency_rad_s == w) & (rows.controller == name)].complete.all()
            if complete:
                matrix = np.asarray(lengths)
                t = subset[0]['t']
                ax.plot(t, matrix.mean(0), color=COLORS[j], ls=STYLES[j], lw=1.)
                if len(matrix) > 1:
                    ax.fill_between(t, matrix.min(0), matrix.max(0), color=COLORS[j], alpha=.12)
            else:
                for d, e in zip(subset, lengths):
                    ax.plot(d['t'], e, color=COLORS[j], ls=STYLES[j], lw=.9)
            for s, d, e in zip(stage['seeds'], subset, lengths):
                error_rows.extend(dict(frequency_rad_s=w, controller=name, seed=s, t=t, error_cm=v)
                                  for t, v in zip(d['t'], e))
        ax.set_title(f'{w:g} rad/s', fontsize=8)
        ax.set_xlim(0, stage['duration_s'])
        ax.set_ylim(0, ymax * 1.05)
        ax.set_xlabel('Time (s)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.25, lw=.4)
    axes[0, 0].set_ylabel('Position error (cm)')
    fig.legend([Line2D([], [], color=c, ls=s, lw=1.) for c, s in zip(COLORS, STYLES)], LABELS,
        loc='lower center', bbox_to_anchor=(.54, .005), ncol=3, fontsize=7,
        frameon=True, fancybox=False, edgecolor='black')
    save(fig, folder / 'position_error_speed')
    pd.DataFrame(error_rows).to_csv(out / 'position_error_per_run.csv', index=False)
    lines = ['# 八字轨迹速度敏感性结果', '', '## Material Passport', '',
        f'- 阶段：{frozen["stage"]}；共 {len(rows)} 例，{int(rows.complete.sum())} 例完整完成。',
        f'- 每个速度/方法 {len(stage["seeds"])} 个配对初态；轨迹图固定使用 {seed}。',
        '- Pelican 辨识模型上的闭环仿真；只改变参考频率；独立核验 PASS。',
        '- 开发单初态结果仅用于可行性和方向检查，不能当作正式多初态结果。' if frozen['stage'] == 'development'
            else '- 均值按配对初态等权计算；不进行显著性推断。', '',
        '## 全程位置 RMSE（cm）', '',
        '| 控制器 | ' + ' | '.join(f'{w:g} rad/s' for w in frequencies) + ' |',
        '| --- | ' + ' | '.join('---:' for w in frequencies) + ' |']
    for name, label in zip(ORDER, LABELS):
        values = []
        for w in frequencies:
            r = stats[(stats.frequency_rad_s == w) & (stats.controller == name)].iloc[0]
            values.append(f'{r.position_rmse_cm:.3f}' if pd.notna(r.position_rmse_cm)
                          else f'—（{int(r.completed)}/{int(r.cases)} 完成）')
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 约束与求解记录', '',
        '| 频率 (rad/s) | 方法 | 完成数 | 幅值约束活跃 (%) | 变化率约束活跃 (%) | 回退次数 |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for r in stats.itertuples(index=False):
        lines.append(f'| {r.frequency_rad_s:g} | {r.controller} | {r.completed}/{r.cases} | '
                     f'{r.input_bound_active_percent:.2f} | {r.input_slew_active_percent:.2f} | {r.fallbacks} |')
    lines += ['', '## 文件', '',
        '- figures/trajectory_speed_grid.pdf：三行速度、五列控制器的原始路径。',
        '- figures/position_error_speed.pdf：同步位置误差；多初态时为均值及最小–最大范围。',
        '- tracking_summary.csv：全程和后段 RMSE、约束活跃比例与完成数。',
        '- tracking_per_run.csv 与 cases/：全部逐例记录，包括提前终止的前缀。',
        '- verification.json：来源、配对、独立参考、物理积分、残差/偏差及指标核验。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(out / 'artifact_hashes.json', {'files': {
        p.relative_to(ROOT).as_posix(): {'sha256': checksum(p), 'bytes': p.stat().st_size}
        for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}})
    print(stats[['frequency_rad_s', 'controller', 'completed', 'position_rmse_cm', 'late_rmse_cm', 'fallbacks']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    report(parser.parse_args().run_dir)
