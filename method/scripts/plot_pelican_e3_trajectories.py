"""Plot actual saved E3 paths against references, without rerunning experiments.

Additive figures only: previously sealed code, results and plots remain intact.
Full sampled paths are plotted per seed; trajectories are never seed-averaged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
KINDS = ['hover', 'circle', 'figure8', 'helix']
TITLES = {'hover': 'Hover', 'circle': 'Circle', 'figure8': 'Figure-eight', 'helix': 'Helix'}
VARIANTS = ['original', 'equilibrium', 'online', 'combined']
LABELS = {'original': 'Original history SINDy-MPC', 'equilibrium': 'Equilibrium correction',
          'online': 'Online compensation', 'combined': 'Equilibrium + online'}
COLORS = {'original': '#595959', 'equilibrium': '#0072B2', 'online': '#E69F00', 'combined': '#009E73'}
STYLES = {'original': '-.', 'equilibrium': '--', 'online': ':', 'combined': '-'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_seal(root, seal):
    for rel, item in seal.items():
        if digest(root/rel) != item['sha256']:
            raise ValueError('Sealed file changed: '+rel)


def load_cases(out):
    plan = json.loads((out/'planned_cases.json').read_text(encoding='utf-8'))
    cases, source_hashes = {}, {}
    for row in plan:
        key = tuple(row[k] for k in ['scenario', 'trajectory', 'seed', 'variant'])
        name = '{scenario}_{trajectory}_{seed}_{variant}'.format(**row)
        folder = out/'cases'/name
        path = folder/'trajectory.npz'
        metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        assert metrics['complete'], name+' did not complete'
        with np.load(path) as f:
            data = {k: f[k].copy() for k in ['t', 'physical', 'reference']}
        n = len(data['t'])
        assert data['physical'].shape == (n, 17) and data['reference'].shape == (n, 13)
        assert np.isfinite(data['physical']).all() and np.isfinite(data['reference']).all()
        assert np.all(np.diff(data['t']) > 0)
        actual = data['physical'][:, :3]
        error = np.linalg.norm(actual-data['reference'][:, :3], axis=1)
        np.testing.assert_allclose(np.sqrt(np.mean(error**2)), metrics['position_rmse_m'], atol=1e-12)
        cases[key] = data
        source_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    return cases, source_hashes


def group(cases, scenario, kind, seed):
    variants = VARIANTS if scenario == 'nominal' else ['equilibrium', 'combined']
    selected = {variant: cases[(scenario, kind, seed, variant)] for variant in variants}
    first = next(iter(selected.values()))
    for value in selected.values():
        np.testing.assert_array_equal(value['t'], first['t'])
        np.testing.assert_array_equal(value['reference'], first['reference'])
        np.testing.assert_array_equal(value['physical'][0], first['physical'][0])
    return selected


def path_limits(ax, positions, three_dimensional=False):
    points = np.concatenate(positions, axis=0)
    lo, hi = points.min(0), points.max(0)
    center = (hi+lo)/2
    span = np.maximum(hi-lo, .02)
    if three_dimensional:
        # Common physical unit scale; expand tiny axes rather than exaggerate Z.
        span = np.maximum(span, .28*span.max())*1.18
        for axis, c, extent in zip(['x', 'y', 'z'], center, span):
            getattr(ax, 'set_'+axis+'lim')(c-extent/2, c+extent/2)
        ax.set_box_aspect(span)
        ax.set_proj_type('ortho')
        ax.view_init(elev=27, azim=-58)
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='')
        # A 2D label is included reliably in tight export bounds for outer panels.
        ax.text2D(.98, .78, 'Z (m)', transform=ax.transAxes,
                  ha='left', va='bottom', fontsize=9, clip_on=False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        ax.tick_params(labelsize=8, pad=1)
    else:
        span = span*1.15
        ax.set_xlim(center[0]-span[0]/2, center[0]+span[0]/2)
        ax.set_ylim(center[1]-span[1]/2, center[1]+span[1]/2)
        ax.set_aspect('equal', adjustable='box')
        ax.set(xlabel='X (m)', ylabel='Y (m)')
        ax.grid(alpha=.2)
        ax.ticklabel_format(style='plain', useOffset=False)


def spatial(ax, selected, three_dimensional=False):
    reference = next(iter(selected.values()))['reference'][:, :3]
    ncoords = 3 if three_dimensional else 2
    # Draw all ablations first, then a thinner dashed reference so overlap stays visible.
    for variant, data in selected.items():
        p = data['physical'][:, :3]
        ax.plot(*p[:, :ncoords].T, color=COLORS[variant], ls=STYLES[variant],
                lw=1.7 if variant == 'combined' else 1.2, alpha=.95, label=LABELS[variant])
    ax.plot(*reference[:, :ncoords].T, color='black', ls=(0, (5, 4)), lw=1.0, label='Reference', zorder=8)
    if np.max(np.ptp(reference, axis=0)) < 1e-12:
        ax.scatter(*reference[0, :ncoords], c='black', marker='*', s=65, zorder=10)
    else:
        ax.scatter(*reference[0, :ncoords], c='black', marker='o', s=18, zorder=10)
        ax.scatter(*reference[-1, :ncoords], c='black', marker='s', s=22, zorder=10)
    path_limits(ax, [reference]+[v['physical'][:, :3] for v in selected.values()], three_dimensional)


def height(ax, selected, scenario):
    data = next(iter(selected.values()))
    for variant, value in selected.items():
        ax.plot(value['t'], value['physical'][:, 2], color=COLORS[variant], ls=STYLES[variant],
                lw=1.5, label=LABELS[variant])
    ax.plot(data['t'], data['reference'][:, 2], color='black', ls=(0, (5, 4)), lw=1.1, label='Reference')
    if scenario == 'bias_step':
        ax.axvline(8., color='#888888', ls=':', lw=1.0)
    ax.set(xlabel='Time (s)', ylabel='Z (m)', xlim=(data['t'][0], data['t'][-1]))
    ax.grid(alpha=.2); ax.ticklabel_format(style='plain', useOffset=False, axis='y')


def legend(fig, variants, top=.91):
    handles = [Line2D([0], [0], color='black', ls=(0, (5, 4)), lw=1.2, label='Reference')]
    handles += [Line2D([0], [0], color=COLORS[v], ls=STYLES[v], lw=1.6, label=LABELS[v]) for v in variants]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, top), ncol=3,
               frameon=False, fontsize=8.5, handlelength=3.)


def save(fig, path, scenario, seed, kinds, selected_variants, registry):
    paths = []
    for extension in ['png', 'pdf']:
        output = path.with_suffix('.'+extension)
        fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=.12)
        paths.append(output.relative_to(ROOT).as_posix())
    registry.append(dict(files=paths, scenario=scenario, seed=seed, trajectories=kinds,
                         variants=selected_variants, actual_source='physical[:, :3]',
                         reference_source='reference[:, :3]', selection='Single specified seed; full saved 0-20 s path; no averaging or smoothing'))
    plt.close(fig)


def main(out):
    out = out.resolve(); figures = out/'figures'; details = figures/'trajectory_details'
    details.mkdir(parents=True, exist_ok=True)
    sealed = json.loads((out/'artifact_hashes.json').read_text(encoding='utf-8'))['files']
    check_seal(ROOT, sealed)
    verification = json.loads((out/'verification.json').read_text(encoding='utf-8'))
    assert verification['status'] == 'PASS'
    cases, source_hashes = load_cases(out)
    seeds = sorted({key[2] for key in cases}); representative = seeds[0]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 11, 'axes.labelsize': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    registry = []
    for mode in ['3d', 'xy', 'height']:
        fig = plt.figure(figsize=(10.3, 8.0))
        for i, kind in enumerate(KINDS):
            ax = fig.add_subplot(2, 2, i+1, projection='3d' if mode == '3d' else None)
            selected = group(cases, 'nominal', kind, representative)
            if mode == 'height': height(ax, selected, 'nominal')
            else: spatial(ax, selected, mode == '3d')
            ax.set_title(f"({'abcd'[i]}) {TITLES[kind]}", pad=9)
        label = {'3d': '3D flight paths', 'xy': 'XY ground-track projections', 'height': 'Altitude tracking'}[mode]
        fig.suptitle(f'{label} | nominal simulation | seed {representative}', y=.99, fontsize=13)
        legend(fig, VARIANTS, .95)
        fig.subplots_adjust(top=.81, bottom=.09, left=.07, right=.89 if mode == '3d' else .95,
                            hspace=.34, wspace=.26)
        note = 'Full 0-20 s paths. Reference: circle = start, square = end, star = hover target.' if mode != 'height' else 'Actual Z coordinate and reference altitude; full 0-20 s simulation.'
        fig.text(.5, .015, note, ha='center', fontsize=8)
        save(fig, figures/f'trajectory_tracking_{mode}', 'nominal', representative, KINDS, VARIANTS, registry)
    # Every seed is available independently, including the disturbed cases.
    for scenario, kinds in [('nominal', KINDS), ('bias_step', ['hover', 'figure8'])]:
        variants = VARIANTS if scenario == 'nominal' else ['equilibrium', 'combined']
        for seed in seeds:
            for kind in kinds:
                selected = group(cases, scenario, kind, seed)
                fig = plt.figure(figsize=(12.6, 4.8))
                ax3 = fig.add_subplot(1, 3, 1, projection='3d')
                axxy = fig.add_subplot(1, 3, 2); axz = fig.add_subplot(1, 3, 3)
                spatial(ax3, selected, True); spatial(axxy, selected); height(axz, selected, scenario)
                for ax, title in zip([ax3, axxy, axz], ['(a) 3D flight path', '(b) XY projection', '(c) Altitude tracking']):
                    ax.set_title(title, pad=9)
                context = 'nominal simulation' if scenario == 'nominal' else 'constant disturbance added at 8 s'
                fig.suptitle(f'{TITLES[kind]} | {context} | seed {seed}', y=.99, fontsize=13)
                legend(fig, variants, .94)
                fig.subplots_adjust(top=.70, bottom=.17, left=.055, right=.98, wspace=.34)
                fig.text(.5, .025, 'Full saved trajectories; no smoothing or seed averaging. Spatial axes use metres.', ha='center', fontsize=8)
                save(fig, details/f'{scenario}_{kind}_{seed}', scenario, seed, [kind], variants, registry)
    # One directly accessible disturbance overview alongside the nominal views.
    fig = plt.figure(figsize=(12.6, 8.2))
    for row, kind in enumerate(['hover', 'figure8']):
        selected = group(cases, 'bias_step', kind, representative)
        ax3 = fig.add_subplot(2, 3, row*3+1, projection='3d')
        axxy = fig.add_subplot(2, 3, row*3+2); axz = fig.add_subplot(2, 3, row*3+3)
        spatial(ax3, selected, True); spatial(axxy, selected); height(axz, selected, 'bias_step')
        for ax, view in zip([ax3, axxy, axz], ['3D flight path', 'XY projection', 'Altitude']):
            ax.set_title(f'{TITLES[kind]}: {view}', pad=8)
    fig.suptitle(f'Trajectory tracking with a disturbance at 8 s | seed {representative}', y=.99, fontsize=13)
    legend(fig, ['equilibrium', 'combined'], .95)
    fig.subplots_adjust(top=.85, bottom=.09, left=.055, right=.98, hspace=.42, wspace=.34)
    save(fig, figures/'trajectory_tracking_disturbed', 'bias_step', representative, ['hover', 'figure8'], ['equilibrium', 'combined'], registry)
    note = f'''# 轨迹跟踪可视化

这里补充的是参考轨迹与实际空间轨迹，不是误差曲线。所有图直接读取已保存的
`cases/*/trajectory.npz`，实际位置来自 `physical[:, :3]`，参考位置来自 `reference[:, :3]`。
保留完整 0–20 s 时序、真实初始偏移与瞬态，不做平滑、重采样或跨种子平均，也未重跑仿真。

## 先看这些图

- [三维轨迹总览](figures/trajectory_tracking_3d.png)：悬停、圆、八字、螺旋，参考及四个控制版本。
- [XY 俯视轨迹总览](figures/trajectory_tracking_xy.png)：直观比较路径形状和横向偏移。
- [高度跟踪总览](figures/trajectory_tracking_height.png)：实际 Z 坐标与参考高度随时间变化。
- [新增持续扰动场景](figures/trajectory_tracking_disturbed.png)：悬停与八字的三维、俯视、高度对照。

上述总览均使用按编号选取的第一个验证种子 {representative}，未按表现选择种子。
`figures/trajectory_details/` 中保留两个种子 {seeds} 的单独图，每张包含三维轨迹、XY 投影和高度跟踪。
每张图均导出 PNG 和矢量 PDF，共 {len(registry)} 张图、{2*len(registry)} 个图片文件。

图例：黑色虚线为参考，灰色为原历史 SINDy-MPC，蓝色为仅平衡点修正，橙色为仅在线补偿，
绿色为平衡点修正与在线补偿结合。空间图中的黑圆点、方块分别为参考起点和终点；悬停参考为黑星。
三维视图各轴使用相同物理尺度，细小的 Z 变化也可通过高度图查看。
多个修正版本与参考线重合是保存数据本身的跟踪效果，没有人为错开曲线。
悬停目标是固定点，悬停图显示各版本从初态向目标靠近的路径。

## 重画

在 `huiyi/method` 下运行：

```powershell
.\\.venv\\Scripts\\python.exe scripts/plot_pelican_e3_trajectories.py --run-dir runs/pelican_e3_corrected_v1
```

此次绘图核对了所有 40 个已完成算例的位置 RMSE、配对参考和初态，并验证原有 {len(sealed)} 个封存文件未改变。
新增图单独记录在 `trajectory_figures_manifest.json`，其中包括源数据 SHA256、图与算例映射、输出文件 SHA256。
'''
    guide = out/'TRAJECTORY_FIGURES.md'; guide.write_text(note, encoding='utf-8')
    check_seal(ROOT, sealed)
    produced = [ROOT/path for item in registry for path in item['files']]+[guide, Path(__file__)]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='PASS',
                    existing_sealed_files_unchanged=len(sealed), source_cases=len(cases),
                    source_position_samples=sum(len(v['t']) for v in cases.values()),
                    source_sha256=source_hashes, plots=registry,
                    files={p.relative_to(ROOT).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in produced})
    (out/'trajectory_figures_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'figures': len(registry), 'image_files': 2*len(registry),
                      'source_cases': len(cases), 'existing_sealed_files_unchanged': len(sealed),
                      'directory': str(figures)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/pelican_e3_corrected_v1')
    main(parser.parse_args().run_dir)
