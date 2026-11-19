"""Descriptive formal comparison, actual trajectories, errors and compute costs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_pelican_e2 import checksum, write_json
from plot_pelican_e3_trajectories import path_limits
from run_pelican_e3_nominal_comparison import check_preserved

ORDER = ['pd', 'sindy_current', 'linear_history', 'motor_sindy', 'sindy_history']
TITLES = {'pd': 'PD', 'sindy_current': 'Current SINDy', 'linear_history': 'Linear history',
          'motor_sindy': 'Motor + SINDy', 'sindy_history': 'History SINDy'}
CN = {'pd': 'PD（名义对象知识）', 'sindy_current': '无历史 SINDy-MPC',
      'linear_history': '线性历史 MPC', 'motor_sindy': '电机 + SINDy-MPC', 'sindy_history': '历史 SINDy-MPC'}
COLORS = {'pd': '#777777', 'sindy_current': '#D55E00', 'linear_history': '#0072B2',
          'motor_sindy': '#CC79A7', 'sindy_history': '#009E73'}
STYLES = {'pd': ':', 'sindy_current': '-.', 'linear_history': '--', 'motor_sindy': (0, (3, 1, 1, 1)), 'sindy_history': '-'}
KINDS = ['circle', 'figure8', 'helix']
KIND_TITLES = {'circle': 'Circle', 'figure8': 'Figure-eight', 'helix': 'Helix'}


def load_data(out, frame):
    data, controls = {}, {}
    for row in frame.itertuples(index=False):
        key = (row.trajectory, row.seed, row.controller)
        folder = out/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}'
        with np.load(folder/'trajectory.npz') as f: data[key] = {k: f[k] for k in ['t', 'physical', 'reference']}
        controls[key] = pd.read_csv(folder/'control.csv')
        errors = np.linalg.norm(data[key]['physical'][:, :3]-data[key]['reference'][:, :3], axis=1)
        np.testing.assert_allclose(np.sqrt(np.mean(errors**2)), row.prefix_position_rmse_m, atol=1e-12)
    return data, controls


def aggregate(frame, controls, columns):
    rows = []
    for key, g in frame.groupby(columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(columns, key)); row.update(cases=len(g), completed=int(g.complete.sum()),
                                                  tracking_successes=int(g.tracking_success.sum()))
        for field in ['position_rmse_m', 'late_rmse_m', 'late_bias_norm_m', 'command_variation_squared']:
            # No silent successful-case-only leaderboard.
            row[field] = float(g[field].mean()) if g.complete.all() else None
        row['position_rmse_std_m'] = float(g.position_rmse_m.std(ddof=1)) if g.complete.all() else None
        row['max_position_error_m'] = float(g.max_position_error_m.max())
        for field in ['fallbacks', 'solver_failures', 'input_bound_violations', 'input_slew_violations', 'observer_rejected', 'observer_clipped']:
            row[field] = int(g[field].sum())
        c = pd.concat([controls[(r.trajectory, r.seed, r.controller)] for r in g.itertuples(index=False)])
        for field in ['command_compute_s', 'full_control_period_s', 'seconds']:
            row[field+'_median'] = float(c[field].median())
            row[field+'_p95'] = float(c[field].quantile(.95))
            row[field+'_p99'] = float(c[field].quantile(.99))
            row[field+'_max'] = float(c[field].max())
        row['full_control_over_period_fraction'] = float((c.full_control_period_s > .05).mean())
        row['control_calls'] = len(c)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(frame):
    history = frame[frame.controller == 'sindy_history'].set_index(['trajectory', 'seed'])
    rows = []
    for baseline in ORDER[:-1]:
        base = frame[frame.controller == baseline].set_index(['trajectory', 'seed'])
        assert set(base.index) == set(history.index)
        for key in sorted(history.index):
            a, b = base.loc[key], history.loc[key]
            complete = bool(a.complete and b.complete)
            rows.append(dict(baseline=baseline, trajectory=key[0], seed=key[1], pair_complete=complete,
                baseline_rmse_m=a.position_rmse_m if a.complete else None,
                history_rmse_m=b.position_rmse_m if b.complete else None,
                history_minus_baseline_m=b.position_rmse_m-a.position_rmse_m if complete else None,
                reduction_percent=100*(1-b.position_rmse_m/a.position_rmse_m) if complete else None,
                history_control_variation=b.command_variation_squared,
                baseline_control_variation=a.command_variation_squared))
    return pd.DataFrame(rows)


def add_legend(fig, top=.94):
    handles = [Line2D([0], [0], color='black', lw=1.1, ls=(0, (5, 4)), label='Reference')]
    handles += [Line2D([0], [0], color=COLORS[n], ls=STYLES[n], lw=1.5, label=TITLES[n]) for n in ORDER]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, top), ncol=3, frameon=False, fontsize=9)


def spatial(ax, data, kind, seed, is3d):
    ref = data[(kind, seed, ORDER[0])]['reference'][:, :3]; points = [ref]
    for name in ORDER:
        p = data[(kind, seed, name)]['physical'][:, :3]; points.append(p)
        ax.plot(*p[:, :3 if is3d else 2].T, color=COLORS[name], ls=STYLES[name], lw=1.6 if name == 'sindy_history' else 1.2)
    ax.plot(*ref[:, :3 if is3d else 2].T, color='black', ls=(0, (5, 4)), lw=1.0, zorder=8)
    for index, marker in [(0, 'o'), (-1, 's')]:
        ax.scatter(*ref[index, :3 if is3d else 2], color='black', marker=marker, s=22, zorder=10)
    path_limits(ax, points, is3d)


def altitude(ax, data, kind, seed):
    for name in ORDER:
        d = data[(kind, seed, name)]
        ax.plot(d['t'], d['physical'][:, 2], color=COLORS[name], ls=STYLES[name], lw=1.3)
    ax.plot(d['t'], d['reference'][:, 2], color='black', ls=(0, (5, 4)), lw=1.0)
    ax.set(xlabel='Time (s)', ylabel='Z (m)', xlim=(0, 20))
    ax.ticklabel_format(style='plain', useOffset=False, axis='y'); ax.grid(alpha=.2)


def savefig(fig, path, registry):
    for extension in ['png', 'pdf']:
        p = path.with_suffix('.'+extension); fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=.12)
        registry.append(p)
    plt.close(fig)


def make_figures(out, frame, summary, data):
    directory = out/'figures'; directory.mkdir(exist_ok=True)
    details = directory/'trajectories'; details.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    seeds = sorted(frame.seed.unique()); seed = int(seeds[0]); images = []
    for mode in ['3d', 'xy']:
        fig = plt.figure(figsize=(12.8, 5.2))
        for i, kind in enumerate(KINDS):
            ax = fig.add_subplot(1, 3, i+1, projection='3d' if mode == '3d' else None)
            spatial(ax, data, kind, seed, mode == '3d')
            ax.set_title(KIND_TITLES[kind], pad=10)
        fig.suptitle(f'Nominal trajectory tracking | seed {seed} | all MPCs equally compensated', y=.99, fontsize=12)
        add_legend(fig); fig.subplots_adjust(left=.05, right=.95, top=.70, bottom=.17, wspace=.30)
        fig.text(.5, .02, 'Full 0-20 s saved paths; circle = reference start, square = reference end.', ha='center', fontsize=8)
        savefig(fig, directory/f'trajectory_tracking_{mode}', images)
    for kind in KINDS:
        for seed in seeds:
            fig = plt.figure(figsize=(12.8, 5.2))
            a = fig.add_subplot(1, 3, 1, projection='3d'); b = fig.add_subplot(1, 3, 2); c = fig.add_subplot(1, 3, 3)
            spatial(a, data, kind, seed, True); spatial(b, data, kind, seed, False); altitude(c, data, kind, seed)
            for ax, title in zip([a, b, c], ['3D flight path', 'XY projection', 'Altitude']): ax.set_title(title, pad=10)
            fig.suptitle(f'{KIND_TITLES[kind]} | seed {seed} | all MPCs equally compensated', y=.99, fontsize=12)
            add_legend(fig); fig.subplots_adjust(left=.05, right=.98, top=.70, bottom=.17, wspace=.34)
            fig.text(.5, .02, 'Actual physical positions and reference; no smoothing or cross-seed averaging.', ha='center', fontsize=8)
            savefig(fig, details/f'{kind}_{seed}', images)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for ax, kind in zip(axes, KINDS):
        for name in ORDER:
            group = [data[(kind, seed, name)] for seed in seeds]
            errors = [100*np.linalg.norm(d['physical'][:, :3]-d['reference'][:, :3], axis=1) for d in group]
            if len({len(e) for e in errors}) == 1:
                e = np.asarray(errors)
                ax.plot(group[0]['t'], e.mean(0), color=COLORS[name], ls=STYLES[name], label=TITLES[name], lw=1.4)
                ax.fill_between(group[0]['t'], e.min(0), e.max(0), color=COLORS[name], alpha=.08, lw=0)
            else:
                for j, (d, e) in enumerate(zip(group, errors)):
                    ax.plot(d['t'], e, color=COLORS[name], ls=STYLES[name], alpha=.5, label=TITLES[name] if j == 0 else None)
        ax.set(title=KIND_TITLES[kind], xlabel='Time (s)', ylabel='Position error (cm)'); ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle('Nominal tracking error | all MPCs equally compensated', y=.99, fontsize=12)
    fig.subplots_adjust(left=.065, right=.98, top=.70, bottom=.17, wspace=.32)
    fig.text(.5, .02, 'Five paired initial states: lines = mean; bands = min-max range, not confidence intervals.', ha='center', fontsize=8)
    savefig(fig, directory/'tracking_error', images)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, field, scale, title, ylabel in [
        (axes[0], 'position_rmse_m', 100, 'Position tracking', 'Per-case RMSE (cm)'),
        (axes[1], 'command_variation_squared', 1, 'Control variation', 'Sum of squared command changes'),
        (axes[2], 'full_control_period_p95_s', 1000, 'Algorithm time', 'Per-case p95 full period (ms)')]:
        for j, name in enumerate(ORDER):
            vals = frame.loc[frame.controller == name, field].dropna().to_numpy()*scale
            jitter = np.linspace(-.15, .15, len(vals))
            ax.scatter(j+jitter, vals, s=15, color=COLORS[name], alpha=.7, zorder=3)
            if len(vals): ax.plot([j-.25, j+.25], [vals.mean()]*2, color='black', lw=1.8)
        ax.set_xticks(range(len(ORDER)), ['PD', 'Current', 'Linear', 'Motor', 'History'], rotation=25)
        ax.set(title=title, ylabel=ylabel); ax.grid(axis='y', alpha=.2)
    axes[2].set_yscale('log'); axes[2].axhline(50, color='#777777', ls='--', lw=1)
    fig.subplots_adjust(left=.06, right=.98, bottom=.22, top=.85, wspace=.35)
    fig.text(.5, .02, 'Dots = trajectory/seed cases; black ticks = means. Correlated scenarios are not independent significance tests.', ha='center', fontsize=8)
    savefig(fig, directory/'performance_summary', images)
    return images


def table(summary, detailed=False):
    headers = (['轨迹'] if detailed else [])+['方法', '完成', '位置 RMSE (m)', '后段 RMSE (m)', '指令变化平方和', '完整周期 p95 (ms)']
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for r in summary.itertuples(index=False):
        cells = ([r.trajectory] if detailed else [])+[CN[r.controller], f'{r.completed}/{r.cases}',
                f'{r.position_rmse_m:.6f}', f'{r.late_rmse_m:.6f}', f'{r.command_variation_squared:.3f}',
                f'{1000*r.full_control_period_s_p95:.3f}']
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def timing_table(summary):
    lines = ['| 方法 | 中位数 (ms) | p95 (ms) | p99 (ms) | 最大值 (ms) | 超过 50 ms |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in summary.itertuples(index=False):
        lines.append(f'| {CN[r.controller]} | {1000*r.full_control_period_s_median:.3f} | '
                     f'{1000*r.full_control_period_s_p95:.3f} | {1000*r.full_control_period_s_p99:.3f} | '
                     f'{1000*r.full_control_period_s_max:.3f} | {100*r.full_control_over_period_fraction:.3f}% |')
    return '\n'.join(lines)


def main(out):
    out = out.resolve(); verification = json.loads((out/'verification.json').read_text())
    assert verification['status'] == 'PASS'
    assert verification['tracking_sha256'] == checksum(out/'tracking_per_run.csv')
    frame = pd.read_csv(out/'tracking_per_run.csv'); assert len(frame) == 75
    data, controls = load_data(out, frame)
    summary = aggregate(frame, controls, ['controller']).set_index('controller').loc[ORDER].reset_index()
    detailed = aggregate(frame, controls, ['trajectory', 'controller'])
    detailed['controller'] = pd.Categorical(detailed.controller, ORDER, ordered=True)
    detailed['trajectory'] = pd.Categorical(detailed.trajectory, KINDS, ordered=True)
    detailed = detailed.sort_values(['trajectory', 'controller'])
    pairs = paired_comparisons(frame)
    summary.to_csv(out/'tracking_summary.csv', index=False); detailed.to_csv(out/'tracking_by_trajectory.csv', index=False)
    pairs.to_csv(out/'paired_comparisons.csv', index=False)
    seed_table = frame.pivot(index=['trajectory', 'seed'], columns='controller', values='position_rmse_m')[ORDER]
    seed_table.to_csv(out/'rmse_by_seed.csv')
    images = make_figures(out, frame, summary, data)
    values = summary.set_index('controller'); comparisons = []
    for name in ORDER[:-1]:
        p = pairs[pairs.baseline == name]; valid = p[p.pair_complete]
        ratio = 100*(1-values.loc['sindy_history', 'position_rmse_m']/values.loc[name, 'position_rmse_m'])
        comparisons.append(f"- 对 {CN[name]}：平均 RMSE 相对变化对应改善 {ratio:.2f}%；完整配对 {len(valid)}/15，历史模型 RMSE 更低 {int((valid.history_minus_baseline_m < 0).sum())} 例。负的改善率表示历史模型更差。")
    equal_compensation = '四种 MPC 均包含同等悬停校准和在线偏差补偿；表中方法名省略该共同后缀。'
    rank = summary.sort_values('position_rmse_m').controller.tolist()
    conclusion = ('本轮总体位置 RMSE 从低到高为：'+'、'.join(CN[n] for n in rank)+'。') if frame.complete.all() else '存在提前终止，本轮不按成功案例均值宣布总体优胜方法。'
    cost_ratio = values.loc['sindy_history', 'command_variation_squared']/values.loc['linear_history', 'command_variation_squared']
    counts = check_preserved()
    text = f'''# E3 同等补偿后的正式名义对比结果

## Material Passport

- 状态：正式运行完成 {len(frame)} 例，其中完整完成 {int(frame.complete.sum())} 例；独立核验 PASS。
- 条件：圆、八字、螺旋 × 五个配对初态（13301–13305）× 五种控制器，每例 20 s。
- 开发：独立种子 12301 的十个 12 s 算例通过；随后锁定配置，未根据正式排名调参。
- 数据：真实 Pelican 日志辨识得到的既有 E2 模型；闭环结果来自冻结的名义仿真器，未新增实机飞行。
- 协议：`docs/PELICAN_E3_NOMINAL_COMPARISON_PROTOCOL.md`；所有先前封存文件保持一致。

## 总体结果

{equal_compensation}
PD 沿用原 E3 的控制分配和名义对象参数，作为经典参考单列。
位置指标按完整算例等权平均，后段为 10–20 s。p95 是汇合该方法所有控制周期之后计算的分位数。

{table(summary)}

{conclusion}

## 预先指定的配对比较

以下改善率以总体平均 RMSE 的比值计算；逐配对的绝对差和百分比另存 `paired_comparisons.csv`。
三个轨迹共享初态种子，15 个相关场景不是 15 个独立物理对象；不作统计显著性或泛化保证。

{chr(10).join(comparisons)}

历史模型相对线性历史基线的指令变化平方和比值为 **{cost_ratio:.3f}**。
不能只依据位置误差忽略动作变化与计算开销，也不将编码指令变化指标解释为真实能耗。

## 分轨迹结果

{table(detailed, True)}

## 稳定运行与计算开销

- 跟踪成功 {int(frame.tracking_success.sum())}/{len(frame)}，按原 E3 的冻结门槛判定。
- 求解失败 {int(frame.solver_failures.sum())} 次；PD 回退 {int(frame.fallbacks.sum())} 次；输入上下界违规 {int(frame.input_bound_violations.sum())} 次；变化率违规 {int(frame.input_slew_violations.sum())} 次。
- 观测器拒绝 {int(frame.observer_rejected.sum())} 次，估计上限投影 {int(frame.observer_clipped.sum())} 次。
- 完整周期算法开销包括当前历史打包、参考取样、求解、回退、限幅，以及上个控制时刻以来五次观测/偏差更新和中间四次打包；未重复计入当前打包。
- 计时不包含物理仿真器积分、日志、预热、图构建和预生成参考。超过 50 ms 的比例保存在 `tracking_summary.csv`；仿真没有因超时插入执行延迟，不能视为硬实时证明。

以下均为完整控制周期算法开销，每种方法汇合 6,000 个控制周期；分位数未先按算例平均。

{timing_table(summary)}

## 独立核验

- 四项针对新运行器的测试通过：与原 PD/修复版历史控制器的轨迹一致性、因果电机估计及残差回放、完整周期计时无重复计数。
- 检查 {verification['observed_transitions']:,} 个位姿差分观测；以独立 NumPy 特征计算重算四种 MPC 的 {verification['residual_rows']:,} 个残差，最大误差 {verification['max_residual_replay_error']:.3g}。
- 核对 {verification['causal_bias_checks']:,} 次 MPC 偏差索引、{verification['timing_checks']:,} 次周期计时、{verification['pd_command_checks']:,} 次 PD/回退命令。
- 自适应积分抽查 {verification['independent_plant_steps']} 个物理转移，最大状态分量差 {verification['max_adaptive_state_error']:.3g}；各分量按指定绝对/相对容差核验。
- 四个模型校准后的悬停输出残差均在浮点精度内。只修改副本的 `y_mu`，未修改原 E2 模型参数。
- 原文件清单核验：{json.dumps(counts, ensure_ascii=False)}。各清单可能重叠，不将条目数相加解释为独立文件总数。

## 图片与数据入口

- `figures/trajectory_tracking_3d.png/.pdf`：实际三维轨迹与参考，总览固定选第一个正式初态 13301。
- `figures/trajectory_tracking_xy.png/.pdf`：XY 俯视路径，便于比较细小偏移。
- `figures/trajectories/`：每条轨迹、每个初态单独一图，包含三维、俯视与高度。使用完整原始路径，不平滑、不跨种子平均。
- `figures/tracking_error.png/.pdf`：五个初态的误差均值与最小–最大范围，范围不是置信区间。
- `figures/performance_summary.png/.pdf`：误差、指令变化与完整周期时间，各点为一个轨迹/初态算例。
- `tracking_per_run.csv`、`tracking_summary.csv`、`tracking_by_trajectory.csv`、`rmse_by_seed.csv`：逐例、整体、分轨迹、分初态结果。
- `cases/*/trajectory.npz` 和 `control.csv`：位置/姿态、观测、参考、实际指令、残差、在线估计、因果电机估计及完整计时。
- `FROZEN.json`、`verification.json`、`artifact_hashes.json`：预先冻结、独立核验和结果完整性记录。

## 当前证据边界

本轮检验名义场景中模型结构的闭环差别。所有学习模型都采用相同补偿，因而不把补偿条件不同造成的差异归因于历史或非线性结构。
校准使用已知名义仿真器悬停点；PD 还使用其分配与拖曳等先验。结果不能直接推广为未知实机性能。
尚未在本轮加入传感器噪声、参数变化或外部扰动，也未证明一般性零稳态误差和稳定性。
后续 E4 必须另行冻结扰动/不匹配条件；无论本轮排名怎样，均保留全部方法和算例。
'''
    report = ROOT/'docs/PELICAN_E3_NOMINAL_COMPARISON_RESULTS.md'
    report.write_text(text, encoding='utf-8'); (out/'REPORT.md').write_text(text, encoding='utf-8')
    # Read the serialized summary back and recompute every reported mean.
    checked = pd.read_csv(out/'tracking_summary.csv')
    for r in checked.itertuples(index=False):
        g = frame[frame.controller == r.controller]
        if g.complete.all():
            np.testing.assert_allclose(r.position_rmse_m, g.position_rmse_m.mean(), atol=1e-14)
            np.testing.assert_allclose(r.command_variation_squared, g.command_variation_squared.mean(), atol=1e-10)
    files = [p for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json']
    files += [Path(__file__), ROOT/'scripts/run_pelican_e3_nominal_comparison.py',
              ROOT/'scripts/verify_pelican_e3_nominal_comparison.py', ROOT/'scripts/plot_pelican_e3_trajectories.py',
              ROOT/'tests/test_pelican_e3_nominal_comparison.py', ROOT/'configs/pelican_e3_nominal_comparison.json',
              ROOT/'docs/PELICAN_E3_NOMINAL_COMPARISON_PROTOCOL.md', report]
    write_json(out/'artifact_hashes.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        files={p.relative_to(ROOT).as_posix(): dict(sha256=checksum(p), bytes=p.stat().st_size) for p in sorted(set(files))}))
    print(summary[['controller', 'cases', 'completed', 'position_rmse_m', 'late_rmse_m', 'command_variation_squared', 'full_control_period_s_p95']].to_string(index=False))
    print('Report:', report, 'Figures:', len(images)//2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir)
