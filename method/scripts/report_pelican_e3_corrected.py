"""Descriptive paired results and standalone figures for the E3 repair."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_pelican_e2 import checksum, write_json

ORDER = ['original', 'equilibrium', 'online', 'combined']
LABELS = {'original': 'Original', 'equilibrium': 'Equilibrium', 'online': 'Online', 'combined': 'Combined'}
CN = {'original': '原历史 SINDy-MPC', 'equilibrium': '仅平衡点修正', 'online': '仅在线补偿', 'combined': '平衡点修正 + 在线补偿'}
COLORS = {'original': '#595959', 'equilibrium': '#0072B2', 'online': '#E69F00', 'combined': '#009E73'}
STYLES = {'original': ':', 'equilibrium': '--', 'online': '-.', 'combined': '-'}


def load_case(out, scenario, trajectory, seed, variant):
    with np.load(out/'cases'/f'{scenario}_{trajectory}_{seed}_{variant}'/'trajectory.npz') as f:
        return {key: f[key] for key in f.files}


def summary_table(frame, grouping):
    rows = []
    for key, group in frame.groupby(grouping, sort=False):
        row = dict(zip(grouping, key if isinstance(key, tuple) else (key,)))
        row.update(cases=len(group), completed=int(group.complete.sum()))
        for field in ['position_rmse_m', 'late_rmse_m', 'late_bias_norm_m', 'command_variation_squared']:
            row[field] = float(group[field].mean())
        row['max_position_error_m'] = float(group.max_position_error_m.max())
        for field in ['fallbacks', 'solver_failures', 'input_bound_violations', 'input_slew_violations', 'observer_rejected', 'observer_clipped']:
            row[field] = int(group[field].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def error_plot(ax, out, frame, scenario, trajectory, variants):
    group = frame[(frame.scenario == scenario) & (frame.trajectory == trajectory)]
    for variant in variants:
        errors = []
        for seed in sorted(group.seed.unique()):
            data = load_case(out, scenario, trajectory, seed, variant)
            errors.append(100*np.linalg.norm(data['physical'][:, :3]-data['reference'][:, :3], axis=1))
        errors = np.asarray(errors)
        ax.plot(data['t'], errors.mean(0), label=LABELS[variant], color=COLORS[variant], ls=STYLES[variant], lw=1.4)
        ax.fill_between(data['t'], errors.min(0), errors.max(0), color=COLORS[variant], alpha=.08, linewidth=0)
    ax.set(xlabel='Time (s)', ylabel='Position error (cm)')
    ax.grid(alpha=.2)


def figures(out, frame):
    directory = out/'figures'; directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), layout='constrained', sharex=True)
    for label, ax, kind in zip('abcd', axes.flat, ['hover', 'circle', 'figure8', 'helix']):
        error_plot(ax, out, frame, 'nominal', kind, ORDER)
        ax.set_title(f'({label}) {kind.capitalize()}')
    axes[0, 0].legend(fontsize=8, loc='center right')
    for extension in ['png', 'pdf']:
        fig.savefig(directory/f'nominal_tracking.{extension}', dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), layout='constrained')
    for label, ax, kind in zip('ab', axes[0], ['hover', 'figure8']):
        error_plot(ax, out, frame, 'bias_step', kind, ['equilibrium', 'combined'])
        ax.axvline(8, color='#777777', ls=':', lw=1)
        ax.set_title(f'({label}) {kind.capitalize()}: disturbance at 8 s')
    axes[0, 0].legend(fontsize=8)
    seed = int(frame.seed.min())
    data = load_case(out, 'bias_step', 'hover', seed, 'combined')
    for offset, ax, title, unit in [(0, axes[1, 0], '(c) Force residual', r'm/s$^2$'),
                                    (3, axes[1, 1], '(d) Angular residual', r'rad/s$^2$')]:
        for j, color in enumerate(['#0072B2', '#D55E00', '#009E73']):
            ax.plot(data['t'], data['bias_estimate'][:, offset+j], color=color, label='xyz'[j], lw=1.3)
            ax.plot(data['t'][:-1], data['injected_body_acceleration'][:, offset+j], color=color, ls='--', lw=.8, alpha=.7)
        ax.axvline(8, color='#777777', ls=':', lw=1)
        ax.set(title=title, xlabel='Time (s)', ylabel=unit); ax.grid(alpha=.2)
    axes[1, 0].legend(fontsize=8, ncol=3)
    fig.supxlabel('Bottom row: solid = estimated total residual; dashed = injected increment.', fontsize=8)
    for extension in ['png', 'pdf']:
        fig.savefig(directory/f'bias_compensation.{extension}', dpi=300)
    plt.close(fig)


def md_table(data, include_trajectory=False):
    headers = (['轨迹'] if include_trajectory else [])+['版本', '例数', '整段 RMSE (m)', '后 10 s RMSE (m)', '后 10 s 平均偏移范数 (m)']
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for r in data.itertuples(index=False):
        values = ([r.trajectory] if include_trajectory else [])+[CN[r.variant], str(r.cases), f'{r.position_rmse_m:.6f}', f'{r.late_rmse_m:.6f}', f'{r.late_bias_norm_m:.6f}']
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main(out):
    out = out.resolve()
    verification = json.loads((out/'verification.json').read_text())
    assert verification['status'] == 'PASS'
    assert verification['tracking_sha256'] == checksum(out/'tracking_per_run.csv')
    frame = pd.read_csv(out/'tracking_per_run.csv')
    assert frame.complete.all() and len(frame) == 40
    summary = summary_table(frame, ['scenario', 'variant'])
    by_trajectory = summary_table(frame, ['scenario', 'trajectory', 'variant'])
    summary.to_csv(out/'tracking_summary.csv', index=False)
    by_trajectory.to_csv(out/'tracking_by_trajectory.csv', index=False)
    pairs = []
    for scenario, base in [('nominal', 'original'), ('nominal', 'equilibrium'), ('bias_step', 'equilibrium')]:
        part = frame[frame.scenario == scenario]
        a = part[part.variant == base].set_index(['trajectory', 'seed'])
        b = part[part.variant == 'combined'].set_index(['trajectory', 'seed'])
        assert a.index.equals(b.index)
        for idx in a.index:
            pairs.append(dict(scenario=scenario, baseline=base, trajectory=idx[0], seed=idx[1],
                              baseline_rmse_m=a.loc[idx, 'position_rmse_m'], combined_rmse_m=b.loc[idx, 'position_rmse_m'],
                              reduction_percent=100*(1-b.loc[idx, 'position_rmse_m']/a.loc[idx, 'position_rmse_m'])))
    pd.DataFrame(pairs).to_csv(out/'paired_comparisons.csv', index=False)
    figures(out, frame)
    nominal = summary[summary.scenario == 'nominal'].set_index('variant')
    perturbed = summary[summary.scenario == 'bias_step'].set_index('variant')
    reduction = 100*(1-nominal.loc['combined', 'position_rmse_m']/nominal.loc['original', 'position_rmse_m'])
    disturbance_reduction = 100*(1-perturbed.loc['combined', 'position_rmse_m']/perturbed.loc['equilibrium', 'position_rmse_m'])
    nominal_pairs = [p for p in pairs if p['scenario'] == 'nominal' and p['baseline'] == 'equilibrium']
    worse = [p for p in nominal_pairs if p['combined_rmse_m'] > p['baseline_rmse_m']]
    worse_note = '；'.join(f"{p['trajectory']}/{p['seed']}: {p['baseline_rmse_m']:.6f} → {p['combined_rmse_m']:.6f} m" for p in worse) or '无'
    all_controls = pd.concat([pd.read_csv(out/'cases'/f'{r.scenario}_{r.trajectory}_{r.seed}_{r.variant}'/'control.csv')
                             for r in frame.itertuples(index=False)], ignore_index=True)
    settings = json.loads((out/'config.json').read_text()); eq = json.loads((out/'equilibrium_calibration.json').read_text())
    nominal_detail = by_trajectory[by_trajectory.scenario == 'nominal']
    disturbance_detail = by_trajectory[by_trajectory.scenario == 'bias_step']
    text = f'''# E3 平衡点与在线偏差补偿：实现与验证结果

## Material Passport

- 状态：已实现；40 个冻结参数的配对验证算例全部运行完成；独立核验 PASS。
- 数据来源：已有 Pelican E2 历史 SINDy 模型，以及训练数据校准的 E3 名义仿真器。
- 证据性质：仿真闭环，两个验证初态种子 11301、11302；未新增真实飞行实验。
- 原始结果：`{out.relative_to(ROOT).as_posix()}/tracking_per_run.csv` 及 `cases/`。
- 协议：`docs/PELICAN_E3_CORRECTED_PROTOCOL.md`；代码：`pelican_e3_corrected/`。

## 实际修复了什么

原模型在相同名义悬停状态和指令下仍输出非零加速度。新模块复制模型，并从输出截距减去
`{np.array(eq['removed_acceleration_bias']).round(9).tolist()}`。
前三项单位为 m/s²，后三项为 rad/s²。校正后完整历史悬停状态的一步固定点最大误差为
`{eq['packed_fixed_point_max_error']:.3g}`。原稀疏系数、历史抽头和原始文件不变。

在线模块使用每 0.01 s 到达的后向差分观测，以及刚刚实际施加的电机指令，计算机体系残差。
六维残差以 0.5 s 时间常数滤波，每次 MPC 求解使用当时已有估计并在时域内保持不变。
残差预测不包含在线估计，因此不存在重复扣除偏差。六维偏差进入加速度后统一积分所有运动学状态。
MPC 原代价、时域、指令约束和求解器设置保留。

## 名义场景结果

每个版本包含悬停、圆、八字、螺旋四条轨迹，各两个配对初态，共八例；每例 20 s。
下表是每例指标的算术平均，整段指标包含初始误差。后段单独统计 10–20 s，不能与整段指标混用。

{md_table(summary[summary.scenario == 'nominal'])}

结合版相对原版的平均整段 RMSE 降低 **{reduction:.2f}%**，按两组平均 RMSE 的比值计算。
相对仅平衡点修正版，结合版在 {len(nominal_pairs)-len(worse)}/{len(nominal_pairs)} 个名义配对算例中整段 RMSE 更低；
未改善的配对为：{worse_note}。不把两个初态当作大样本统计检验，也不要求在线补偿在已充分匹配的名义悬停下必然改善所有瞬态指标。

{md_table(nominal_detail, True)}

## 运行后新增持续偏差

扰动在第 8 s 加入物理仿真器，未传给校准或控制器。机体系比力增量为
`{settings['disturbance']['body_acceleration'][:3]} m/s²`，角加速度增量为
`{settings['disturbance']['body_acceleration'][3:]} rad/s²`。悬停和八字各两个初态，共八例。
这里比较仅平衡点修正与结合版，分别观察离线校准和在线补偿的能力。

{md_table(summary[summary.scenario == 'bias_step'])}

加入在线补偿后，平均整段 RMSE 相比仅平衡点修正降低 **{disturbance_reduction:.2f}%**。
该结果支持对本组恒定附加偏差的补偿效果，不能外推为任意风场或任意参数漂移的鲁棒性保证。
新增扰动下仍有约 1 cm 的后段平均位置偏移，不能称为完全消除了任意持续扰动造成的稳态误差。

{md_table(disturbance_detail, True)}

## 数值核验和运行情况

- 六项单元测试通过：平衡点及原模型保护、零偏差预测和多步 NumPy/CasADi 对照、旋转坐标系下的偏差收敛、实际输入与异常观测处理、零偏差 MPC 与原版一致、物理积分对照。
- 独立批量重算 {verification['residual_rows']:,} 个残差，最大误差 {verification['max_residual_replay_error']:.3g}；核对 {verification['control_bias_checks']:,} 次求解的偏差索引，均只包含当前及过去观测。
- 独立自适应积分抽查 {verification['adaptive_plant_steps']} 个物理转移，最大状态分量差 {verification['max_adaptive_state_error']:.3g}（混合单位，逐分量容差检查）。
- 求解失败 {int(frame.solver_failures.sum())} 次；PD 回退 {int(frame.fallbacks.sum())} 次；指令上下界违规 {int(frame.input_bound_violations.sum())} 次；变化率违规 {int(frame.input_slew_violations.sum())} 次。
- 观测器拒绝 {int(frame.observer_rejected.sum())} 次；估计投影到上限 {int(frame.observer_clipped.sum())} 次。
- 全部 {len(all_controls):,} 次控制调用中，记录的控制段墙钟耗时超过 0.05 s 的比例为 {100*(all_controls.controller_wall_s > .05).mean():.2f}%；该计时含求解和回退/限幅，不含先行历史打包及观测器更新，只是本机离线运行记录，不是端到端或硬实时保证。
- 原 E3 的 {verification['original_artifacts_verified']['pelican_e3_v1']} 个封存文件，以及原诊断的 {verification['original_artifacts_verified']['pelican_e3_diagnostic_v1']} 个封存文件，SHA256 全部保持一致。

## 图与复用入口

- `figures/nominal_tracking.pdf`：四轨迹位置误差，线为两个初态的平均，浅色带为范围，**不是置信区间**。
- `figures/bias_compensation.pdf`：新增扰动下的误差和估计。估计曲线选预先固定的第一个验证种子 11301；实线为总残余误差估计，虚线为注入增量，两者含义不同。
- `tracking_by_trajectory.csv`：各轨迹汇总；`paired_comparisons.csv`：逐初态配对改善量。
- `cases/*/trajectory.npz`：物理状态、可观测状态、参考、实际指令、估计、残差和事件标记。
- `cases/*/control.csv`：每次求解的成功状态、耗时、实际指令和当时使用的偏差。
- `verification.json`：独立核验结果；`FROZEN.json`：运行前固定的代码、配置和模型哈希。

## 论文可支持的表述与限制

可以写：针对 Pelican 辨识模型迁移到名义仿真器后的悬停不一致，采用平衡点校准与因果在线残差补偿；
在本次小规模配对仿真中降低了位置误差，具体幅度来自已保存、可重算的结果。本次未进行统计显著性检验。

必须说明：平衡点使用已知名义仿真器，校准不是由真实测试轨迹拟合，也不是完整的受约束 SINDy 重训练。
这不是实机飞行结论，也不是一般性 offset-free MPC 保证。未给电机或线性基线施加同等补偿，因此不据此宣称历史 SINDy 优于那些方法。
原 E3 和原诊断结果保留，本文结果是明确针对已发现问题的后续修复验证。当前两种初态不足以支持广泛泛化或新方法优越性声明。
观测沿用原 E3 的仿真位姿，未额外加入传感器噪声；本次没有验证噪声下的差分残差补偿性能。
'''
    report = ROOT/'docs/PELICAN_E3_CORRECTED_RESULTS.md'
    report.write_text(text, encoding='utf-8')
    (out/'REPORT.md').write_text(text, encoding='utf-8')
    # Round-trip the published summaries, independently grouped from per-case metrics.
    reread = pd.read_csv(out/'tracking_summary.csv')
    for r in reread.itertuples(index=False):
        group = frame[(frame.scenario == r.scenario) & (frame.variant == r.variant)]
        np.testing.assert_allclose(r.position_rmse_m, group.position_rmse_m.mean(), atol=1e-14)
        np.testing.assert_allclose(r.late_rmse_m, group.late_rmse_m.mean(), atol=1e-14)
    paths = [p for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json']
    paths += list((ROOT/'pelican_e3_corrected').glob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/run_pelican_e3_corrected.py', ROOT/'scripts/verify_pelican_e3_corrected.py',
              ROOT/'tests/test_pelican_e3_corrected.py', ROOT/'configs/pelican_e3_corrected.json',
              ROOT/'docs/PELICAN_E3_CORRECTED_PROTOCOL.md', report]
    write_json(out/'artifact_hashes.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
               files={p.relative_to(ROOT).as_posix(): dict(sha256=checksum(p), bytes=p.stat().st_size) for p in sorted(set(paths))}))
    print(summary.to_string(index=False))
    print('Report:', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir)
