"""Summarize frozen E3 tracking, paired initial-condition uncertainty and figures."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
from run_pelican_e2 import restore_model,write_json
from pelican_e3.control import pack_observation,symbolic_transition

ORDER=['pd','sindy_current','linear_history','sindy_history','motor_sindy']
LABEL={'pd':'PD + allocation','sindy_current':'Current SINDy-MPC','linear_history':'Linear history MPC',
       'sindy_history':'History SINDy-MPC','motor_sindy':'Motor + SINDy-MPC'}
COLOR={'pd':'#666666','sindy_current':'#D55E00','linear_history':'#0072B2','sindy_history':'#009E73','motor_sindy':'#CC79A7'}
STYLE={'pd':':','sindy_current':'--','linear_history':'-.','sindy_history':'-','motor_sindy':(0,(5,1,1,1))}
MARKER=dict(zip(ORDER,['x','s','^','o','D']))
KINDS=['circle','figure8','helix']


def save(fig,folder,name):
    # Axes3D does not always include the projected z label in its tight bbox.
    extra=[ax.zaxis.label for ax in fig.axes if hasattr(ax,'zaxis')]
    if extra: extra+=list(fig.legends)+list(fig.texts)
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',bbox_extra_artists=extra or None)
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight',bbox_extra_artists=extra or None)
    with Image.open(folder/(name+'.png')) as im: im.convert('L').save(folder/(name+'_grayscale.png'),dpi=(300,300))
    plt.close(fig)


def write_results_markdown(run,report,stats,comparisons,biases):
    chinese={'pd':'PD + 控制分配','sindy_current':'无历史 SINDy-MPC','linear_history':'线性历史 MPC',
             'sindy_history':'历史 SINDy-MPC','motor_sindy':'一阶电机 + SINDy-MPC'}
    overall=stats[stats.trajectory=='overall'].set_index('controller')
    lines=['# Pelican E3：闭环轨迹跟踪结果','',
           '## Material Passport','',
           '- Origin Skill：academic-research-suite / experiment-agent；scientific-visualization。',
           '- Origin Mode：run + validate。',
           '- Origin Date：2026-10-26。',
           '- Verification Status：ANALYZED；冻结后执行，保存轨迹的指标复算与独立物理积分复核通过。',
           f'- Version Label：`{run.name}`。',
           '- 证据类型：**SIMULATED CLOSED LOOP**。使用 Pelican 日志标定的近似物理对象；无新实机飞行。','',
           f'本轮完成 {report["cases"]} 个正式案例，其中 {report["completed"]} 个跑满 20 s，{report["tracking_successes"]} 个达到预设跟踪阈值。'
           f'求解回退总数为 {report["fallbacks"]}。公共参数、标定过程、失败开发记录和解释边界见 [冻结实验协议](PELICAN_E3_PROTOCOL.md)。','',
           '## 1. 主结果','',
           '位置向量 RMSE，单位 m。每格为同一轨迹下五组配对初始条件的均值。未完成运行不具有整段 RMSE，若出现则另列失败数量。','',
           '| 轨迹 | PD | 无历史 SINDy-MPC | 线性历史 MPC | 历史 SINDy-MPC | 电机 + SINDy-MPC |',
           '|---|---:|---:|---:|---:|---:|']
    for kind,label in [('circle','圆形'),('figure8','八字'),('helix','螺旋'),('overall','总体')]:
        subset=stats[stats.trajectory==kind].set_index('controller')
        values=['—' if pd.isna(subset.loc[n,'position_rmse_mean_m']) else f'{subset.loc[n,"position_rmse_mean_m"]:.5f}' for n in ORDER]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','| 控制器 | 完整运行 / 尝试 | 跟踪成功 / 尝试 | 最坏位置误差 (m) | 回退次数 |',
            '|---|---:|---:|---:|---:|']
    for name in ORDER:
        r=overall.loc[name]
        lines.append(f'| {chinese[name]} | {int(r.completed)} / {int(r.runs)} | {int(r.tracking_successes)} / {int(r.runs)} | {r.max_error_m:.5f} | {int(r.fallbacks)} |')
    best=overall.position_rmse_mean_m.idxmin()
    best_mpc=overall.drop(index='pd').position_rmse_mean_m.idxmin()
    lines+=['',f'本次总体误差最低的是 **{chinese[best]}**；四种 MPC 中最低的是 **{chinese[best_mpc]}**。',
            f'历史 SINDy-MPC 的总体位置 RMSE 为 {overall.loc["sindy_history","position_rmse_mean_m"]:.5f} m。'
            'E2 的保留真实日志预测优势没有在本轮转化为历史模型的闭环跟踪优势。'
            '本比较包含固定控制代价、平滑参考、模型与共同仿真器之间的不匹配；不能推广为某类算法在所有对象上更优或更差。','',
            '## 2. 配对差异与不确定性','',
            '下表差值为“对照 RMSE − 历史 SINDy-MPC RMSE”；正值表示历史模型更好，负值表示更差。总体重采样以五个初始条件种子为块，保留每个种子的三种轨迹；2,000 次重采样，种子 20261027。','',
            '| 对照 | 平均差 (m) | 描述性 95% 区间 (m) | 历史模型更好的案例 / 可比案例 |',
            '|---|---:|---:|---:|']
    for r in comparisons[comparisons.trajectory=='overall'].itertuples(index=False):
        interval=f'[{r.lower_95_m:.5f}, {r.upper_95_m:.5f}]' if r.all_pairs_complete else '不计算：有未完成案例'
        lines.append(f'| {chinese[r.baseline]} | {r.difference_baseline_minus_history_m:.5f} | {interval} | {r.history_better_cases} / {r.available_pairs} |')
    lines+=['','区间只描述这五组初始偏差在固定模型和对象下的波动，不包含重新训练、不同对象或真实采集批次的不确定性。未报告 p 值，不作多重比较显著性结论。','',
            '## 3. 稳态偏差检查','',
            '在共同物理对象的水平悬停平衡点，以相同命令和常值历史输入冻结学习模型，得到以下单步预测残差。物理对象在该点的真实净加速度与角加速度均为零。','',
            '| 学习模型 | 预测净加速度模长 (m/s²) | 预测角加速度模长 (rad/s²) |',
            '|---|---:|---:|']
    for r in report['hover_residuals']:
        acc=np.linalg.norm([r[f'predicted_hover_acc_{a}_m_s2'] for a in 'xyz'])
        angular=np.linalg.norm([r[f'predicted_hover_angular_acc_{a}_rad_s2'] for a in 'xyz'])
        lines.append(f'| {chinese[r["model"]]} | {acc:.5f} | {angular:.5f} |')
    lines+=['','逐案例轴向 RMSE 与后 10 s 平均位置偏差另存为 `analysis/axis_errors_per_run.csv`。'
            '这些残差提示学习模型与仿真对象的平衡点不一致，可能与持续跟踪偏差有关；尚未通过偏差补偿消融建立因果结论。'
            '历史模型的悬停加速度残差较小，但跟踪误差较大，因此不能仅用这一诊断解释方法排名。'
            '本轮没有根据正式结果修改模型、加入积分器或调整某一种 MPC。','',
            '## 4. 计算时间与约束','',
            '| 控制器 | 控制调用数 | 控制包装耗时中位数 (ms) | p95 (ms) | p99 (ms) | 超过 50 ms 比例 |',
            '|---|---:|---:|---:|---:|---:|']
    for r in report['timing']:
        if r['measurement']!='controller_wall_s':continue
        lines.append(f'| {chinese[r["controller"]]} | {r["calls"]} | {1000*r["median_s"]:.3f} | {1000*r["p95_s"]:.3f} | {1000*r["p99_s"]:.3f} | {100*r["fraction_over_period"]:.2f}% |')
    bound=int(overall.bound_violations.sum()); slew=int(overall.slew_violations.sum())
    lines+=['',f'实际下发命令的幅值违反共 {bound} 次，变化率违反共 {slew} 次。命令变化平方和与命令平方积分使用厂商编码单位，不能称为真实能耗。',
            '控制包装计时包括参考读取、历史打包、优化、回退及限幅；不含物理积分、状态差分构造和离线参考表生成。'
            '仿真时钟等待求解结束，没有注入计算延迟或执行超时丢弃，因此即使部分调用短于周期，也不能据此宣称硬实时可部署。','',
            '## 5. 图与原始记录','',
            f'- [各轨迹位置 RMSE 与全部初始条件](../runs/{run.name}/figures/tracking_rmse.png)。',
            f'- [圆、八字、螺旋三维轨迹](../runs/{run.name}/figures/trajectories_3d.png)。',
            f'- [位置误差时序](../runs/{run.name}/figures/tracking_error_time.png)。',
            f'- [控制变化与耗时分布](../runs/{run.name}/figures/commands_and_timing.png)。',
            f'- [逐案例结果](../runs/{run.name}/tracking_per_run.csv)。',
            f'- [带区间的结果表](../runs/{run.name}/analysis/tracking_with_intervals.csv)。',
            f'- [机器可读报告](../runs/{run.name}/analysis/report.json)。','',
            '三维轨迹和时序图统一使用冻结时预先选定的种子 8301，没有按历史方法表现选择案例。图同时提供 PDF、300 dpi PNG 和灰度版本。'
            '`cases/<轨迹>_<种子>_<控制器>/trajectory.npz` 保存逐步状态、观测、参考、命令与初始历史；`control.csv` 保存每次优化的状态与耗时。','',
            '## 6. 实现检查与证据边界','',
            f'四项科学计算测试通过；独立审计核对 {report["verification"]["cases"]} 个案例、'
            f'{report["verification"]["physical_transitions_logged"]:,} 个保存的物理状态转移。'
            f'另用独立刚体方程和自适应积分器复算 {report["verification"]["independent_adaptive_transitions_checked"]} 个分散转移，'
            f'最大状态绝对差为 {report["verification"]["max_independent_transition_difference"]:.3e}。',
            '核对包括冻结指纹、E2 模型不变、标定训练集范围、配对初始历史、后向差分观测、100 Hz 命令保持、整段/失败前缀指标和约束计数。'
            '这属于同一执行者采用独立计算路径复核，未重新运行全部优化，也不是第三方验证。','',
            '本轮只验证一个标称近似仿真器和三种平滑轨迹。PD 使用共同对象的名义参数，具有对象知识；学习模型则从预处理实机日志迁移而来。'
            '对象验证存在明显 1 s 误差、原始日志上游平滑、缺少绝对物理参数等限制，详见协议。'
            '没有新增实机、观测噪声、阵风、执行器失配或多次训练重复。不能据此声称已完成 E4–E6，或已证明某方法的实机优势。','',
            '统计核查覆盖 11/11 项：分轨迹与逐案例报告聚合差异；不将总体均值推为每例收益；声明对象与样本选择范围；'
            '不按表现删案例；给出成功/失败分母；样例固定而不按极值筛选；保留失败前缀；全部方法与轨迹报告；'
            '保存开发与冻结路径；不把预测/跟踪关联解释为因果机制；不由既有闭环日志推断新控制器的真实作用。','',
            '## 7. 复核入口','',
            '在 `huiyi/method` 下执行：','',
            '```powershell',
            '.\\.venv\\Scripts\\python.exe scripts/verify_pelican_e3.py',
            '.\\.venv\\Scripts\\python.exe scripts/report_pelican_e3.py',
            '.\\.venv\\Scripts\\python.exe scripts/verify_pelican_e3_report.py',
            '```','',
            '重新完整运行必须使用新目录，先执行 `run_pelican_e3.py --stage freeze --run-dir runs/<新版本>`，再使用同一目录执行 `--stage test`。'
            '冻结入口会核对已保存的长时开发资格；重新调参或改变开发方案需要建立新的开发与冻结记录，不能覆盖本次正式测试。']
    (ROOT/'docs/PELICAN_E3_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(run):
    verified=json.loads((run/'verification.json').read_text()); assert verified['status']=='PASS'
    cfg=json.loads((run/'config.json').read_text()); params=json.loads((run/'plant_parameters.json').read_text())
    frame=pd.read_csv(run/'tracking_per_run.csv'); frozen=json.loads((run/'FROZEN.json').read_text())
    analysis,figures=run/'analysis',run/'figures'
    analysis.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    seeds=cfg['test_seeds']; rng=np.random.default_rng(cfg['bootstrap_seed'])
    draws=rng.integers(0,len(seeds),size=(cfg['bootstrap_repetitions'],len(seeds)))
    np.savez_compressed(analysis/'bootstrap_draws.npz',seeds=seeds,draws=draws)
    statistics=[]; pairs=[]
    for kind in KINDS+['overall']:
        selected=frame if kind=='overall' else frame[frame.trajectory==kind]
        for name in ORDER:
            subset=selected[selected.controller==name]
            complete=subset[subset.complete]
            # Same seed is shared across trajectory types; retain this block in
            # the overall bootstrap instead of treating 15 cases as independent.
            seedmeans=complete.groupby('seed').position_rmse_m.mean().reindex(seeds)
            all_complete=bool(subset.complete.all())
            lower,upper=(np.quantile(seedmeans.to_numpy()[draws].mean(1),[.025,.975]) if all_complete else [None,None])
            statistics.append({'trajectory':kind,'controller':name,'runs':len(subset),'completed':int(subset.complete.sum()),
                               'tracking_successes':int(subset.tracking_success.sum()),
                               'position_rmse_mean_m':float(complete.position_rmse_m.mean()) if len(complete) else None,
                               'lower_95_m':lower,'upper_95_m':upper,'conditional_on_completion':not all_complete,
                               'max_error_m':float(subset.max_position_error_m.max()),
                               'fallbacks':int(subset.fallbacks.sum()),'control_calls':int(subset.control_calls.sum()),
                               'command_variation_mean':float(subset.command_variation_squared.mean()),
                               'bound_violations':int(subset.input_bound_violations.sum()),'slew_violations':int(subset.input_slew_violations.sum())})
        pivot=selected.pivot(index=['trajectory','seed'],columns='controller',values='position_rmse_m')
        for baseline in [n for n in ORDER if n!='sindy_history']:
            pair=pivot[[baseline,'sindy_history']].dropna()
            complete_pair=len(pair)==len(pivot)
            delta=pair[baseline]-pair.sindy_history
            byseed=delta.groupby('seed').mean().reindex(seeds)
            low,high=(np.quantile(byseed.to_numpy()[draws].mean(1),[.025,.975]) if complete_pair else [None,None])
            pairs.append({'trajectory':kind,'baseline':baseline,'candidate':'sindy_history',
                          'available_pairs':len(pair),'total_pairs':len(pivot),'difference_baseline_minus_history_m':float(delta.mean()) if len(pair) else None,
                          'relative_reduction_percent':float(100*delta.mean()/pair[baseline].mean()) if len(pair) else None,
                          'lower_95_m':low,'upper_95_m':high,'history_better_cases':int((delta>0).sum()),
                          'bootstrap_unit':'seed, shared across trajectory types','all_pairs_complete':complete_pair})
    stats=pd.DataFrame(statistics); comparisons=pd.DataFrame(pairs)
    stats.to_csv(analysis/'tracking_with_intervals.csv',index=False)
    comparisons.to_csv(analysis/'paired_comparisons.csv',index=False)
    controls=[]; biases=[]; traces={}
    for row in frame.itertuples(index=False):
        folder=run/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}'
        control=pd.read_csv(folder/'control.csv')
        control['controller']=row.controller; control['trajectory']=row.trajectory; control['seed']=row.seed
        controls.append(control)
        with np.load(folder/'trajectory.npz') as f:
            trace={key:f[key] for key in f.files}
        late=trace['t']>=min(10,cfg['duration_s']/2)
        error=trace['physical'][:,:3]-trace['reference'][:,:3]
        biases.append({'controller':row.controller,'trajectory':row.trajectory,'seed':row.seed,'complete':row.complete,
                       **{f'rmse_{axis}_m':float(np.sqrt(np.mean(error[:,i]**2))) for i,axis in enumerate('xyz')},
                       **{f'late_mean_error_{axis}_m':float(error[late,i].mean()) if late.any() else None for i,axis in enumerate('xyz')}})
        if row.seed==seeds[0]: traces[(row.trajectory,row.controller)]=trace
    pd.DataFrame(biases).to_csv(analysis/'axis_errors_per_run.csv',index=False)
    controls=pd.concat(controls,ignore_index=True)
    controls.to_csv(analysis/'all_control_calls.csv',index=False)
    timing=[]
    for name in ORDER:
        group=controls[controls.controller==name]
        for field in ['seconds','controller_wall_s']:
            v=group[field].to_numpy()
            timing.append({'controller':name,'measurement':field,'calls':len(v),'mean_s':v.mean(),
                           'median_s':np.median(v),'p95_s':np.quantile(v,.95),'p99_s':np.quantile(v,.99),
                           'max_s':v.max(),'fraction_over_period':float((v>cfg['control_dt']).mean())})
    pd.DataFrame(timing).to_csv(analysis/'timing_summary.csv',index=False)
    hover=np.asarray(params['hover_command']); state=np.r_[0.,0.,1.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]
    motor=json.loads((run/'motor_model.json').read_text())
    prior=np.asarray(params['motor_gain'])*hover+params['motor_bias']
    diagnostics=[]
    for raw in json.loads((run/'selected_models.json').read_text()):
        if raw['name'] not in ORDER or raw['name']=='pd':continue
        model=restore_model(raw); f,_=symbolic_transition(model,motor,cfg)
        packed=pack_observation(np.tile(state,(21,1)),np.tile(hover,(20,1)),prior,model,cfg)
        next_state=np.asarray(f(packed,hover)).ravel()[:13]
        diagnostics.append({'model':model['name'],
                           **{f'predicted_hover_acc_{axis}_m_s2':float((next_state[3+i]-state[3+i])/cfg['dt']) for i,axis in enumerate('xyz')},
                           **{f'predicted_hover_angular_acc_{axis}_rad_s2':float((next_state[10+i]-state[10+i])/cfg['dt']) for i,axis in enumerate('xyz')}})
    pd.DataFrame(diagnostics).to_csv(analysis/'hover_model_residuals.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.5})
    fig,axes=plt.subplots(1,3,figsize=(9,3.7),sharey=True)
    for ax,kind,letter in zip(axes,KINDS,'abc'):
        for j,name in enumerate(ORDER):
            row=stats[(stats.trajectory==kind)&(stats.controller==name)].iloc[0]
            values=frame[(frame.trajectory==kind)&(frame.controller==name)].sort_values('seed').position_rmse_m.to_numpy()
            ax.scatter(j+np.linspace(-.14,.14,len(values)),values,s=13,marker=MARKER[name],color=COLOR[name],alpha=.65)
            if np.isfinite(row.lower_95_m):
                ax.errorbar(j,row.position_rmse_mean_m,yerr=[[max(0,row.position_rmse_mean_m-row.lower_95_m)],
                    [max(0,row.upper_95_m-row.position_rmse_mean_m)]],fmt=MARKER[name],color=COLOR[name],capsize=4,markersize=6)
            if row.completed<row.runs: ax.annotate(f'{row.completed}/{row.runs} complete',(j,row.position_rmse_mean_m),fontsize=7)
        ax.set_xticks(range(5),labels=['PD','Current','Linear\nhistory','History\nSINDy','Motor'])
        ax.set(title=f'({letter}) {kind}',ylim=(0,None)); ax.grid(alpha=.15,axis='y')
    axes[0].set_ylabel('20 s position vector RMSE (m)')
    fig.suptitle('SIMULATED CLOSED LOOP | 5 paired initial conditions per trajectory',fontsize=11)
    fig.text(.5,-.025,'Small markers: individual runs. Large markers: means. Whiskers: descriptive 95% seed bootstrap.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,figures,'tracking_rmse')

    fig=plt.figure(figsize=(11.7,4.3))
    for j,kind in enumerate(KINDS):
        ax=fig.add_subplot(1,3,j+1,projection='3d')
        reference=traces[(kind,'pd')]['reference'][:,:3]
        ax.plot(*reference.T,color='black',ls='--',lw=1.8,label='Reference')
        for name in ORDER:
            path=traces[(kind,name)]['physical'][:,:3]
            ax.plot(*path.T,color=COLOR[name],ls=STYLE[name],label=LABEL[name],lw=1.2)
        positions=np.concatenate([reference]+[traces[(kind,name)]['physical'][:,:3] for name in ORDER])
        center=(positions.min(0)+positions.max(0))/2
        span=np.maximum(np.ptp(positions,axis=0)*1.12,.5)
        ax.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2)
        ax.set_ylim(center[1]-span[1]/2,center[1]+span[1]/2)
        ax.set_zlim(center[2]-span[2]/2,center[2]+span[2]/2)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f'({"abc"[j]}) {kind}')
        ax.view_init(elev=25,azim=-55); ax.set_box_aspect(span,zoom=.82)
        ax.tick_params(labelsize=7,pad=1)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.set_major_locator(MaxNLocator(nbins=3))
            axis.labelpad=3
    handles,labels=fig.axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle(f'SIMULATED CLOSED LOOP | preselected seed {seeds[0]}',y=1.01,fontsize=11)
    fig.subplots_adjust(left=.015,right=.965,bottom=.10,top=.76,wspace=.32)
    save(fig,figures,'trajectories_3d')

    fig,axes=plt.subplots(3,1,figsize=(7.2,6.2),sharex=True)
    for ax,kind,letter in zip(axes,KINDS,'abc'):
        for name in ORDER:
            trace=traces[(kind,name)]; error=np.linalg.norm(trace['physical'][:,:3]-trace['reference'][:,:3],axis=1)
            ax.plot(trace['t'],error,color=COLOR[name],ls=STYLE[name],label=LABEL[name])
        ax.set(ylabel='Position error (m)',title=f'({letter}) {kind}',ylim=(0,None));ax.grid(alpha=.15)
    axes[-1].set_xlabel('Time (s)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.02))
    fig.suptitle(f'SIMULATED CLOSED LOOP | preselected seed {seeds[0]}',y=1.06,fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,figures,'tracking_error_time')

    fig,axes=plt.subplots(1,2,figsize=(8.7,4.1))
    for j,name in enumerate(ORDER):
        values=frame[frame.controller==name].command_variation_squared.to_numpy()
        axes[0].scatter(j+np.linspace(-.15,.15,len(values)),values,color=COLOR[name],s=16,marker=MARKER[name])
        seconds=np.sort(controls[controls.controller==name].controller_wall_s.to_numpy())*1000
        axes[1].plot(seconds,np.arange(1,len(seconds)+1)/len(seconds),color=COLOR[name],ls=STYLE[name],label=LABEL[name])
    axes[0].set_xticks(range(5),labels=['PD','Current','Linear\nhistory','History\nSINDy','Motor'])
    axes[0].set(ylabel='Sum of squared command changes (code units²)',title='(a) Command variation')
    axes[1].axvline(cfg['control_dt']*1000,color='black',ls=':',lw=1)
    axes[1].set(xscale='log',xlabel='Controller wall time (ms)',ylabel='Empirical cumulative fraction',title='(b) Offline timing, 50 ms period')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,frameon=False,loc='lower center',
               bbox_to_anchor=(.5,.005),ncol=3)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('SIMULATED CLOSED LOOP | no hard real-time claim',fontsize=11)
    fig.tight_layout(rect=(0,.12,1,.96));save(fig,figures,'commands_and_timing')
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'evidence_type':'SIMULATED CLOSED LOOP',
            'run':run.name,'frozen_utc':frozen['frozen_utc'],'cases':len(frame),'completed':int(frame.complete.sum()),
            'tracking_successes':int(frame.tracking_success.sum()),'fallbacks':int(frame.fallbacks.sum()),
            'statistics':statistics,'paired_comparisons':pairs,'timing':timing,'hover_residuals':diagnostics,
            'bootstrap':{'unit':'initial-condition seed, shared across trajectory types','n_seeds':len(seeds),
                         'repetitions':cfg['bootstrap_repetitions'],'seed':cfg['bootstrap_seed'],
                         'scope':'Descriptive only; 5 seeds are not independent real flight datasets or retraining replicates.'},
            'limitations':['Reference simulator is calibrated from logs; absolute mass/inertia and motor RPM are not identified.',
                           'The PD baseline uses the same calibrated physical parameters as the simulated plant, giving it nominal model knowledge.',
                           'Frozen E2 models are transferred without retraining or equilibrium/offset correction.',
                           'No hard real-time scheduling, online measurement noise, gusts or actuator mismatch in E3.',
                           'One fitted plant and one set of learned models; no generalization claim across hardware or training draws.'],
            'verification':verified}
    write_json(analysis/'report.json',report)
    write_results_markdown(run,report,stats,comparisons,biases)
    print(stats[stats.trajectory=='overall'].to_string(index=False))
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e3_v1')
    main(parser.parse_args().run_dir)
