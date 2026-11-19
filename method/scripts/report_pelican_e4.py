"""E4 descriptive tables and physical trajectory figures, retaining all failures."""
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
from run_pelican_e4 import check_preserved
from pelican_e3.reference import at
from plot_pelican_e3_trajectories import path_limits
from report_pelican_e3_nominal_comparison import ORDER, COLORS, STYLES, TITLES, CN, add_legend, savefig

SCENARIOS = ['nominal','tau_1p5','tau_2','pose_noise','force_pulse']
EN = {'nominal':'Nominal','tau_1p5':'Motor time constant x1.5','tau_2':'Motor time constant x2',
      'pose_noise':'Pose noise: 1 mm / 0.001 rad','force_pulse':'World acceleration pulse: 8-12 s'}
ZH = {'nominal':'名义','tau_1p5':'时间常数 ×1.5','tau_2':'时间常数 ×2','pose_noise':'位姿噪声','force_pulse':'外力脉冲'}
KINDS = ['circle','figure8']


def load(out, frame):
    data,controls={},{}
    cfg=json.loads((out/'e3_config.json').read_text())
    full_t=np.arange(round(cfg['duration_s']/cfg['dt'])+1)*cfg['dt']
    full_refs={kind:np.array([at(t,kind,cfg)[0] for t in full_t]) for kind in KINDS}
    for row in frame.itertuples(index=False):
        key=(row.scenario,row.trajectory,row.seed,row.controller)
        p=out/'cases'/f'{row.scenario}_{row.trajectory}_{row.seed}_{row.controller}'
        with np.load(p/'trajectory.npz') as f:
            data[key]={k:f[k] for k in ['t','physical','reference','command','bias_estimate','accepted','clipped','world_acceleration']}
        d=data[key]; d['error']=np.linalg.norm(d['physical'][:,:3]-d['reference'][:,:3],axis=1)
        d['reference_full']=full_refs[row.trajectory]; d['t_full']=full_t; d['complete']=bool(row.complete)
        np.testing.assert_allclose(np.sqrt(np.mean(d['error']**2)),row.prefix_position_rmse_m,atol=1e-12)
        controls[key]=pd.read_csv(p/'control.csv')
    return data,controls


def aggregate(frame,controls,by):
    rows=[]
    for key,g in frame.groupby(by,sort=False):
        key=key if isinstance(key,tuple) else (key,)
        r=dict(zip(by,key)); r.update(cases=len(g),completed=int(g.complete.sum()),
            successes=int(g.tracking_success.sum()),failure_fraction=float((~g.tracking_success).mean()),
            early_termination_fraction=float((~g.complete).mean()))
        for field in ['position_rmse_m','late_rmse_m','late_bias_norm_m','command_variation_squared','post_event_rmse_m','event_peak_error_m']:
            r[field]=float(g[field].mean()) if g.complete.all() else None
        r['position_rmse_std_m']=float(g.position_rmse_m.std(ddof=1)) if g.complete.all() else None
        r['completed_subset_position_rmse_m']=float(g.position_rmse_m.mean()) if g.complete.any() else None
        r['worst_position_error_m']=float(g.max_position_error_m.max())
        for field in ['fallbacks','solver_failures','input_bound_violations','input_slew_violations','observer_rejected','observer_clipped']:
            r[field]=int(g[field].sum())
        count=sum(len(controls[(x.scenario,x.trajectory,x.seed,x.controller)]) for x in g.itertuples())
        c=pd.concat([controls[(x.scenario,x.trajectory,x.seed,x.controller)] for x in g.itertuples()])
        r['control_calls']=count
        for field in ['seconds','command_compute_s','full_control_period_s']:
            for suffix,value in [('mean',c[field].mean()),('median',c[field].median()),('p95',c[field].quantile(.95)),
                                 ('p99',c[field].quantile(.99)),('max',c[field].max())]: r[field+'_'+suffix]=float(value)
        r['over_50ms_fraction']=float((c.full_control_period_s>.05).mean())
        r['observer_rejected_fraction']=r['observer_rejected']/max(1,int(np.rint(g.duration_completed_s.sum()/.01)))
        r['recovered_cases']=int(g.recovered.sum())
        r['recovery_time_s']=float(g.recovery_time_s.mean()) if g.recovered.all() and g.complete.all() else None
        rows.append(r)
    return pd.DataFrame(rows)


def paired(frame):
    index=['scenario','trajectory','seed']; history=frame[frame.controller=='sindy_history'].set_index(index)
    rows=[]
    for name in ORDER[:-1]:
        base=frame[frame.controller==name].set_index(index)
        assert set(base.index)==set(history.index)
        for key in history.index:
            a,b=history.loc[key],base.loc[key]; ok=bool(a.complete and b.complete)
            rows.append(dict(zip(index,key),baseline=name,pair_complete=ok,
                history_rmse_m=a.position_rmse_m,baseline_rmse_m=b.position_rmse_m,
                history_minus_baseline_m=a.position_rmse_m-b.position_rmse_m if ok else None,
                reduction_percent=100*(1-a.position_rmse_m/b.position_rmse_m) if ok else None))
    degradation=[]
    base=frame[frame.scenario=='nominal'].set_index(['trajectory','seed','controller'])
    for r in frame[frame.scenario!='nominal'].itertuples(index=False):
        b=base.loc[(r.trajectory,r.seed,r.controller)]; ok=bool(r.complete and b.complete)
        degradation.append(dict(scenario=r.scenario,trajectory=r.trajectory,seed=r.seed,controller=r.controller,
            pair_complete=ok,nominal_rmse_m=b.position_rmse_m,scenario_rmse_m=r.position_rmse_m,
            rmse_increase_m=r.position_rmse_m-b.position_rmse_m if ok else None,
            increase_percent=100*(r.position_rmse_m/b.position_rmse_m-1) if ok else None))
    return pd.DataFrame(rows),pd.DataFrame(degradation)


def paths(ax,data,scenario,kind,seed,mode):
    ref=data[(scenario,kind,seed,'pd')]['reference_full']; allpoints=[ref[:,:3]]
    for name in ORDER:
        d=data[(scenario,kind,seed,name)]; p=d['physical'][:,:3]; allpoints.append(p)
        if mode=='height': ax.plot(d['t'],p[:,2],color=COLORS[name],ls=STYLES[name],lw=1.2)
        else: ax.plot(*p[:,:3 if mode=='3d' else 2].T,color=COLORS[name],ls=STYLES[name],lw=1.4)
        if not d['complete']:
            if mode=='height': ax.scatter(d['t'][-1],p[-1,2],marker='x',color=COLORS[name],s=35,zorder=10)
            else: ax.scatter(*p[-1,:3 if mode=='3d' else 2],marker='x',color=COLORS[name],s=35,zorder=10)
    if mode=='height':
        ax.plot(d['t_full'],ref[:,2],color='black',ls=(0,(5,4)),lw=1)
        ax.set(xlabel='Time (s)',ylabel='Z (m)',xlim=(0,20)); ax.grid(alpha=.2)
        ax.ticklabel_format(useOffset=False,style='plain',axis='y')
    else:
        ref=ref[:,:3]
        ax.plot(*ref[:,:3 if mode=='3d' else 2].T,color='black',ls=(0,(5,4)),lw=1,zorder=8)
        path_limits(ax,allpoints,mode=='3d')


def errors(ax,data,scenario,kind,seeds,settings):
    for name in ORDER:
        group=[data[(scenario,kind,seed,name)] for seed in seeds]
        if all(len(d['t'])==2001 for d in group):
            e=100*np.array([d['error'] for d in group]); t=group[0]['t']
            ax.plot(t,e.mean(0),color=COLORS[name],ls=STYLES[name],lw=1.3,label=TITLES[name])
            ax.fill_between(t,e.min(0),e.max(0),color=COLORS[name],alpha=.08,lw=0)
        else:
            for j,d in enumerate(group):
                ax.plot(d['t'],100*d['error'],color=COLORS[name],ls=STYLES[name],alpha=.6,label=TITLES[name] if j==0 else None)
    if scenario=='force_pulse': ax.axvspan(settings['force_start_s'],settings['force_end_s'],color='#777777',alpha=.1)
    ax.set(xlabel='Time (s)',ylabel='Position error (cm)',xlim=(0,20)); ax.grid(alpha=.2)


def make_figures(out,frame,summary,data,controls,settings):
    directory=out/'figures'; directory.mkdir(exist_ok=True)
    detail=directory/'trajectories'; detail.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    images=[]; seeds=sorted(frame.seed.unique()); representative=seeds[0]
    for scenario in SCENARIOS:
        fig=plt.figure(figsize=(11,8))
        for i,kind in enumerate(KINDS):
            for j,mode in enumerate(['3d','xy']):
                ax=fig.add_subplot(2,2,2*i+j+1,projection='3d' if mode=='3d' else None)
                paths(ax,data,scenario,kind,representative,mode); ax.set_title(f'{kind} | {mode.upper()}',pad=10)
        fig.suptitle(f'{EN[scenario]} | seed {representative}',y=.99)
        add_legend(fig,.95); fig.subplots_adjust(top=.78,bottom=.08,left=.08,right=.92,wspace=.35,hspace=.35)
        fig.text(.5,.01,'Actual physical paths and reference; all MPCs use the same compensation.',ha='center',fontsize=8)
        savefig(fig,directory/f'trajectory_{scenario}',images)
        for kind in KINDS:
            for seed in seeds:
                fig=plt.figure(figsize=(12.5,4.8))
                for j,mode in enumerate(['3d','xy','height']):
                    ax=fig.add_subplot(1,3,j+1,projection='3d' if mode=='3d' else None)
                    paths(ax,data,scenario,kind,seed,mode); ax.set_title(['3D flight path','XY projection','Altitude'][j],pad=10)
                fig.suptitle(f'{EN[scenario]} | {kind} | seed {seed}',y=.99)
                add_legend(fig,.94); fig.subplots_adjust(top=.70,bottom=.17,left=.06,right=.97,wspace=.36)
                fig.text(.5,.01,'Saved physical paths, full planned reference. No smoothing or seed averaging; x marks early termination.',ha='center',fontsize=8)
                savefig(fig,detail/f'{scenario}_{kind}_{seed}',images)
        fig,axes=plt.subplots(1,2,figsize=(10,4.4))
        for ax,kind in zip(axes,KINDS): errors(ax,data,scenario,kind,seeds,settings); ax.set_title(kind)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
        fig.suptitle(EN[scenario],y=.99); fig.subplots_adjust(top=.70,bottom=.18,wspace=.30)
        fig.text(.5,.01,'Five paired states: mean and min-max range if all complete; otherwise individual saved prefixes.',ha='center',fontsize=8)
        savefig(fig,directory/f'error_{scenario}',images)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for ax,kind in zip(axes,KINDS):
        for name in ORDER:
            means=[]; lows=[]; highs=[]
            for scenario in SCENARIOS[:3]:
                g=frame[(frame.scenario==scenario)&(frame.trajectory==kind)&(frame.controller==name)]
                v=g.position_rmse_m*100
                means.append(v.mean() if g.complete.all() else np.nan)
                lows.append(v.min() if g.complete.all() else np.nan); highs.append(v.max() if g.complete.all() else np.nan)
            ax.plot([1,1.5,2],means,color=COLORS[name],ls=STYLES[name],marker='o',ms=4,label=TITLES[name])
            ax.fill_between([1,1.5,2],lows,highs,color=COLORS[name],alpha=.08)
        ax.set(title=kind,xlabel='Physical motor time-constant multiplier',ylabel='Position RMSE (cm)',xticks=[1,1.5,2]); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,frameon=False)
    fig.subplots_adjust(top=.76,bottom=.18,wspace=.30)
    fig.text(.5,.01,'Mean and five-state range; group omitted if any early termination. Controller poles stay nominal; no pure delay.',ha='center',fontsize=8)
    savefig(fig,directory/'motor_tau_robustness',images)
    fig,ax=plt.subplots(figsize=(10,4.5))
    rates=summary.pivot(index='controller',columns='scenario',values='failure_fraction').loc[ORDER,SCENARIOS].to_numpy()*100
    heat=ax.imshow(rates,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    for i,name in enumerate(ORDER):
        for j,scenario in enumerate(SCENARIOS):
            r=summary[(summary.scenario==scenario)&(summary.controller==name)].iloc[0]
            ax.text(j,i,f'{int(r.cases-r.successes)}/{int(r.cases)}\n{rates[i,j]:.0f}%',
                    ha='center',va='center',color='white' if rates[i,j]>50 else 'black',fontsize=10)
    ax.set_xticks(range(5),['Nominal','Tau x1.5','Tau x2','Pose noise','Force pulse'])
    ax.set_yticks(range(5),[TITLES[n] for n in ORDER])
    ax.set_title('Tracking failures / planned cases | five paired states per trajectory',pad=12)
    ax.set_xticks(np.arange(-.5,5,1),minor=True); ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2); ax.tick_params(which='minor',bottom=False,left=False)
    fig.colorbar(heat,ax=ax,label='Tracking failure (%)',ticks=[0,50,100])
    fig.tight_layout(); savefig(fig,directory/'failure_rates',images)
    for kind in KINDS:
        fig,axes=plt.subplots(3,1,figsize=(9,7.3),sharex=True)
        errors(axes[0],data,'force_pulse',kind,seeds,settings)
        for name in ORDER:
            d=data[('force_pulse',kind,representative,name)]
            axes[1].plot(d['t'][:-1],np.linalg.norm(d['command']-np.array(json.loads((out/'plant_parameters.json').read_text())['hover_command']),axis=1),
                         color=COLORS[name],ls=STYLES[name],lw=1.1,label=TITLES[name])
        force=data[('force_pulse',kind,representative,'pd')]
        for j,label in enumerate(['World X','World Y','World Z']): axes[2].step(force['t'][:-1],force['world_acceleration'][:,j],where='post',label=label)
        axes[1].set_ylabel('Command distance\nfrom hover (code units)'); axes[2].set_ylabel('Injected acceleration\n(m/s²)'); axes[2].legend(ncol=3,frameon=False)
        for ax in axes[1:]: ax.grid(alpha=.2); ax.set_xlim(0,20)
        axes[2].set_xlabel('Time (s)'); axes[0].set_xlabel('')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
        fig.suptitle(f'Force pulse response | {kind} | command trace seed {representative}',y=.995)
        fig.subplots_adjust(top=.82,bottom=.09,left=.13,hspace=.26)
        savefig(fig,directory/f'force_response_{kind}',images)
    for kind in KINDS:
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
        for name in ORDER[1:]:
            d=data[('pose_noise',kind,representative,name)]
            axes[0].plot(d['t'][1:],np.cumsum(~d['accepted'])/np.arange(1,len(d['accepted'])+1)*100,color=COLORS[name],ls=STYLES[name],label=TITLES[name])
            axes[1].plot(d['t'],np.linalg.norm(d['bias_estimate'][:,:3],axis=1),color=COLORS[name],ls=STYLES[name])
        axes[0].set_ylabel('Cumulative rejected residuals (%)'); axes[1].set(xlabel='Time (s)',ylabel='Estimated force-bias norm (m/s²)')
        axes[0].legend(ncol=2,frameon=False)
        for ax in axes: ax.grid(alpha=.2); ax.set_xlim(0,20)
        fig.suptitle(f'Noisy-pose residual filter | {kind} | seed {representative}')
        fig.tight_layout(); savefig(fig,directory/f'noise_observer_{kind}',images)
    return images


def fmt(value,scale=1):
    return '缺失（有提前终止或未恢复）' if pd.isna(value) else f'{scale*value:.3f}'


def main(out):
    out=out.resolve(); verification=json.loads((out/'verification.json').read_text())
    assert verification['status']=='PASS' and verification['tracking_sha256']==checksum(out/'robustness_per_run.csv')
    frame=pd.read_csv(out/'robustness_per_run.csv'); assert len(frame)==250
    settings=json.loads((out/'config.json').read_text()); cfg=json.loads((out/'e3_config.json').read_text())
    data,controls=load(out,frame)
    summary=aggregate(frame,controls,['scenario','controller'])
    detail=aggregate(frame,controls,['scenario','trajectory','controller'])
    comparisons,degradation=paired(frame)
    for name,df in [('robustness_summary',summary),('robustness_by_trajectory',detail),('paired_comparisons',comparisons),('paired_nominal_degradation',degradation)]:
        df.to_csv(out/(name+'.csv'),index=False)
    matrix=summary.pivot(index='scenario',columns='controller',values='position_rmse_m').loc[SCENARIOS,ORDER]
    matrix.to_csv(out/'rmse_matrix_m.csv')
    images=make_figures(out,frame,summary,data,controls,settings)
    lines=['| 条件 | '+' | '.join(CN[n] for n in ORDER)+' |','| --- | '+' | '.join(['---']*5)+' |']
    for scenario in SCENARIOS:
        cells=[]
        for name in ORDER:
            r=summary[(summary.scenario==scenario)&(summary.controller==name)].iloc[0]
            cells.append(fmt(matrix.loc[scenario,name],100) if r.completed==r.cases else f'—（{r.completed}/{r.cases} 完成）')
        lines.append('| '+ZH[scenario]+' | '+' | '.join(cells)+' |')
    comparisons_text=[]
    for scenario in SCENARIOS:
        g=summary[summary.scenario==scenario].set_index('controller')
        if g.completed.eq(g.cases).all():
            ranking=g.position_rmse_m.sort_values().index
            relative=100*(1-g.loc['sindy_history','position_rmse_m']/g.loc['linear_history','position_rmse_m'])
            comparisons_text.append(f'- {ZH[scenario]}：最低均值为 {CN[ranking[0]]}；历史模型相对线性历史的 RMSE 改善率 {relative:.2f}%（负值表示更差）。')
        else: comparisons_text.append(f'- {ZH[scenario]}：存在提前终止，主表保留缺失，不用成功子集均值宣布总体胜者。')
    diagnostics=['| 条件 | 方法 | 完成/总数 | 跟踪成功 | 最坏误差 (cm) | 回退 | 拒绝率 (%) | 完整周期 p95 (ms) |',
                 '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for scenario in SCENARIOS:
        for name in ORDER:
            r=summary[(summary.scenario==scenario)&(summary.controller==name)].iloc[0]
            diagnostics.append(f'| {ZH[scenario]} | {CN[name]} | {r.completed}/{r.cases} | {r.successes} | {100*r.worst_position_error_m:.3f} | {r.fallbacks} | {100*r.observer_rejected_fraction:.2f} | {1000*r.full_control_period_s_p95:.2f} |')
    recovery=['| 方法 | 8–12 s 峰值均值 (cm) | 14–20 s RMSE (cm) | 恢复例数 | 恢复时间均值 (s) |',
              '| --- | --- | --- | --- | --- |']
    for name in ORDER:
        r=summary[(summary.scenario=='force_pulse')&(summary.controller==name)].iloc[0]
        recovery.append(f'| {CN[name]} | {fmt(r.event_peak_error_m,100)} | {fmt(r.post_event_rmse_m,100)} | {r.recovered_cases}/{r.cases} | {fmt(r.recovery_time_s)} |')
    text=f'''# E4 执行器不匹配与扰动实验结果

## Material Passport

- 正式运行：250 例，完整完成 {int(frame.complete.sum())}，通过冻结跟踪门槛 {int(frame.tracking_success.sum())}；独立核验 PASS。
- 场景：名义、电机时间常数 ×1.5/×2、位姿噪声、外力脉冲；圆/八字 × 五个配对初态 × 五种控制器。
- 开发与正式分离；所有 E3 控制模型、平衡点校准、代价、约束与在线偏差滤波器保持冻结。
- 开发 25 例中 20 例完整完成、5 例提前终止；环境与数据校验通过后，按原协议保留强度进入正式批次，详见 `docs/PELICAN_E4_DEVELOPMENT.md`。
- 数据来源：Pelican 日志辨识的模型，闭环为名义有效刚体仿真；没有新增实机飞行。
- 设计与定义：`docs/PELICAN_E4_PROTOCOL.md`；原 E1–E3 结果及正式名义比较保持封存。

## 主结果：位置 RMSE（cm，越小越好）

每个单元格是同一条件下 2 条轨迹 × 5 个初态的十个完整算例等权平均。四种 MPC 都带同等校准和在线补偿。
PD 使用名义对象知识，作为经典参考。每个初态确有随机状态变化，噪声按初态配对共享。

{chr(10).join(lines)}

{chr(10).join(comparisons_text)}

这些是描述性结果，不把相关轨迹/时间点当独立物理对象，不作显著性声明。各场景单列，不以任意场景权重合成一个总排名。
`robustness_by_trajectory.csv` 保存分轨迹结果，`paired_comparisons.csv` 保存历史模型对每个基线的配对差值。
`paired_nominal_degradation.csv` 保存同一轨迹/初态/控制器相对本轮名义参照的退化，不能将旧 E3 的不同初态均值当配对参照。

## 运行情况与计算代价

沿用 E3 的成功门槛：完整运行 20 s、位置 RMSE 不超过 {cfg['tracking_success_rmse_m']} m、最大位置误差不超过 {cfg['tracking_success_max_error_m']} m，并且输入幅值/变化率无违规。
成功率需要与厘米单位的误差主表一起阅读。

{chr(10).join(diagnostics)}

输入幅值违规 {int(frame.input_bound_violations.sum())} 次，变化率违规 {int(frame.input_slew_violations.sum())} 次；
求解失败 {int(frame.solver_failures.sum())} 次，PD 回退 {int(frame.fallbacks.sum())} 次，估计截断 {int(frame.observer_clipped.sum())} 次。
拒绝率指六通道残差门限导致整次观测器更新被拒绝，PD 不使用该观测器。拒绝不等于安全性或准确性保证。
完整周期包含历史打包、控制计算和因果观测/偏差更新，计时明细含 p99、最大值和超过 50 ms 比例，见汇总 CSV。
物理积分和噪声采集模拟不计入算法时间；位姿差分处理计入。仿真未注入计算超时造成的执行延迟，不能证明硬实时。

## 外力脉冲与恢复

世界系等效加速度 [0.4, -0.25, 0.15] m/s² 仅在 [8,12) s 加入对象，控制器不知道其幅值和起止时间。
恢复定义为撤力后首次连续 1 s 的三维位置误差不超过 3 cm。未恢复计缺失，不用观察到的短暂过阈值冒充恢复。

{chr(10).join(recovery)}

## 噪声与模型解释边界

噪声为 100 Hz 原始位姿的独立高斯扰动：位置每轴 1 mm，姿态旋转向量每轴 0.001 rad。
速度/角速度仍通过后向差分得到，未新增状态滤波器；因此本轮同时暴露既有差分观测和残差门限在噪声下的行为。
不能把结果解释为某种实测传感器、滤波器或任意风场的表现。观测器估计是总模型残差，不等于注入外力真值。
时间常数 ×1.5/×2 只改变隐藏物理电机状态演化，控制器极点和标定均不更新；没有混入纯输入延迟或同时改变质量。
轨迹速度沿用 E3 的 0.45 rad/s 和原变化率限制，不能将小幅退化外推为任意高动态下的执行器鲁棒性。

## 独立校验与可复现性

- 五项实现测试通过，包括五种控制器零干预轨迹与 E3 一致、解析电机响应、独立物理积分、噪声几何/配对/因果性、恢复判据。
- 重建 {verification['observed_transitions']:,} 个有噪/无噪观测；独立 NumPy 重算 {verification['residual_rows']:,} 个残差，最大差 {verification['max_residual_replay_error']:.3g}。
- 核对 {verification['causal_bias_checks']:,} 次 MPC 估计使用、{verification['timing_checks']:,} 次完整周期计时和 {verification['pd_command_checks']:,} 次 PD/回退命令。
- 自适应积分抽查 {verification['independent_plant_steps']:,} 个物理转移，包括扰动施加/撤除边界；最大状态分量差 {verification['max_adaptive_state_error']:.3g}，均满足逐分量容差。
- 独立检查环境噪声种子、初态配对、约束、全部指标与汇总一致性；所有旧封存文件通过哈希检查。

## 文件入口

- `figures/trajectory_*.png/.pdf`：每个条件的圆/八字三维与 XY 总览，固定种子 15301。
- `figures/trajectories/`：全部 50 个条件/轨迹/初态的路径和高度图，使用真实保存的物理位置，不平滑、不跨种子平均。
- `figures/error_*.png/.pdf`：五个初态均值及最小–最大范围；有提前终止时显示各例原始前缀，范围不是置信区间。
- `figures/motor_tau_robustness.png/.pdf`、`failure_rates.png/.pdf`：时间常数与误差、各条件失败率。
- `figures/force_response_*.png/.pdf`、`noise_observer_*.png/.pdf`：外力响应、控制指令幅度与有噪观测器诊断。
- `robustness_per_run.csv`、`robustness_summary.csv`、`robustness_by_trajectory.csv`、`rmse_matrix_m.csv`：全部逐例与汇总结果，CSV 内长度单位为 m。
- `cases/*/trajectory.npz`、`control.csv`、`metrics.json`：完整原始记录；`pose_noise` 包含预热及整个计划试验的配对噪声，`world_acceleration` 为实际施加值。
- `FROZEN.json`、`verification.json`、`artifact_hashes.json`：实验冻结、独立核验、全部结果和脚本指纹。
'''
    report=ROOT/'docs/PELICAN_E4_RESULTS.md'; report.write_text(text,encoding='utf-8'); (out/'REPORT.md').write_text(text,encoding='utf-8')
    check_preserved()
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='artifact_hashes.json']
    files += [Path(__file__),ROOT/'scripts/verify_pelican_e4.py',ROOT/'scripts/run_pelican_e4.py',ROOT/'configs/pelican_e4.json',
              ROOT/'tests/test_pelican_e4.py',ROOT/'docs/PELICAN_E4_PROTOCOL.md',ROOT/'docs/PELICAN_E4_DEVELOPMENT.md',report,*list((ROOT/'pelican_e4').glob('*.py'))]
    write_json(out/'artifact_hashes.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        files={p.relative_to(ROOT).as_posix():dict(sha256=checksum(p),bytes=p.stat().st_size) for p in sorted(set(files))}))
    print((matrix*100).to_string()); print('Report:',report,'Figure pairs:',len(images)//2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args().run_dir)
