"""Generate post-hoc E3 diagnostic figures and an evidence-bounded Chinese report."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from run_pelican_e2 import checksum,write_json

OUT=ROOT/'runs/pelican_e3_diagnostic_v1'
ORDER=['sindy_current','linear_history','sindy_history','motor_sindy']
LABEL={'sindy_current':'Current SINDy','linear_history':'Linear history','sindy_history':'History SINDy','motor_sindy':'Motor + SINDy'}
ZH={'sindy_current':'无历史 SINDy-MPC','linear_history':'线性历史 MPC','sindy_history':'历史 SINDy-MPC','motor_sindy':'电机 + SINDy-MPC'}
COLOR={'sindy_current':'#D55E00','linear_history':'#0072B2','sindy_history':'#009E73','motor_sindy':'#CC79A7'}
STYLE={'sindy_current':'--','linear_history':'-.','sindy_history':'-','motor_sindy':':'}
MARKER=dict(zip(ORDER,['s','^','o','D']))


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight')
    with Image.open(folder/(name+'.png')) as f:f.convert('L').save(folder/(name+'_grayscale.png'),dpi=(300,300))
    plt.close(fig)


def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='PASS' and verification['scope']=='all'
    prediction=pd.read_csv(OUT/'replay_summary.csv');bias=pd.read_csv(OUT/'late_error_decomposition.csv')
    sens=pd.read_csv(OUT/'input_sensitivity_summary.csv')
    cases=pd.concat([pd.read_csv(OUT/'intervention/per_run.csv'),pd.read_csv(OUT/'bias_components/per_run.csv')],ignore_index=True)
    cases.to_csv(OUT/'all_intervention_cases.csv',index=False)
    summaries=[];pairs=[];late=[]
    for (name,variant),g in cases.groupby(['controller','variant']):
        summaries.append(dict(controller=name,variant=variant,cases=len(g),completed=int(g.complete.sum()),
                              tracking_successes=int(g.tracking_success.sum()),mean_rmse_m=g.position_rmse_m.mean(),
                              min_rmse_m=g.position_rmse_m.min(),max_rmse_m=g.position_rmse_m.max(),fallbacks=int(g.fallbacks.sum()),
                              bound_violations=int(g.input_bound_violations.sum()),slew_violations=int(g.input_slew_violations.sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'intervention_summary.csv',index=False)
    for name in ORDER:
        g=cases[cases.controller==name].pivot(index=['trajectory','seed'],columns='variant',values='position_rmse_m')
        delta=g.original-g.hover_corrected
        pairs.append(dict(controller=name,pairs=len(g),original_m=g.original.mean(),corrected_m=g.hover_corrected.mean(),
                          difference_m=delta.mean(),relative_reduction_percent=100*delta.mean()/g.original.mean(),
                          improved_cases=int((delta>0).sum()),min_difference_m=delta.min(),max_difference_m=delta.max()))
    pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'paired_intervention_differences.csv',index=False)
    for row in cases.itertuples(index=False):
        parent='intervention' if row.variant in ['original','hover_corrected'] else 'bias_components'
        with np.load(OUT/parent/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}_{row.variant}'/'trajectory.npz') as f:
            mask=f['t']>=10;error=f['physical'][mask,:3]-f['reference'][mask,:3]
        m=error.mean(0)
        late.append(dict(controller=row.controller,variant=row.variant,trajectory=row.trajectory,seed=row.seed,
                         late_rmse_m=np.sqrt(np.mean(np.sum(error**2,axis=1))),late_bias_norm_m=np.linalg.norm(m)))
    late=pd.DataFrame(late);late.to_csv(OUT/'intervention_late_errors.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.6))
    for name in ORDER:
        f=prediction[(prediction.source_controller=='all')&(prediction.model==name)].sort_values('horizon_s')
        axes[0].plot(f.horizon_s,f.velocity_rmse_m_s,ls=STYLE[name],marker=MARKER[name],color=COLOR[name],label=LABEL[name])
    axes[0].set(xscale='log',yscale='log',xlabel='Prediction horizon (s)',ylabel='Velocity endpoint RMSE (m/s)',title='(a) Same recorded simulation windows')
    for j,name in enumerate(ORDER):
        b=bias[bias.controller==name]
        axes[1].bar(j,100*b.bias_fraction_mse.mean(),color=COLOR[name],alpha=.75)
        axes[1].scatter(j+np.linspace(-.15,.15,len(b)),100*b.bias_fraction_mse,s=8,marker=MARKER[name],color='black')
    axes[1].set_xticks(range(4),['Current','Linear\nhistory','History\nSINDy','Motor'])
    axes[1].set(ylabel='Constant-bias fraction of late MSE (%)',ylim=(0,105),title='(b) Original closed-loop errors, 10–20 s')
    for ax in axes:ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.015),frameon=False)
    fig.suptitle('POST-HOC SIMULATION DIAGNOSIS | frozen E3 records',fontsize=11)
    fig.tight_layout(rect=(0,.06,1,.95));save(fig,'prediction_and_bias')

    fig,axes=plt.subplots(1,2,figsize=(8.2,3.5),sharex=True)
    for name in ORDER:
        f=sens[(sens.region=='trajectory')&(sens.model==name)&(sens.amplitude==.25)].sort_values('horizon_s')
        for ax,field in zip(axes,['v_relative_error','w_relative_error']):
            ax.plot(f.horizon_s,f[field],color=COLOR[name],ls=STYLE[name],marker=MARKER[name],label=LABEL[name])
    for ax,title in zip(axes,['(a) Velocity input response','(b) Angular-rate input response']):
        ax.set(xscale='log',yscale='log',xlabel='Step-response horizon (s)',ylabel='Relative sensitivity error',title=title)
        ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.015),frameon=False)
    fig.suptitle('POST-HOC SIMULATION DIAGNOSIS | symmetric ±0.25 command steps',fontsize=11)
    fig.tight_layout(rect=(0,.06,1,.95));save(fig,'input_response')

    fig,axes=plt.subplots(1,2,figsize=(8.4,3.9))
    for j,name in enumerate(ORDER):
        f=cases[(cases.controller==name)&cases.variant.isin(['original','hover_corrected'])]
        pivot=f.pivot(index=['trajectory','seed'],columns='variant',values='position_rmse_m')
        for k,row in enumerate(pivot.itertuples()):
            shift=(k-1.5)*.025
            axes[0].plot([j-.18+shift,j+.18+shift],[row.original,row.hover_corrected],color=COLOR[name],lw=.8,alpha=.65)
            axes[0].scatter(j-.18+shift,row.original,marker=MARKER[name],facecolors='none',edgecolors=COLOR[name],s=28)
            axes[0].scatter(j+.18+shift,row.hover_corrected,marker=MARKER[name],color=COLOR[name],s=28)
    axes[0].set_xticks(range(4),['Current','Linear\nhistory','History\nSINDy','Motor'])
    axes[0].set(ylabel='20 s position RMSE (m)',title='(a) Original → all six biases corrected',ylim=(0,None))
    variants=['original','force_only','angular_only','hover_corrected']
    for name,offset in [('sindy_history',-.12),('motor_sindy',.12)]:
        f=summary[(summary.controller==name)].set_index('variant').loc[variants]
        axes[1].plot(np.arange(4)+offset,f.mean_rmse_m,color=COLOR[name],marker=MARKER[name],ls=STYLE[name],label=LABEL[name])
        for j,var in enumerate(variants):
            values=cases[(cases.controller==name)&(cases.variant==var)].position_rmse_m.to_numpy()
            axes[1].scatter(j+offset+np.linspace(-.035,.035,len(values)),values,s=12,color=COLOR[name],alpha=.6)
    axes[1].set_xticks(range(4),['None','Force\nonly','Angular\nonly','Both'])
    axes[1].set(ylabel='20 s position RMSE (m)',xlabel='Intercept correction applied',title='(b) Adaptive component diagnosis',ylim=(0,None))
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(alpha=.15,axis='y')
    fig.suptitle('POST-HOC SIMULATION DIAGNOSIS | 2 trajectories × 2 paired seeds',fontsize=11)
    fig.text(.5,-.015,'Calibration uses known simulated hover; no real-flight or general offset-free guarantee.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,'bias_interventions')

    global_prediction=prediction[prediction.source_controller=='all']
    hist_bias=bias[bias.controller=='sindy_history']
    lines=['# Pelican E3：预测与控制差异诊断','',
           '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent；scientific-visualization。',
           '- Origin Mode: run + validate。',
           '- Origin Date: 2026-10-26。',
           '- Verification Status: ANALYZED；记录复算及独立数值路径复核通过，未完整重跑所有闭环。',
           '- Version Label: pelican_e3_diagnostic_v1。',
           '- 证据类型：E3 之后的探索性仿真诊断。原 E3 冻结结果未覆盖；不能作为新的独立确认性测试或实机证据。','',
           '## 1. 诊断结论与范围','',
           '本轮支持将**学习模型与仿真对象的平衡点不一致，以及控制器未补偿这种持续偏差**列为原 E3 跟踪误差的主要可干预来源。'
           '把全部模型在同一个仿真悬停点的六维输出偏差消除后，历史模型的跟踪误差下降；模型结构、其余系数、代价和约束没有改变。'
           '这一结论限于冻结仿真器及本次配对场景，不等同于已识别真实 Pelican 的物理故障，也不证明历史结构普遍优于电机结构。','',
           '初始方案在本轮计算前保存于 [诊断协议](PELICAN_E3_DIAGNOSTIC_PROTOCOL.md)。固定偏差干预出现改善后追加力/角加速度分量检查，'
           '追加选择及配置另存 `bias_components/ADAPTIVE_PROTOCOL.json`。所有结果均保留，没有按结果删案例或重调正式 E3。','',
           '## 2. 同一批仿真轨迹上的预测','',
           '读取原 E3 的全部 75 个案例，各取 37 个共同起点，共 2,775 个窗口；四个模型均输入同一历史和同一已执行未来命令。'
           '每个案例先计算窗口端点向量 RMSE，再对 75 例等权平均。窗口并非独立实验。','',
           '| 模型 | 0.01 s 速度 RMSE (m/s) | 0.1 s | 0.5 s | 1 s |',
           '|---|---:|---:|---:|---:|']
    for name in ORDER:
        f=global_prediction[global_prediction.model==name].set_index('horizon_s')
        lines.append('| '+ZH[name]+' | '+' | '.join(f'{f.loc[t,"velocity_rmse_m_s"]:.5f}' for t in [.01,.1,.5,1.])+' |')
    lines+=['','原真实日志 E2 中历史模型在 0.5 s 的速度预测领先，但在当前仿真轨迹上，其 0.5 s 速度误差已落后于其余三个学习模型。'
            '历史模型在较短时域仍准确；不能将它概括成所有预测能力都失效。分原控制器、分时域及失败数见 `replay_summary.csv`。'
            '这一结果显示真实日志上的相对预测优势没有迁移到当前仿真分布；两个数据域的绝对 RMSE 不能直接比较来计算迁移损失。'
            '尚不能单独确定噪声、滤波、状态覆盖或某个物理项各自的作用。','',
            '## 3. 原 E3 的误差主要是什么形态','',
            '后 10 s 分解满足 `mean(||e||²) = ||mean(e)||² + mean(||e-mean(e)||²)`。','',
            '| 控制器 | 后 10 s RMSE (m) | 平均偏移模长 (m) | 去均值 RMSE (m) | 偏移占 MSE 比例 |',
            '|---|---:|---:|---:|---:|']
    for name in ORDER:
        f=bias[bias.controller==name].mean(numeric_only=True)
        lines.append(f'| {ZH[name]} | {f.late_rmse_m:.5f} | {f.late_bias_norm_m:.5f} | {f.late_demeaned_rmse_m:.5f} | {100*f.bias_fraction_mse:.2f}% |')
    lines+=['',f'历史模型平均偏移约为 ({hist_bias.late_bias_x_m.mean():.5f}, {hist_bias.late_bias_y_m.mean():.5f}, {hist_bias.late_bias_z_m.mean():.5f}) m。'
            '误差形态主要是稳定偏离，而非求解失败或轨迹发散。仅靠这项分解还不能确定成因，因而进行了以下干预。','',
            '## 4. 悬停平衡点干预','',
            '在物理仿真对象的悬停平衡点读取每个学习模型预测的净加速度与角加速度，记为六维 `b`。'
            '仅把回归输出反标准化的截距改为 `y_mu_new = y_mu_old - b`；在水平悬停点，世界系与机体系一致。'
            '六个预测输出因此与同一物理平衡点一致。修改未触及候选库、稀疏系数、历史长度、MPC 权重或指令约束。'
            '本轮利用已知仿真平衡点，是带对象知识的诊断性校准；并非在线扰动观测器，也没有任意轨迹零偏差保证。','',
            '圆形与八字各 20 s，新种子 9301/9302。下表每格为四个配对案例的均值，原版也在相同新条件重新运行。','',
            '| 控制器 | 原版 RMSE (m) | 六维偏差修正后 (m) | 相对下降 | 改善案例 / 配对案例 |',
            '|---|---:|---:|---:|---:|']
    for r in pairs.itertuples(index=False):
        lines.append(f'| {ZH[r.controller]} | {r.original_m:.5f} | {r.corrected_m:.5f} | {r.relative_reduction_percent:.2f}% | {r.improved_cases}/{r.pairs} |')
    lines+=['','这一干预比单纯相关分析更直接：在保持其余设置不变的条件下，平衡点校准改变了跟踪结果。'
            '但当前测试只有两种轨迹、两个初始条件种子和一个仿真对象；不报告 p 值或一般化的显著性结论。'
            '无历史模型的剩余误差也提醒我们，常数偏差修正并不解决所有动态误差。','',
            '## 5. 力与角加速度分量的追加干预','',
            '以下是看到首组全量修正收益后追加的探索性分量检查。原版与全修正复用上节结果；两个分量各新增四个配对场景。','',
            '| 模型 | 不修正 (m) | 只修正比力 (m) | 只修正角加速度 (m) | 两者都修正 (m) |',
            '|---|---:|---:|---:|---:|']
    for name in ['sindy_history','motor_sindy']:
        f=summary[summary.controller==name].set_index('variant')
        lines.append('| '+ZH[name]+' | '+' | '.join(f'{f.loc[v,"mean_rmse_m"]:.5f}' for v in variants)+' |')
    lines+=['','历史模型仅修正比力已降低误差，仅修正角加速度反而增加误差；两者联合修正后最低。'
            '因此主要定位是整体平衡点不一致，不能简单归咎于角加速度通道或某一个电机。'
            '只有联合修正才能代表全部六维平衡点一致；单独修正会与其余未修正通道相互作用。'
            '不能把联合下降百分比拆成独立、可相加的两个因果贡献。','',
            '## 6. 电机指令响应是否完全正确','',
            '对精确悬停及 75 个预先固定轨迹状态，分别扰动四个电机，采用正负 0.25/1 个编码单位的恒定阶跃。'
            '中心差分扣除了各自无扰动轨迹的共同漂移；报告多个幅值的一致性。下表为轨迹状态上 0.5 s、幅值 0.25 的汇总，'
            '相对误差定义为所有测试状态/电机的预测敏感度与物理敏感度之差的 Frobenius 范数，除以物理敏感度范数。','',
            '| 模型 | 位置响应相对误差 | 速度响应相对误差 | 角速度响应相对误差 | 角速度响应增益比 |',
            '|---|---:|---:|---:|---:|']
    for name in ORDER:
        r=sens[(sens.region=='trajectory')&(sens.model==name)&(sens.horizon_s==.5)&(sens.amplitude==.25)].iloc[0]
        lines.append(f'| {ZH[name]} | {100*r.p_relative_error:.2f}% | {100*r.v_relative_error:.2f}% | {100*r.w_relative_error:.2f}% | {r.w_gain_ratio:.3f} |')
    lines+=['','历史模型的 0.5 s 角速度响应幅度偏低，但位置响应敏感度比电机模型更接近物理对象。'
            '因此不能说历史模型在所有输入响应上都更差；这些未修正模型的局部响应误差，也不能单独解释原闭环排名。'
            '短至 0.01 s 时物理阶跃响应很小，相对误差可能很大，应与绝对敏感度文件一起读。'
            '两种扰动幅值下各分量整体相对差最大约 '
            f'{100*verification["max_amplitude_relative_difference"]:.4f}%，未观察到主要结论由差分步长选择造成。','',
            '## 7. 完成量、复核与未解决问题','',
            f'新增闭环 {len(cases)} 例，完整运行 {int(cases.complete.sum())} 例；求解回退 {int(cases.fallbacks.sum())} 次，'
            f'命令幅值/变化率违反分别为 {int(cases.input_bound_violations.sum())}/{int(cases.input_slew_violations.sum())} 次。'
            f'保存并复算 {verification["replay_endpoints_recomputed"]:,} 行预测端点指标、{verification["finite_difference_rows_recomputed"]:,} 行敏感度。'
            f'另用符号模型复算 {verification["symbolic_windows_checked"]} 个窗口，最大差 {verification["max_symbolic_difference"]:.3e}；'
            f'用独立连续方程/自适应积分器检查 {verification["adaptive_branch_trajectories"]} 条 0.5 s 响应和 '
            f'{verification["intervention_adaptive_transitions"]} 个新增闭环转移。原 E3 已封存的 {verification["e3_sealed_files_unchanged"]} 个文件指纹不变。','',
            '仍未解决：真实飞行中的平衡点/偏差如何在线估计；任意轨迹与执行器失配下校准是否有效；仿真对象与电机模型共享参数的贡献；'
            '历史特征中滤波与原控制器关联的贡献；共同代价与预测时域是否适合其他场景。当前诊断没有替代 E4 鲁棒性或实机验证。','',
            '下一版优先考虑用训练/验证数据建立合理平衡点约束，或加入因果在线偏差估计，再冻结新的开发配置。'
            '应保留四种方法的同等补偿对照，不能只给历史方法使用仿真真值后宣称方法优越。','',
            '## 8. 统计解释核查（11/11）','',
            '1. 聚合反转：预测按原控制器分层，闭环逐轨迹/种子保存。',
            '2. 生态推断：均值不代表每个窗口，报告配对改善数量。',
            '3. 选择偏差：范围限于当前仿真对象、平滑参考及已定义初始条件。',
            '4. 条件选择：不按误差或成功程度排除场景。',
            '5. 分母：完整运行、失败、回退和案例总数明确。',
            '6. 均值回归：所有方法接受同样校准，原版在相同新初始条件重新运行。',
            '7. 幸存者偏差：失败前缀与完整 RMSE 分开保存。',
            '8. 多处寻找：所有时域、方法和分量干预均保存，不报告选择性显著性。',
            '9. 分析路径：本轮事后诊断和追加分量检查明确标记，不冒充预注册确认。',
            '10. 因果范围：平衡点干预支持当前仿真设置下的影响，不能推断实机物理机制。',
            '11. 反向因果：日志预测与主动输入干预分开，不把原控制器相关性当新控制效果。','',
            '## 9. 文件与复现','',
            '- [预测迁移与误差分解图](../runs/pelican_e3_diagnostic_v1/figures/prediction_and_bias.png)',
            '- [电机输入响应图](../runs/pelican_e3_diagnostic_v1/figures/input_response.png)',
            '- [偏差干预与分量对比图](../runs/pelican_e3_diagnostic_v1/figures/bias_interventions.png)',
            '- [全部新增闭环案例](../runs/pelican_e3_diagnostic_v1/all_intervention_cases.csv)',
            '- [干预汇总](../runs/pelican_e3_diagnostic_v1/intervention_summary.csv)',
            '- [独立核查记录](../runs/pelican_e3_diagnostic_v1/verification.json)','',
            '在 `huiyi/method` 下执行：','',
            '```powershell',
            '.\\.venv\\Scripts\\python.exe scripts/verify_pelican_e3_diagnostics.py',
            '.\\.venv\\Scripts\\python.exe scripts/report_pelican_e3_diagnostics.py',
            '.\\.venv\\Scripts\\python.exe scripts/verify_pelican_e3_diagnostic_report.py',
            '```','',
            '完整重新执行需给 `diagnose_pelican_e3.py --stage analyze` 指定新的 `--run-dir`，随后同目录执行 `--stage intervene` 与 '
            '`diagnose_pelican_e3_bias_components.py --run-dir ...`。各阶段拒绝覆盖已有结果。报告脚本默认读取本版结果；再生成后文件哈希应重新记录。']
    doc=ROOT/'docs/PELICAN_E3_DIAGNOSTICS.md';doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),interventions=summaries,paired_changes=pairs.to_dict(orient='records'),
                verification=verification,post_hoc=True,raw_formal_e3_unchanged=True)
    write_json(OUT/'report.json',report)
    print(pairs.to_string(index=False));print(summary.to_string(index=False))


if __name__=='__main__':main()
