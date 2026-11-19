# SINDy-MPC

Research code for **SINDy-MPC with History Augmentation and Bias Compensation for Quadrotor Trajectory Tracking** by Junfeng Shen, Huihui Li, Biao Chen, and Jing Wang (MEAE 2026).

The controller uses lagged motion and motor-command differences to predict body-frame accelerations. A hover calibration and a filtered residual estimate correct the predictions used by MPC.

## Source code

- [Model identification and hyperparameter search](method/scripts/run_pelican_e2.py)
- [Sparse feature library](method/pelican_e2/model.py) and [flight-data preparation](method/pelican_e2/data.py)
- [Controller, equilibrium calibration and bias observer](method/pelican_e3_corrected/control.py)
- [Component ablation](method/scripts/run_component_ablation.py), [noise and disturbance tests](method/scripts/run_pelican_noise_pulse.py), and [timing measurements](method/scripts/profile_pelican_realtime.py)
- [Paired statistical analysis](method/scripts/reproduce_paper_statistics.py), [model replay](method/scripts/replay_paper_models.py), and [figure generation](latex/scripts/)

## Run the code

The project uses Python 3.11. Install the dependencies from `method/requirements.txt`. The statistical and replay scripts expect the archived model and experiment inputs from the paper's accompanying reproducibility supplement. Extract that supplement under `method/` before running those entry points. The data provider distributes the original flight logs through the [WAVELab Pelican repository](https://github.com/wavelab/pelican_dataset).
