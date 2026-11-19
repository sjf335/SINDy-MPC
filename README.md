# SINDy-MPC

Method implementation for *SINDy-MPC with History Augmentation and Bias Compensation for Quadrotor Trajectory Tracking* by Junfeng Shen, Huihui Li, Biao Chen, and Jing Wang.

## Train the predictor

Use Python 3.11 and the original AscTec Pelican flight MAT file from [WAVELab](https://github.com/wavelab/pelican_dataset).

```sh
python -m pip install -r requirements.txt
python train.py path/to/AscTec_Pelican_Flight_Dataset.mat --out fitted_history_model.json
```

`train.py` reproduces the 32/11/11 whole-flight split, fits the 18 history-model candidates, and selects the configuration from validation flights. The output JSON contains the fitted coefficients, feature scales, selected parameters and flight IDs. `dynamics.py` constructs the states and body-frame targets from pose and motor commands. `identification.py` builds the lagged feature library, fits the sparse regression, and scores multistep predictions.

## Use the controller

`controller.py` implements the hover correction, residual observer, constrained MPC and bounded fallback. The caller supplies the vehicle's hover command, motor command limits, and parameters for fallback allocation. The paper lists the MPC cost weights, solver settings, and observer limits in Table I and Section III-B.

Load the fitted model into NumPy arrays, apply `calibrate_equilibrium(model, cfg, parameters)`, and construct `BiasMPC(corrected_model, parameters, cfg)`. Initialize `BiasObserver(mpc.transition, cfg['dt'], observer_settings)`. At each 0.05 s control tick, call `control_update` with the measured history, applied command history and the ten end-of-interval reference states. At each 0.01 s observation step, call `observer.update` with the packed state at the start of the interval, the command actually applied, and the newly observed state. Quaternions follow the `xyzw` convention.
