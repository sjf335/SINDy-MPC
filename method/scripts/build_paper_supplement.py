"""Build the portable, allow-listed numerical supplement for ME1036."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent/'supplement'/'ME1036-reproducibility.zip'


def main():
    files={}
    def add(relative, name=None):
        path=ROOT/relative
        files[name or path.relative_to(ROOT).as_posix()]=path.read_bytes()
    def group(directory, patterns):
        for pattern in patterns.split():
            targets=sorted((ROOT/directory).glob(pattern))
            if not targets: raise FileNotFoundError(f'{directory}/{pattern}')
            for path in targets:
                add(path.relative_to(ROOT))
    add('docs/PAPER_REPRODUCIBILITY.md','README.md')
    add('requirements.txt')
    group('scripts','reproduce_paper_statistics.py replay_paper_models.py run_pelican_e2.py fetch_pelican.py run_pelican_e3.py run_pelican_e3_nominal_comparison.py')
    for folder in ('pelican_e2','pelican_e3','pelican_e3_corrected'):
        group(folder,'*.py')
    group('configs','pelican_e2.json pelican_e3.json pelican_e3_nominal_comparison.json pelican_e4.json pelican_trajectory_search.json')
    group('runs/pelican_e1_v1','splits.json motor_response_models.json')
    source=json.loads((ROOT/'runs/pelican_e1_v1/raw_provenance.json').read_text())
    files['runs/pelican_e1_v1/raw_provenance.json']=json.dumps({key:source[key] for key in ('source_url','bytes','sha256')},indent=2).encode()
    group('runs/pelican_e2_v1','config.json splits.json environment.json motor_model.json models/*.json models/training_statistics.npz validation/*.csv test/prediction_per_window.csv test/prediction_per_flight.csv test/prediction_summary.csv analysis/coefficients_long.csv analysis/primary_paired_bootstrap.csv analysis/metric_intervals.csv')
    group('runs/pelican_trajectory_search_v1','config.json corrected_models.json equilibrium_calibration.json motor_model.json plant_parameters.json observer.json selected_parameters.csv selected_configs/*.json validate/initial_conditions/*.npz validate/tracking_per_run.csv')
    group('runs/pelican_e4_v1','config.json e3_config.json robustness_per_run.csv robustness_summary.csv robustness_by_trajectory.csv')
    group('runs/pelican_e4_estimation_diagnostic_v1','paired_all_runs.csv diagnostic_summary.csv')
    group('runs/pelican_e4_estimation_filter_sanity_v1','*per_run.csv')
    group('runs/pelican_component_ablation_v1','tracking_per_case.csv prediction_per_flight.csv prediction_per_window.csv.gz prediction_summary.csv ablation_results.csv matched_feature_control/model.json matched_feature_control/prediction_per_flight.csv')
    group('runs/pelican_noise_pulse_v1','noise_per_case.csv pulse_per_case.csv observer_lag_per_case.csv pulse_mean_traces.csv estimation_diagnostic.csv')
    group('runs/pelican_realtime_v1','timing_summary.csv timing_by_trial.csv')
    group('runs/pelican_statistics_v1','paper_contrasts.csv paired_intervals.csv')
    hashes={name:sha256(payload).hexdigest() for name,payload in sorted(files.items())}
    files['SHA256SUMS.json']=json.dumps(hashes,indent=2).encode()
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=DEST.parent,suffix='.zip',delete=False) as tmp:
        temp_path=Path(tmp.name)
    with ZipFile(temp_path,'w',ZIP_DEFLATED) as bundle:
        for name,payload in sorted(files.items()):
            info=ZipInfo(name,date_time=(2026,11,19,0,0,0));info.compress_type=ZIP_DEFLATED
            bundle.writestr(info,payload)
    temp_path.replace(DEST)
    print(DEST,len(files),DEST.stat().st_size)

if __name__=='__main__':main()
