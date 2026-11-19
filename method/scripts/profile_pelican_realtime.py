"""Profile nominal controllers serially, in fresh Windows processes, using the paper's trials."""
import argparse
import ctypes
from ctypes import wintypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import winreg

THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
for variable in THREAD_ENV:
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from pelican_e3_corrected.control import BiasMPC
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_e3_nominal_comparison import run_case

SOURCE = ROOT / 'runs/pelican_trajectory_search_v1'
DEFAULT_OUT = ROOT / 'runs/pelican_realtime_v1'
ORDER = ('pd', 'sindy_current', 'linear_history', 'motor_sindy', 'sindy_history')
KINDS = ('circle', 'figure8', 'helix')
SEED = 24301


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize',
        'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def memory():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
    query.restype = wintypes.BOOL
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_bytes': counters.WorkingSetSize, 'peak_working_set_bytes': counters.PeakWorkingSetSize}


def environment():
    processor_key = r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, processor_key) as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    ram_kib = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(ram_kib)):
        raise OSError('Could not read installed system memory')
    return dict(cpu=cpu, os_build=sys.getwindowsversion().build, logical_cpus=os.cpu_count(),
                installed_ram_gib=ram_kib.value / 2**20, python=platform.python_version(),
                threads={key:os.environ[key] for key in THREAD_ENV},
                versions={key:importlib.metadata.version(key) for key in ('casadi','numpy','scipy','pandas')})


def run_trial(out, controller_name, trajectory):
    case = f'{trajectory}_{SEED}_{controller_name}'
    folder = out / 'cases' / case
    memory_file = out / 'memory' / f'{case}.json'
    cfg = load(SOURCE / 'selected_configs' / f'{trajectory}.json')
    assert cfg['duration_s'] == 20 and cfg['dt'] == .01 and cfg['control_dt'] == .05
    plant = load(SOURCE/'plant_parameters.json')
    motor = load(SOURCE/'motor_model.json')
    settings = dict(observer=load(SOURCE/'observer.json'), stage_runtime_limit_s=3600)
    with np.load(SOURCE/'validate/initial_conditions'/f'{SEED}.npz') as saved:
        initial = dict(saved)
    memory_log = {'before_construction': memory()}
    controller = None
    if controller_name != 'pd':
        model = restore_model(load(SOURCE/'corrected_models.json')[controller_name])
        controller = BiasMPC(model, motor, plant, cfg)
    memory_log['after_construction'] = memory()
    # The frozen runner first creates the trial directory after simulation and metric calculation,
    # immediately before serializing trajectory arrays. Record the OS high-water mark at that boundary.
    expected_path = os.fspath(folder)
    def at_archive(event, args):
        if event == 'os.mkdir' and args[0] == expected_path:
            memory_log['before_archive'] = memory()
    sys.addaudithook(at_archive)
    metrics = run_case(controller_name, trajectory, SEED, controller, initial, plant, motor, cfg,
                       settings, folder, lambda *args, **kwargs: None, time.monotonic())
    assert 'before_archive' in memory_log
    assert metrics['complete'] and metrics['control_calls'] == 400
    write_json(memory_file, dict(trial=case, process_id=os.getpid(), environment=environment(),
               memory=memory_log, threadpools=threadpool_info(),
               hashes={name:checksum(folder/name) for name in ('metrics.json','control.csv','trajectory.npz')}))


def run_benchmark(out):
    sources = [SOURCE/'config.json', SOURCE/'corrected_models.json', SOURCE/'plant_parameters.json',
               SOURCE/'motor_model.json', SOURCE/'observer.json', SOURCE/'validate/initial_conditions'/f'{SEED}.npz',
               ROOT/'scripts/run_pelican_e3_nominal_comparison.py', ROOT/'pelican_e3/control.py',
               ROOT/'pelican_e3_corrected/control.py', ROOT/'pelican_e3/plant.py', ROOT/'pelican_e3/reference.py', Path(__file__)]
    sources += [SOURCE/'selected_configs'/f'{kind}.json' for kind in KINDS]
    trials = [(kind,name) for kind in KINDS for name in ORDER]
    random.Random(1036).shuffle(trials)
    design = dict(seed=SEED, trials=[list(trial) for trial in trials], hardware=environment(),
                  sources={str(p.relative_to(ROOT)):checksum(p) for p in sources}, independent_workers=1,
                  budget_ms=50, measurement='Wall-clock, first through last tick, including initial cold solve')
    for folder in (out, out/'cases',out/'memory'):
        folder.mkdir(parents=True, exist_ok=True)
    design_file = out/'design.json'
    if design_file.exists() and load(design_file)!=design:
        raise ValueError('Benchmark inputs differ. Use a new --out directory for a new measurement.')
    write_json(design_file,design)
    for index,(kind,name) in enumerate(trials,1):
        case = f'{kind}_{SEED}_{name}'
        stamp = out/'memory'/f'{case}.json'
        if stamp.exists():
            for key,value in load(stamp)['hashes'].items():
                assert checksum(out/'cases'/case/key)==value
            continue
        print(f'[{index}/{len(trials)}] {kind} {name}', flush=True)
        subprocess.run([sys.executable, str(Path(__file__)), '--out', str(out), '--trial', kind, name],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path, default=DEFAULT_OUT)
    parser.add_argument('--trial', nargs=2, metavar=('TRAJECTORY','CONTROLLER'))
    args=parser.parse_args()
    if args.trial:
        kind,name=args.trial
        if kind not in KINDS or name not in ORDER:
            parser.error('Unknown trajectory or controller')
        with threadpool_limits(limits=1):
            run_trial(args.out.resolve(),name,kind)
    else:
        run_benchmark(args.out.resolve())
