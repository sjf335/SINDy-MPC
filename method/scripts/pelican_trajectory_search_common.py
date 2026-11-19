"""Deterministic bounded candidate generation and disclosed selection rule."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from run_pelican_speed_sweep import read_json, check_preserved as previous_preserved, SNAPSHOTS
from run_pelican_e2 import checksum

ORDER = ['pd', 'sindy_current', 'linear_history', 'motor_sindy', 'sindy_history']
KEYS = ('trajectory_frequency_rad_s', 'trajectory_radius_m', 'helix_vertical_speed_m_s')


def check_preserved():
    result = previous_preserved()
    rel = 'runs/pelican_speed_sweep_dev_v1/artifact_hashes.json'
    for file, meta in read_json(ROOT / rel)['files'].items():
        assert checksum(ROOT / file) == meta['sha256'], file
    result[rel] = len(read_json(ROOT / rel)['files'])
    return result


def candidate(kind, frequency, radius, climb=.03):
    d = dict(trajectory=kind, frequency_rad_s=round(float(frequency), 6),
             radius_m=round(float(radius), 6), climb_m_s=round(float(climb if kind == 'helix' else .03), 6))
    key = sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
    return dict(candidate_id=kind + '_' + key, **d)


def grid(settings):
    return [candidate(kind, w, r, v) for kind in settings['trajectories']
            for w in settings['grid_frequency_rad_s'] for r in settings['grid_radius_m']
            for v in (settings['grid_helix_vertical_speed_m_s'] if kind == 'helix' else [.03])]


def configuration(base, c, duration):
    cfg = deepcopy(base)
    cfg['duration_s'] = float(duration)
    for key, value in zip(KEYS, [c['frequency_rad_s'], c['radius_m'], c['climb_m_s']]):
        cfg[key] = float(value)
    return cfg


def rank(rows):
    results = []
    for cid, group in rows.groupby('candidate_id', sort=True):
        counts = group.groupby('controller').size()
        eligible = (set(counts.index) == set(ORDER) and len(set(counts)) == 1
            and group.complete.all() and group.tracking_success.all()
            and group.fallbacks.sum() == 0
            and group.input_bound_violations.sum() == group.input_slew_violations.sum() == 0)
        record = {k: group.iloc[0][k] for k in ['candidate_id', 'trajectory', 'frequency_rad_s', 'radius_m', 'climb_m_s']}
        record.update(eligible=bool(eligible), seeds=int(group.seed.nunique()), completed=int(group.complete.sum()),
                      cases=len(group), score_percent=None, learned_score_percent=None, history_rmse_m=None)
        if eligible:
            means = group.groupby('controller')[['position_rmse_m', 'late_rmse_m']].mean()
            h = means.loc['sindy_history'].to_numpy()
            gains, learned = [], []
            for name in ORDER[:-1]:
                values = 100 * (1 - h / means.loc[name].to_numpy())
                gains.extend(values)
                if name != 'pd':
                    learned.extend(values)
            record.update(score_percent=float(min(gains)), learned_score_percent=float(min(learned)),
                          history_rmse_m=float(h[0]), history_late_rmse_m=float(h[1]))
        results.append(record)
    return pd.DataFrame(results).sort_values(['eligible', 'score_percent', 'history_rmse_m', 'candidate_id'],
                                             ascending=[False, False, True, True], na_position='last')


def choose(rows, settings, count):
    ranking = rank(rows)
    chosen = []
    for kind in settings['trajectories']:
        subset = ranking[(ranking.trajectory == kind) & ranking.eligible].head(count)
        if len(subset) < count:
            raise ValueError(f'Only {len(subset)} eligible settings for {kind}; need {count}')
        for row in subset.itertuples(index=False):
            chosen.append(candidate(row.trajectory, row.frequency_rad_s, row.radius_m, row.climb_m_s))
    return chosen, ranking


def refinement(winners, existing, settings):
    known = {c['candidate_id'] for c in existing}
    result = []
    for c in winners:
        dimensions = [('frequency_rad_s', 'refinement_frequency_step_rad_s', 'frequency_bounds_rad_s'),
                      ('radius_m', 'refinement_radius_step_m', 'radius_bounds_m')]
        if c['trajectory'] == 'helix':
            dimensions += [('climb_m_s', 'refinement_climb_step_m_s', 'helix_vertical_speed_bounds_m_s')]
        for field, step, bound in dimensions:
            for sign in [-1, 1]:
                point = dict(c)
                point[field] = round(c[field] + sign * settings[step], 6)
                if not settings[bound][0] <= point[field] <= settings[bound][1]:
                    continue
                new = candidate(point['trajectory'], point['frequency_rad_s'], point['radius_m'], point['climb_m_s'])
                if new['candidate_id'] not in known:
                    result.append(new)
                    known.add(new['candidate_id'])
    return result


def load_stages(root, stages):
    return pd.concat([pd.read_csv(root / stage / 'tracking_per_run.csv') for stage in stages], ignore_index=True)


def stage_design(root, stage, settings):
    if stage == 'grid':
        return grid(settings), settings['development_seeds'], None
    before = {'refine': ['grid'], 'confirm': ['grid', 'refine'], 'validate': ['grid', 'refine', 'confirm']}[stage]
    for item in before:
        audit = read_json(root / item / 'verification.json')
        assert audit['status'] == 'PASS'
        assert audit['tracking_sha256'] == checksum(root / item / 'tracking_per_run.csv')
    rows = load_stages(root, before)
    if stage == 'refine':
        winners, ranking = choose(rows, settings, 1)
        choices = refinement(winners, grid(settings), settings)
        seeds = settings['development_seeds']
    elif stage == 'confirm':
        choices, ranking = choose(rows, settings, settings['finalists_per_trajectory'])
        seeds = settings['confirmation_seeds']
    else:
        finalists = read_json(root / 'confirm' / 'candidates.json')
        ids = {c['candidate_id'] for c in finalists}
        rows = rows[rows.candidate_id.isin(ids)]
        assert rows.groupby('candidate_id').seed.nunique().eq(
            len(settings['development_seeds']) + len(settings['confirmation_seeds'])).all()
        choices, ranking = choose(rows, settings, 1)
        seeds = settings['validation_seeds']
    return choices, seeds, ranking
