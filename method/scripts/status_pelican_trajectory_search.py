"""Read-only progress and provisional developer rankings, without changing selection."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pandas as pd
from pelican_trajectory_search_common import read_json, rank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['grid', 'refine', 'confirm', 'validate'], required=True)
    args = parser.parse_args()
    out = args.run_dir / args.stage
    planned = len(read_json(out / 'planned_cases.json'))
    saved = len(list((out / 'cases').glob('*/case_manifest.json')))
    print(f'{args.stage}: {saved}/{planned} saved cases')
    path = out / 'tracking_per_run.csv'
    if path.exists():
        rows = pd.read_csv(path)
        print(rows.groupby('trajectory').agg(cases=('controller', 'size'), completed=('complete', 'sum'),
              fallbacks=('fallbacks', 'sum')).to_string())
        ranking = rank(rows)
        print(ranking[ranking.eligible].groupby('trajectory', sort=False).head(2)[[
            'trajectory', 'frequency_rad_s', 'radius_m', 'climb_m_s', 'seeds', 'score_percent',
            'learned_score_percent']].to_string(index=False))
