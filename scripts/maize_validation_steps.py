"""Helper steps of run_maize_validation.ps1 / .sh.

  select   rank the screened cultivars by the mean RMSE on the department census over both soil sources,
           print the TOP as code:label,code:label (input of --cultivars)
  summary  collect every comparison_nitro*/<code>/soil_source_comparison.csv in one table

Usage
  python scripts/maize_validation_steps.py select validation_maize 3
  python scripts/maize_validation_steps.py summary validation_maize
"""
import glob
import os
import sys

import pandas as pd


def select(out, top):
    t = pd.concat([pd.read_csv(os.path.join(out, f'screening_{s}', 'metrics.csv'), dtype={'cultivar_code': str})
                   .assign(soil=s) for s in ('local', 'soilgrids')])
    t = t[(t['level'] == 'department') & (t['source_type'] == 'census/survey')]
    rank = (t.groupby(['cultivar_code', 'cultivar'])
            .agg(rmse_mean=('rmse', 'mean'), bias_mean=('bias', 'mean'), n_sources=('soil', 'nunique'))
            .reset_index())
    rank = rank[rank['n_sources'] == 2].sort_values('rmse_mean')
    rank.to_csv(os.path.join(out, 'cultivar_ranking_department_census.csv'), index=False)
    print(rank.round(3).to_string(index=False), file=sys.stderr)
    print(','.join(f'{c}:{l}' for c, l in zip(rank['cultivar_code'].head(top), rank['cultivar'].head(top))))


def summary(out):
    rows = [pd.read_csv(f, dtype={'cultivar_code': str}).assign(nitro=f.split('comparison_nitro')[1][0])
            for f in glob.glob(os.path.join(out, 'comparison_nitro*', '*', 'soil_source_comparison.csv'))]
    t = pd.concat(rows)
    cols = ['nitro', 'cultivar_code', 'level', 'source_type', 'n_units', 'rmse_local', 'rmse_soilgrids',
            'd_rmse', 'd_rmse_lo', 'd_rmse_hi', 'p_one_sided']
    t = t[cols].sort_values(['level', 'source_type', 'nitro', 'cultivar_code'])
    t.to_csv(os.path.join(out, 'soil_source_comparison_all.csv'), index=False)
    pd.set_option('display.width', 200)
    print(t.round(3).to_string(index=False))


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select(sys.argv[2], int(sys.argv[3]))
    else:
        summary(sys.argv[2])
