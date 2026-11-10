"""Compare two soil sources (e.g. local soil maps vs SoilGrids) on the same observations and cultivar.

Reads observed_vs_predicted.csv (one value per unit and season) of two validate_maize_cultivars.py runs,
keeps the rows present in both, and estimates per level and source type
  RMSE, MAE, bias and R2 of each source with 95% cluster-bootstrap intervals (units resampled),
  delta = metric(B) - metric(A) on the same resamples (paired), its 95% interval, and
  p_one_sided = share of replicates with RMSE(A) >= RMSE(B) (H1: A has the lower RMSE),
  p_two_sided = 2 min(p, 1 - p), and the same for the bias-free RMSE (ubrmse = SD of the errors) and the
  correlation r, which show whether a source follows the pattern across units and seasons better
  regardless of a constant offset (e.g. the missing fertiliser in unfertilised runs).

Usage (from the repository root)
  python scripts/compare_soil_sources.py --a validation_maize/local --b validation_maize/soilgrids \
      --labels local,soilgrids --cultivar IB0057
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_maize_cultivars import cluster_bootstrap, metrics, save_figure, subsets  # noqa: E402


def load(folder, cultivar):
    df = pd.read_csv(os.path.join(folder, 'observed_vs_predicted.csv'), dtype={'cultivar_code': str, 'unit': str})
    if cultivar is None:
        with open(os.path.join(folder, 'summary.json')) as f:
            cultivar = json.load(f)['best_cultivar_code']
    return df.loc[df['cultivar_code'] == cultivar], cultivar


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='output folder of soil source A (hypothesis: lower error)')
    p.add_argument('--b', required=True, help='output folder of soil source B')
    p.add_argument('--labels', default='local,soilgrids')
    p.add_argument('--cultivar', default=None, help='cultivar code (default: best_cultivar_code of A)')
    p.add_argument('--out', default=None, help='output folder (default: <parent of A>/comparison)')
    p.add_argument('--n-boot', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    la, lb = args.labels.split(',')
    a, code = load(args.a, args.cultivar)
    b, _ = load(args.b, code)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.a)), 'comparison')
    os.makedirs(out, exist_ok=True)

    both = a.merge(b[['row_key', 'pred_t_ha']], on='row_key', suffixes=(f'_{la}', f'_{lb}'))
    print(f'cultivar {code}: rows {la} {len(a)}, {lb} {len(b)}, in both {len(both)}')
    both.to_csv(os.path.join(out, 'paired_predictions.csv'), index=False)

    rows = []
    for level, src, sub in subsets(both):
        if sub['unit'].nunique() < 3:
            continue
        o = sub['obs_t_ha'].to_numpy()
        preds = sub[[f'pred_t_ha_{la}', f'pred_t_ha_{lb}']].to_numpy()
        boot = cluster_bootstrap(o, preds, sub['unit'].to_numpy(), args.n_boot, args.seed)
        row = {'level': level, 'source_type': src, 'cultivar_code': code}
        for k, label in enumerate((la, lb)):
            m = metrics(pd.DataFrame({'obs_t_ha': o, 'pred_t_ha': preds[:, k], 'unit': sub['unit']}))
            row.update({'n': int(m['n']), 'n_units': int(m['n_units'])})
            for name in ('rmse', 'mae', 'bias', 'r2', 'r2_pearson', 'ubrmse', 'r'):
                row[f'{name}_{label}'] = m[name]
            for name in ('rmse', 'r2', 'ubrmse', 'r'):
                row[f'{name}_{label}_lo'], row[f'{name}_{label}_hi'] = np.nanpercentile(boot[name][:, k], [2.5, 97.5])
        for name in ('rmse', 'mae', 'r2', 'ubrmse', 'r'):
            d = boot[name][:, 1] - boot[name][:, 0]
            row[f'd_{name}'] = row[f'{name}_{lb}'] - row[f'{name}_{la}']
            row[f'd_{name}_lo'], row[f'd_{name}_hi'] = np.nanpercentile(d, [2.5, 97.5])
        row['p_one_sided'] = float(np.mean(boot['rmse'][:, 0] >= boot['rmse'][:, 1]))
        row['p_two_sided'] = min(1.0, 2 * min(row['p_one_sided'], 1 - row['p_one_sided']))
        # bias-free error: does A follow the pattern (units, seasons) better than B?
        row['p_one_sided_ubrmse'] = float(np.mean(boot['ubrmse'][:, 0] >= boot['ubrmse'][:, 1]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, 'soil_source_comparison.csv'), index=False)
    pd.set_option('display.width', 220)
    cols = ['level', 'source_type', 'n', 'n_units', f'rmse_{la}', f'rmse_{lb}', 'd_rmse', 'd_rmse_lo', 'd_rmse_hi',
            'p_one_sided', 'p_two_sided', f'bias_{la}', f'bias_{lb}']
    print(table[cols].round(3).to_string(index=False))
    cols = ['level', 'source_type', f'ubrmse_{la}', f'ubrmse_{lb}', 'd_ubrmse', 'd_ubrmse_lo', 'd_ubrmse_hi',
            'p_one_sided_ubrmse', f'r_{la}', f'r_{lb}', 'd_r_lo', 'd_r_hi']
    print('\nwithout the mean bias (pattern across units and seasons):')
    print(table[cols].round(3).to_string(index=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [(lv, src, sub) for lv, src, sub in subsets(both) if src != 'all']
    lim = np.nanmax(both[['obs_t_ha', f'pred_t_ha_{la}', f'pred_t_ha_{lb}']].to_numpy()) * 1.05
    fig, axes = plt.subplots(len(groups), 2, figsize=(8.4, 4.0 * len(groups)), squeeze=False, sharex=True,
                             sharey=True)
    for i, (level, src, sub) in enumerate(groups):
        t = table.loc[(table['level'] == level) & (table['source_type'] == src)]
        for j, label in enumerate((la, lb)):
            ax = axes[i, j]
            ax.scatter(sub['obs_t_ha'], sub[f'pred_t_ha_{label}'], s=24, alpha=0.8, edgecolor='white', lw=0.4,
                       color='#1f77b4' if j == 0 else '#ff7f0e')
            ax.plot([0, lim], [0, lim], color='0.4', lw=1, ls='--')
            if not t.empty:
                r = t.iloc[0]
                text = f"n = {int(r['n'])}\nRMSE = {r[f'rmse_{label}']:.2f}\nR² = {r[f'r2_{label}']:.2f}"
                if j == 1:
                    text += f"\nΔRMSE = {r['d_rmse']:+.2f} [{r['d_rmse_lo']:+.2f}, {r['d_rmse_hi']:+.2f}]" \
                            f"\np = {r['p_one_sided']:.3f}"
                ax.text(0.03, 0.97, text, transform=ax.transAxes, va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal')
            ax.set_title(f'{label}: {level}, {src}', fontsize=9)
            ax.set_xlabel('observed (t/ha)')
            ax.set_ylabel('predicted (t/ha)')
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'scatter_soil_sources.png'))
    plt.close(fig)
    print(f'outputs in {out}')


if __name__ == '__main__':
    main()
