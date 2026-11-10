"""Observed vs predicted yield of the field trials at village (farm/site) level, by fertilisation.

No fertiliser was simulated, so the unfertilised trials are the like-for-like comparison.
Reads validation_deliverables/pred_obs_by_observation.csv and writes, next to it,
  village_trials_{crop}_{fertilization}.csv   the rows plotted
  village_trials_{crop}_{fertilization}.png   one panel per soil source

--fertilization none : unfertilised trials only, points coloured by village and source
--fertilization all  : all village-level trials, points coloured by fertilisation, metrics per group

Usage (from the repository root)
  python scripts/plot_unfertilised_trials.py --crop maize --fertilization none
  python scripts/plot_unfertilised_trials.py --crop bean --fertilization all
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FERT_COLORS = {'none': '#1b9e77', 'fertilized': '#d95f02', 'not reported': '#7570b3'}
PALETTE = plt.get_cmap('tab10').colors


def short_source(citation):
    """First author and year, e.g. 'Rivera Pena 2008'."""
    author = citation.split(',')[0].strip()
    year = next((t.strip('().') for t in citation.split() if t.strip('().').isdigit() and len(t.strip('().')) == 4), '')
    return f'{author} {year}'.strip()


def stats_text(s):
    e = s.pred_t_ha - s.obs_t_ha
    r = np.corrcoef(s.obs_t_ha, s.pred_t_ha)[0, 1] if len(s) > 2 else np.nan
    return f'n={len(s)}  bias={e.mean():+.2f}  RMSE={np.sqrt((e ** 2).mean()):.2f}  r={r:.2f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deliverables', default='validation_deliverables')
    p.add_argument('--crop', default='maize', choices=['maize', 'bean'])
    p.add_argument('--fertilization', default='none', choices=['none', 'all'])
    p.add_argument('--cultivar', default=None, help='simulated cultivar (default: the best cultivar of the crop)')
    p.add_argument('--nitro', default='Y')
    args = p.parse_args()

    d = pd.read_csv(os.path.join(args.deliverables, 'pred_obs_by_observation.csv'), dtype={'unit': str})
    d = d[(d.crop == args.crop) & (d.source_type == 'field trial') & (d.level == 'farm') & (d.nitro == args.nitro)]
    d = d[d.cultivar == args.cultivar] if args.cultivar else d[d.best_cultivar]
    d = d.assign(fertilization=d.fertilization.fillna('not reported'))
    if args.fertilization == 'none':
        d = d[d.fertilization == 'none']
    d = d.assign(site=d['unit'] + ' – ' + d['source_citation'].map(short_source))
    cultivar = d.cultivar.iloc[0]
    tag = f'village_trials_{args.crop}_{args.fertilization}'
    d.to_csv(os.path.join(args.deliverables, f'{tag}.csv'), index=False)

    if args.fertilization == 'none':
        groups = sorted(d['site'].unique())
        colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
        col = 'site'
    else:
        groups = [g for g in FERT_COLORS if g in set(d.fertilization)]
        colors = FERT_COLORS
        col = 'fertilization'

    lim = (0, np.ceil(max(d.obs_t_ha.max(), d.pred_t_ha.max()) + 0.3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, soil in zip(axes, ['local', 'soilgrids']):
        s = d[d.soil_source == soil]
        lines = [f'all: {stats_text(s)}']
        for g in groups:
            x = s[s[col] == g]
            ax.scatter(x.obs_t_ha, x.pred_t_ha, s=40, color=colors[g], edgecolor='white', lw=0.5, alpha=0.85,
                       label=f'{g} (n={len(x)})')
            if args.fertilization == 'all':
                lines.append(f'{g}: {stats_text(x)}')
        ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=7.5,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
        ax.plot(lim, lim, color='0.4', lw=1, ls='--')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_title({'local': 'local soil maps (SH)', 'soilgrids': 'SoilGrids'}[soil], fontsize=10)
        ax.set_xlabel('observed yield (t/ha)')
    axes[0].set_ylabel(f'predicted yield (t/ha), {cultivar.upper()}, NITRO={args.nitro}')
    axes[1].legend(fontsize=7, loc='lower right')
    what = 'unfertilised trials' if args.fertilization == 'none' else 'all trials by fertilisation'
    fig.suptitle(f'{args.crop.title()}, village-level field trials ({what}): observed vs predicted', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.deliverables, f'{tag}.png')
    fig.savefig(out, dpi=150)
    print(f'{len(d)} rows -> {out}')


if __name__ == '__main__':
    main()
