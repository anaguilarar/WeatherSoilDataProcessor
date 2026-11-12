"""P and K yield response (QUEFTS) of a village run folder, from the command line.

The method and the outputs are described in crop_modeling/utils/quefts_response.py; modeladocultivo_.py calls
the same code at the end of a maize or bean run. Run from the repository root, in an environment with PyQUEFTS:
  python scripts/quefts_response.py --run-dir runs/070331 --crop bean
  python scripts/quefts_response.py --run-dir runs/080416 --crop maize
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from crop_modeling.utils.quefts_response import DEFAULT_RATES, QUEFTS_CROP, quefts_response


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, help='village run folder, e.g. runs/070331')
    p.add_argument('--crop', required=True, choices=sorted(QUEFTS_CROP))
    p.add_argument('--p-rates', default=DEFAULT_RATES['p'], help='extra kg P/ha, first:last:step or a list')
    p.add_argument('--k-rates', default=DEFAULT_RATES['k'], help='extra kg K/ha')
    p.add_argument('--n-rates', default=DEFAULT_RATES['n'],
                   help='extra kg N/ha (default 0: the DSSAT attainable yield already includes the N of the run)')
    p.add_argument('--percentile', type=float, default=75,
                   help='planting dates with a mean yield above this percentile are used')
    p.add_argument('--out', default=None, help='default: <run-dir>/quefts')
    args = p.parse_args()
    quefts_response(args.run_dir, args.crop, p_rates=args.p_rates, k_rates=args.k_rates, n_rates=args.n_rates,
                    percentile=args.percentile, out_dir=args.out)


if __name__ == '__main__':
    main()
