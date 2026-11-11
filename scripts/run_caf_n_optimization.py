"""Nitrogen optimisation for coffee (CAF2021) by village, coffee counterpart of run_dssat_n_optimization.py.

For every village and texture class a Bayesian optimisation searches the fertilisation plan
(month0/n0: yearly application of the two establishment years; monthK/nK: applications K months after
every flowering) with the highest mean harvest yield. Bounds and iterations come from the BAYESIAN
section of the coffee configuration.

Output: <output>/<GEOCODIGO>/nitrogen_results_<texture>_coffee_<cultivar>.json, one record per evaluated
plan (month0.., n0.. in kg N/ha, crop_yield, nit_ayear3, nit_year1_2); the first record is the
unfertilised run. This is the folder that NITROGEN_OPT.simulations_output_dir of the configuration must
point to for modeladocultivo_cafe.py to show the recommendations.

Run from the conda env crop_modeling (where Rscript runs CAF):
  conda run -n crop_modeling --no-capture-output python scripts/run_caf_n_optimization.py --geocodes 030106
  conda run -n crop_modeling --no-capture-output python scripts/run_caf_n_optimization.py --workers 6
"""
import argparse
import concurrent.futures
import os
import random
import sys
import tempfile
import time
import traceback

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# vendored Py_DSSATTools imports itself as top-level "DSSATTools" (crop_modeling imports it)
sys.path.insert(1, os.path.join(REPO, "Py_DSSATTools"))

CROP = 'coffee'
CONFIG = 'model_configurations/crop_model_configuration_coffee.yaml'


def init_worker():
    """Import the model packages one at a time: importing DSSATTools replaces DATA.CDE of the DSSAT
    installation and fails when another process holds it."""
    for _ in range(20):
        try:
            import crop_modeling.utils.caf_nitrogen_optimization  # noqa: F401
            return
        except PermissionError:
            time.sleep(1 + 3 * random.random())
    import crop_modeling.utils.caf_nitrogen_optimization  # noqa: F401,F811


def run_optimizer(geocode, config, cultivar='sun'):
    """Baseline and optimised plans of a village: {texture: table of evaluated plans}."""
    from crop_modeling.utils.caf_nitrogen_optimization import CAFFertilizerBayesian
    from crop_modeling.utils.ferti_optimization import _rmtree_retry

    sim_path = os.path.join(config.GENERAL_INFO.working_path, geocode)
    if os.path.exists(sim_path):
        _rmtree_retry(sim_path)
    bayesian = config.BAYESIAN
    napp = bayesian.get('n_applications', 3)
    fert_optimizer = CAFFertilizerBayesian(
        configuration_dict=config, load_env_data=False, cultivar=cultivar,
        duration=bayesian.get('duration', 7), interval_years=bayesian.get('interval_years', 0))
    fert_optimizer.set_enviromental_information(geocode)
    fert_optimizer.set_application_schedule(bayesian.schedules[f'app_{napp}'])

    flowering_dates, baseline_yields = fert_optimizer.baseline_simulation()
    results = {}
    for env, env_path in enumerate(fert_optimizer._cm_envs):
        texture = os.path.basename(env_path)
        if texture not in baseline_yields:
            continue
        optimized = fert_optimizer.simulate_env(
            env, flowering_dates[texture], n_iter=bayesian.get('n_iter', 40),
            init_points=bayesian.get('init_points', 10))
        baseline = {k: 0 for k in optimized.columns}
        baseline['crop_yield'] = baseline_yields[texture]
        results[texture] = pd.concat([pd.DataFrame([baseline]), optimized], ignore_index=True)

    _rmtree_retry(sim_path)
    return results


def run_n_optimization(geocode, config_path=CONFIG, cultivar='sun', output_path=None, work_dir=None,
                       n_iter=None):
    """Optimise a village and export one json per texture class; villages already exported are skipped."""
    config = OmegaConf.load(config_path)
    if work_dir is not None:
        config.GENERAL_INFO.working_path = work_dir
    if n_iter is not None:
        config.BAYESIAN.n_iter = n_iter
    output_path = output_path or config.BAYESIAN.get('output_path', f'{CROP}_optimization')
    sp_output_path = os.path.join(output_path, geocode)
    if os.path.exists(sp_output_path) and len(os.listdir(sp_output_path)) > 0:
        return 'cached'

    results = run_optimizer(geocode, config, cultivar)
    if not results:
        raise RuntimeError('no texture class completed')
    os.makedirs(sp_output_path, exist_ok=True)
    for texture, data in results.items():
        data.to_json(os.path.join(sp_output_path, f'nitrogen_results_{texture}_{CROP}_{cultivar}.json'),
                     orient='records')
    return f'{len(results)} texture classes'


def _worker(job):
    geocode = job[0]
    try:
        return geocode, run_n_optimization(*job), None
    except Exception:
        return geocode, 'FAILED', traceback.format_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=CONFIG)
    p.add_argument('--cultivar', default='sun', help="shade system: 'sun' or a tree species of caf/trees.py")
    p.add_argument('--output', default=None, help='default: BAYESIAN.output_path of the configuration')
    p.add_argument('--geocodes', nargs='*', default=None, help='villages (GEOCODIGO); default: all')
    p.add_argument('--limit', type=int, default=0, help='only the first N villages (pilot)')
    p.add_argument('--workers', type=int, default=4, help='villages in parallel (each runs up to 4 CAF cycles)')
    p.add_argument('--n-iter', type=int, default=None, help='overrides BAYESIAN.n_iter')
    p.add_argument('--work-dir', default=os.path.join(tempfile.gettempdir(), 'caf_n_optimization'),
                   help='temporary CAF run folders (outside OneDrive; removed after each village)')
    args = p.parse_args()

    os.chdir(REPO)   # the configuration and the CAF parameters use paths relative to the repository
    os.makedirs(args.work_dir, exist_ok=True)
    config = OmegaConf.load(args.config)
    if args.geocodes:
        roi_names = [str(g) for g in args.geocodes]
    else:
        import geopandas as gpd
        rois = gpd.read_file(config.SPATIAL_INFO.geospatial_path)
        roi_names = sorted(set(rois[config.SPATIAL_INFO.feature_name].astype(str)))
    if args.limit:
        roi_names = roi_names[:args.limit]
    print(CROP, args.cultivar, f'{len(roi_names)} villages')

    jobs = [(g, args.config, args.cultivar, args.output, args.work_dir, args.n_iter) for g in roi_names]
    failed = []
    with tqdm(total=len(jobs)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
            for geocode, status, error in executor.map(_worker, jobs):
                if error:
                    failed.append(geocode)
                    tqdm.write(f'{geocode}: {status}\n{error}')
                pbar.update(1)
    if failed:
        print(f'{len(failed)} villages failed: {" ".join(failed)}')


if __name__ == '__main__':
    main()
