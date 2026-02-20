#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import t
import permutation_func as hf

TEST_SNRS = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
N_VALUES  = [2, 5, 7, 10, 15, 20, 30, 45, 60]

IMG_SIDE  = 64
ALPHA     = 0.05
NULL_BOUNDARY = 1e-3
SIGNAL_RADIUS = 6
CDT_P     = 0.001

def _worker_seed(base_seed, sigma_value, snr_value):
    sigma_code = int(round(float(sigma_value) * 1000))
    snr_code = int(round(float(snr_value) * 1000))
    return (int(base_seed) + 1000003 * sigma_code + 1009 * snr_code) % (2**32 - 1)


def _set_seed(seed):
    seed = int(seed)
    if hasattr(hf, "set_random_seed"):
        hf.set_random_seed(seed)
    else:
        # fallback if helper_functions.py does not define set_random_seed
        hf.rng = np.random.default_rng(seed)
        np.random.seed(seed)


def run_one_snr_worker(snr_value, sigma_value, n_runs, n_perm, noise_type, base_seed):
    snr_start = time.perf_counter()
    rows = []

    for idx, n_subj in enumerate(N_VALUES):
        cluster_forming_thr = float(t.ppf(1 - CDT_P, n_subj - 1))
        random_state_base   = base_seed + 10_000 * (TEST_SNRS.index(snr_value) * len(N_VALUES) + idx)

        sens_mat, fwer_mat = hf.run_2d_sweep_clusterwise(
            n_runs=n_runs,
            n_subj=n_subj,
            img_side=IMG_SIDE,
            snr_levels=[snr_value],
            sigma_levels=[sigma_value],
            alpha=ALPHA,
            labels=False,
            signal_radius=SIGNAL_RADIUS,
            n_perm=n_perm,
            null_boundary=NULL_BOUNDARY,
            cluster_forming_thr=cluster_forming_thr,
            random_state_base=random_state_base,
            noise_type=noise_type,
        )

        rows.append({
            "fwer":        float(fwer_mat[0, 0]),
            "sensitivity": float(sens_mat[0, 0]),
            "sm_sigma":    float(sigma_value),
            "snr":         float(snr_value),
            "n_subj":      int(n_subj),
            "method":      "clusterwise_perm",
        })

        print(f"[DONE] snr={snr_value} n_subj={n_subj:2d} | " 
              f"fwer={rows[-1]['fwer']:.3f} sens={rows[-1]['sensitivity']:.3f} | ", flush=True)
        
    elapsed = time.perf_counter() - snr_start
    return snr_value, rows, elapsed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma",        type=float, required=True)
    p.add_argument("--n-runs",       type=int,   default=750)
    p.add_argument("--n-perm",       type=int,   default=1500)
    p.add_argument("--noise-type",   type=str,   default="gaussian")
    p.add_argument("--max-workers",  type=int,   default=14)
    p.add_argument("--output-dir",   type=str,   default="./Output")
    p.add_argument("--base-seed",    type=int,   default=0)
    p.add_argument("--start-method", type=str,   default="spawn",
                   choices=["spawn", "fork", "forkserver"])
    return p.parse_args()

def main():

    args = parse_args()
    total_start = time.perf_counter()

    workers = min(args.max_workers, len(TEST_SNRS), os.cpu_count() or 1)
    print(f"Starting pool with {workers} workers for sigma={args.sigma}")
    print(f"Base seed={args.base_seed}, start_method={args.start_method}")

    rows = []
    mp_ctx = get_context(args.start_method)

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futures = [
            ex.submit(
                run_one_snr_worker,
                snr,
                args.sigma,
                args.n_runs,
                args.n_perm,
                args.distribution,
                args.base_seed,
            )
            for snr in TEST_SNRS
        ]

        for fut in as_completed(futures):
            snr_value, snr_rows, elapsed = fut.result()
            rows.extend(snr_rows)
            print(f"SNR {snr_value}: {elapsed:.2f} s ({elapsed / 60:.2f} min)")

    df = pd.DataFrame(rows).sort_values(["n", "sm_sigma", "snr"]).reset_index(drop=True)
    df["method"] = "Voxel-wise permutation"
    df["distribution"] = args.distribution

    expected = len(TEST_SNRS) * len(N_VALUES)
    if len(df) != expected:
        raise RuntimeError(f"Unexpected row count: got {len(df)}, expected {expected}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_tag = str(args.sigma).replace(".", "p")
    out_path = out_dir / f"vv_prm_{args.distribution}_sigma_{sigma_tag}.csv"
    df.to_csv(out_path, index=False)

    total_elapsed = time.perf_counter() - total_start
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Total runtime: {total_elapsed:.2f} s ({total_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()

