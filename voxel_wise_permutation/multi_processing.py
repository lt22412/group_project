from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import time

import pandas as pd

import helper_functions as hf

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


N_RUNS = 20
N_PERM = 50
IMG_SIDE_LENGTH = 64
ALPHA = 0.05
NULL_BOUNDARY = 1e-3
SIGNAL_RADIUS = 6
DISTRIBUTION = "t"

TEST_SNRS = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
             2.0, 2.25, 2.5, 2.75, 3.0]

TEST_SIGMAS = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
               2.0, 2.25, 2.5, 2.75, 3.0]

N_VALUES = [2, 5, 7, 10, 15, 20, 30, 45, 60]


def run_one_snr_worker(snr_value):
    """
    Worker job for one SNR.
    Runs all n and sigma combinations and returns flat rows
    plus total runtime for this SNR.
    """
    snr_start = time.perf_counter()

    worker_rows = []

    for n_subj in N_VALUES:
        sens_mat, fwer_mat = hf.run_2d_sweep(
            n_runs=N_RUNS,
            n_subj=n_subj,
            img_side=IMG_SIDE_LENGTH,
            snr_levels=[snr_value],
            sigma_levels=TEST_SIGMAS,
            alpha=ALPHA,
            signal_radius=SIGNAL_RADIUS,
            n_perm=N_PERM,
            null_boundary=NULL_BOUNDARY,
            noise=DISTRIBUTION,
            verbose=False
        )

        for i, sigma in enumerate(TEST_SIGMAS):
            worker_rows.append(
                {
                    "fwer": float(fwer_mat[i, 0]),
                    "sensitivity": float(sens_mat[i, 0]),
                    "sm_sigma": sigma,
                    "snr": snr_value,
                    "n": n_subj,
                }
            )

    snr_elapsed = time.perf_counter() - snr_start
    return snr_value, worker_rows, snr_elapsed


def main():
    total_start = time.perf_counter()

    max_workers = min(14, len(TEST_SNRS), os.cpu_count() or 1)
    print(f"Starting process pool with {max_workers} workers over {len(TEST_SNRS)} SNR jobs.")

    rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one_snr_worker, snr): snr
            for snr in TEST_SNRS
        }

        for future in as_completed(futures):
            snr_value, worker_rows, snr_elapsed = future.result()
            rows.extend(worker_rows)

            print(
                f"SNR {snr_value}: "
                f"{snr_elapsed:.2f} s "
                f"({snr_elapsed / 60:.2f} min)"
            )

    total_elapsed = time.perf_counter() - total_start

    final_metrics_df = pd.DataFrame(rows)
    final_metrics_df = (
        final_metrics_df
        .sort_values(["n", "sm_sigma", "snr"])
        .reset_index(drop=True)
    )

    final_metrics_df["method"] = "Voxel-wise permutation"
    final_metrics_df["distribution"] = DISTRIBUTION

    expected_rows = len(TEST_SNRS) * len(TEST_SIGMAS) * len(N_VALUES)
    if len(final_metrics_df) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count: got {len(final_metrics_df)}, expected {expected_rows}."
        )

    output_path = (
        Path(__file__).resolve().parent
        / "test_datasets"
        / f"vv_prm_{DISTRIBUTION}.csv"
    )

    final_metrics_df.to_csv(output_path, index=False)

    print(f"Saved {len(final_metrics_df)} rows to {output_path}")
    print(
        f"Total runtime: {total_elapsed:.2f} s "
        f"({total_elapsed / 60:.2f} min, {total_elapsed / 3600:.2f} hr)"
    )


if __name__ == "__main__":
    main()