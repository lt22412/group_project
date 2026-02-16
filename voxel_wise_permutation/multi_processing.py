from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import time

import pandas as pd

import helper_functions as hf

try:
    from tqdm.auto import tqdm
except ImportError:  # Fallback if tqdm is not installed.
    def tqdm(iterable, **kwargs):
        return iterable


N_RUNS = 20             # Number of simulation runs for FWER estimation
N_PERM = 50             # Number of permutations for permutation test
IMG_SIDE_LENGTH = 64    # Side length of the square brain image in voxels
ALPHA = 0.05            # Significance level for FWER
NULL_BOUNDARY = 1e-3    # Boundary value for null hypothesis in visualization
SIGNAL_RADIUS = 6
DISTRIBUTION = "t"      # Type of noise distribution ("t" or "normal")

TEST_SNRS = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
TEST_SIGMAS = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
N_VALUES = [2, 5, 7, 10, 15, 20, 30, 45, 60]


def run_one_snr_worker(snr_value):
    """
    Worker job for one SNR:
    loops over all n values and all sigma values, then returns flat rows.
    """
    worker_rows = []
    timing_rows = []

    for n_subj in N_VALUES:
        start_time = time.perf_counter()
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
        elapsed_sec = time.perf_counter() - start_time
        timing_rows.append((n_subj, elapsed_sec))

        # snr_levels contains one value, so j is always 0
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

    return snr_value, worker_rows, timing_rows


def main():
    max_workers = min(14, len(TEST_SNRS), os.cpu_count() or 1)
    print(f"Starting process pool with {max_workers} workers over {len(TEST_SNRS)} SNR jobs.")

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one_snr_worker, snr): snr for snr in TEST_SNRS}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Completed SNR jobs",
            unit="snr"
        ):
            snr_value, worker_rows, timing_rows = future.result()
            rows.extend(worker_rows)

            for n_subj, elapsed_sec in timing_rows:
                print(
                    f"snr={snr_value}, n={n_subj}: "
                    f"{elapsed_sec:.2f} s ({elapsed_sec / 60:.2f} min)"
                )

    final_metrics_df = pd.DataFrame(rows)
    final_metrics_df = final_metrics_df.sort_values(["n", "sm_sigma", "snr"]).reset_index(drop=True)
    final_metrics_df["method"] = "Voxel-wise permutation"
    final_metrics_df["distribution"] = DISTRIBUTION

    expected_rows = len(TEST_SNRS) * len(TEST_SIGMAS) * len(N_VALUES)
    if len(final_metrics_df) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count: got {len(final_metrics_df)}, expected {expected_rows}."
        )

    output_path = Path(__file__).resolve().parent / "test_datasets" / f"vv_prm_{DISTRIBUTION}.csv"
    final_metrics_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_metrics_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
