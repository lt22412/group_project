import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import helper_functions as hf
import plotters as pl  # keep if needed elsewhere
import importlib

from scipy.stats import norm, t, cauchy, poisson
from scipy.special import gamma
from scipy.ndimage import label as label_clusters
from scipy.stats import expon, probplot
import matplotlib.pyplot as plt


def set_thread_env(n_threads: int) -> None:
    """
    Control threading used by NumPy/SciPy backends (OpenMP/MKL/OpenBLAS).
    This does NOT parallelise Python loops, but can help if heavy work is in
    vectorised/compiled code.
    """
    n = str(max(1, int(n_threads)))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n


def clusterwise_RFT_Full_Test(
    sm_sigma_list,
    snr_list,
    n_subj_list,
    threshold_u,
    overlap_threshold,
    img_side_length=64,
    signal_radius=6,
    no_simulation=1000,
    fwer_alpha=0.05,
    method="normal",
    chunk_id=0,
    n_chunks=1,
    base_seed=2026,
):
    print("----- RFT Simulation Settings -----")
    print(f"Data dist type          : {method}")
    print(f"img_side_length         : {img_side_length}")
    print(f"signal_radius           : {signal_radius}")
    print(f"Length of sm_sigma_list : {len(sm_sigma_list)}")
    print(f"Length of snr_list      : {len(snr_list)}")
    print(f"Length of n_subj_list   : {len(n_subj_list)}")
    print(f"Threshold_u value       : {threshold_u}")
    print(f"Overlap threshold       : {overlap_threshold}")
    print(f"Number of simulations   : {no_simulation}")
    print(f"Chunking                : chunk_id={chunk_id} of n_chunks={n_chunks}")
    print("-----------------------------------")

    results_df = pd.DataFrame(
        columns=["FWER", "sensitivity", "sm_sigma", "snr", "n_subj", "method", "threshold_u", "FWHM"]
    )

    combos = [(s, snr, n) for s in sm_sigma_list for snr in snr_list for n in n_subj_list]

    # Each job only computes a subset of simulations:
    # i = chunk_id, chunk_id+n_chunks, chunk_id+2*n_chunks, ...
    for sm_sigma, snr, n_sub in tqdm(combos, desc="RFT Sweep"):
        decisions = []
        sensitivities = []
        last_fwhm = 0.0

        for i in range(chunk_id, no_simulation, n_chunks):
            # deterministic per-simulation seed, stable across chunks
            hf.rng = np.random.default_rng(seed=base_seed + i)

            FWHM, tmap, res_map, df, true_mask = hf.simulate_one_tmap_test(
                n_subj=n_sub,
                img_side_length=img_side_length,
                smoothing_sigma=sm_sigma,
                snr=snr,
                signal_radius=signal_radius,
                method=method,
            )

            zmap = hf.tmap_to_zmap(tmap, df)

            _, _, _, sig_labeled, _ = hf.Rft_clusterwise_indentification(
                zmap, res_map, threshold_u, fwer_alpha
            )

            decision = hf.RFT_cluster_verification(sig_labeled, true_mask, overlap_threshold)
            decisions.append(decision)
            last_fwhm = FWHM

            sig_map = sig_labeled > 0
            true_positives = np.sum(sig_map & true_mask)
            total_signal = np.sum(true_mask)
            sens = true_positives / total_signal if total_signal > 0 else 0.0
            sensitivities.append(sens)

        # Note: these are PARTIAL results for this chunk.
        # You will merge across chunks later.
        FWER = 1 - np.mean(decisions) if len(decisions) else np.nan
        sensitivity = np.mean(sensitivities) if len(sensitivities) else np.nan

        results_df.loc[len(results_df)] = {
            "FWER": FWER,
            "sensitivity": sensitivity,
            "sm_sigma": sm_sigma,
            "snr": snr,
            "n_subj": n_sub,
            "method": method,
            "threshold_u": threshold_u,
            "FWHM": last_fwhm,
        }

    os.makedirs("./Output", exist_ok=True)
    filename = (
        f"./Output/partial_overlap_t{overlap_threshold}"
        f"_s{no_simulation}_u{threshold_u}_{method}"
        f"_chunk{chunk_id}of{n_chunks}.csv"
    )
    results_df.to_csv(filename, index=False)
    print(f"Complete! Wrote: {filename}")
    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--no_simulation", type=int, required=True)
    parser.add_argument("--img_side_length", type=int, required=True)
    parser.add_argument("--signal_radius", type=int, required=True)
    parser.add_argument("--threshold_u", type=float, required=True)
    parser.add_argument("--overlap_threshold", type=float, required=True)

    # Chunking for Slurm arrays (Option A)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)

    # Thread control for NumPy/SciPy internals (OpenMP/MKL/OpenBLAS)
    parser.add_argument("--omp_threads", type=int, default=1)

    args = parser.parse_args()

    # Control threaded libs (optional but recommended on HPC)
    set_thread_env(args.omp_threads)

    # Define your sweep lists (as you have them)
    sm_sigma_list = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    snr_list = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    n_subj_list = [2, 5, 7, 10, 15, 20, 30, 45, 60]

    clusterwise_RFT_Full_Test(
        sm_sigma_list=sm_sigma_list,
        snr_list=snr_list,
        n_subj_list=n_subj_list,
        threshold_u=args.threshold_u,
        overlap_threshold=args.overlap_threshold,
        img_side_length=args.img_side_length,
        signal_radius=args.signal_radius,
        no_simulation=args.no_simulation,
        method=args.method,
        chunk_id=args.chunk_id,
        n_chunks=args.n_chunks,
        base_seed=2026,
    )


if __name__ == "__main__":
    main()
