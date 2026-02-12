import helper_functions as hf
import plotters as pl
import importlib

from scipy.stats import norm, t, cauchy, poisson
from scipy.special import gamma
from scipy.ndimage import label as label_clusters
from scipy.stats import expon, probplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


importlib.reload(hf)
hf.rng = np.random.default_rng(seed=2026)

def clusterwise_RFT_Full_Test(sm_sigma_list, snr_list, n_subj_list, threshold_u, overlap_threshold,
                               img_side_length=64, signal_radius=6,
                               no_simulation=1000, fwer_alpha=0.05,method="normal"):

    results_df = pd.DataFrame(columns=["FWER", "sensitivity", "sm_sigma", "snr", "n_subj", "method", "threshold_u", "FWHM"])

    combos = [(s, snr, n)
              for s in sm_sigma_list
              for snr in snr_list
              for n in n_subj_list]

    for sm_sigma, snr, n_sub in tqdm(combos, desc="RFT Sweep"):  
        decisions = []
        sensitivities = []
        last_fwhm = 0

        for i in range(no_simulation):
            FWHM, tmap, res_map, df, true_mask = hf.simulate_one_tmap_test(
                n_subj=n_sub, img_side_length=img_side_length,
                smoothing_sigma=sm_sigma, snr=snr, signal_radius=signal_radius,method=method)
            zmap = hf.tmap_to_zmap(tmap, df)
            detect_labeled, detect_num_clusters, cluster_sizes, sig_labeled, sig_num_clusters = \
                hf.Rft_clusterwise_indentification(zmap, res_map, threshold_u, fwer_alpha)

            decision = hf.RFT_cluster_verification(sig_labeled, true_mask, overlap_threshold)
            decisions.append(decision)
            last_fwhm = FWHM

            # Sensitivity: proportion of true signal voxels detected
            sig_map = sig_labeled > 0
            true_positives = np.sum(sig_map & true_mask)
            total_signal = np.sum(true_mask)
            sens = true_positives / total_signal if total_signal > 0 else 0.0
            sensitivities.append(sens)

        FWER = 1 - np.mean(decisions)
        sensitivity = np.mean(sensitivities)

        results_df.loc[len(results_df)] = {
            "FWER": FWER,
            "sensitivity": sensitivity,
            "sm_sigma": sm_sigma,
            "snr": snr,
            "n_subj": n_sub,
            "method": "Clusterwise RFT",
            "threshold_u": threshold_u,
            "FWHM": last_fwhm
        }

    return results_df

method ="cauchy"
no_simulation    = 1000
img_side_length  = 64
signal_radius    = 6
threshold_u      = 3.2
overlap_threshold = 0
# sm_sigma_list    = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
# snr_list         = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
# n_subj_list      = [2, 5, 7, 10, 15, 20, 30, 45, 60]

sm_sigma_list    = [1.5]
snr_list         = [1.75]
n_subj_list      = [40]

print("----- RFT Simulation Settings -----")
print(f"Data dist type: {method}")

print(f"Length of sm_sigma_list : {len(sm_sigma_list)}")
print(f"Length of snr_list      : {len(snr_list)}")
print(f"Length of n_subj_list   : {len(n_subj_list)}")
print(f"Threshold_u value       : {threshold_u}")
print(f"total parameter         : {len(sm_sigma_list)*len(snr_list)*len(n_subj_list)}")

print(f"Overlap threshold       : {0}")
print(f"Number of simulations   : {no_simulation}")
print("-----------------------------------")

results_overlap0_df = clusterwise_RFT_Full_Test(
    sm_sigma_list,
    snr_list,
    n_subj_list,
    threshold_u,
    overlap_threshold,
    no_simulation=no_simulation,
    method=method
)

filename = f"results_overlap_t{overlap_threshold}_s{no_simulation}_u{threshold_u}_{method}_df.csv"
results_overlap0_df.to_csv(filename, index=False)
