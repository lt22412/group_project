import argparse
import permutation_func as hf
import numpy as np
import pandas as pd
from scipy.stats import t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, required=True)
    args = parser.parse_args()

    sigma = args.sigma

    # === your grids ===
    sigma_levels = [sigma]
    snr_levels   = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    n_subj_levels = [2, 5, 7, 10, 15, 20, 30, 45, 60]

    n_runs = 200
    n_perm = 50
    alpha = 0.05
    img_side = 64
    signal_radius = 6
    null_boundary = 1e-3
    cdt_p = 0.001

    rows = []
    grid_index = 0
    random_state_base = 0

    for n_subj in n_subj_levels:
        df = n_subj - 1
        cluster_forming_thr = float(t.ppf(1 - cdt_p, df))

        for snr in snr_levels:
            grid_index += 1
            base = random_state_base + 10_000 * grid_index

            sens_mat, fwer_mat = hf.run_2d_sweep_clusterwise(
                n_runs=n_runs,
                n_subj=n_subj,
                img_side=img_side,
                snr_levels=[snr],
                sigma_levels=[sigma],
                alpha=alpha,
                labels=False,
                signal_radius=signal_radius,
                n_perm=n_perm,
                null_boundary=null_boundary,
                cluster_forming_thr=cluster_forming_thr,
                random_state_base=base,
                noise_type="gaussian"
            )

            rows.append({
                "fwer": float(fwer_mat[0, 0]),
                "sensitivity": float(sens_mat[0, 0]),
                "sm_sigma": float(sigma),
                "snr": float(snr),
                "n_subj": int(n_subj),
                "method": "clusterwise_perm"
            })

    df_out = pd.DataFrame(rows).sort_values(["n_subj", "snr"]).reset_index(drop=True)
    df_out.to_csv(f"clusterwise_sigma_{sigma}.csv", index=False)
    print(f"Saved clusterwise_sigma_{sigma}.csv")

if __name__ == "__main__":
    main()
