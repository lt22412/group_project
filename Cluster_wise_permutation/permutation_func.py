import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

from scipy.ndimage import gaussian_filter
from scipy.stats import t
from scipy.ndimage import label as cc_label
from skimage.measure import regionprops


def create_circular_mask(nx, ny, center, radius):
    Y, X = np.ogrid[:nx, :ny]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return dist_from_center <= radius

def dice_coefficient(A, B, eps=1e-12):
    """
    Dice = 2|A∩B|/(|A|+|B|)
    If both empty -> Dice = 1 by convention.
    """
    A = A.astype(bool)
    B = B.astype(bool)
    a = A.sum()
    b = B.sum()
    if a == 0 and b == 0:
        return 1.0
    inter = np.logical_and(A, B).sum()
    return (2.0 * inter) / (a + b + eps)


def add_circular_signal(data, snr, radius, group_mask=None):
    """
    If group_mask is None -> add to all subjects.
    Else -> add only to subjects where group_mask==True.
    """
    n_subj, nx, ny = data.shape
    center = (nx // 2, ny // 2)
    mask = create_circular_mask(nx, ny, center, radius)

    if group_mask is None:
        data[:, mask] += snr
    else:
        data[group_mask][:, mask] += snr  # for 2 sample

    return data


def simulate_data(n_subj=20, img_side=64, sigma=1.5, snr=0.0, signal_radius=6, labels=False,random_state=None ):
    """
    labels=False  -> one-sample (signal in all subjects)
    labels=True   -> two-sample (signal in group2 only)
    """
    rng = np.random.default_rng(random_state)
    nx = ny = img_side
    data = rng.normal(loc=0.0, scale=1.0, size=(n_subj, nx, ny))

    if snr > 0 and signal_radius > 0:
        if labels:  # two-sample: add only to group2
            group2 = np.zeros(n_subj, dtype=bool)
            group2[n_subj // 2:] = True
            data = add_circular_signal(data, snr, signal_radius, group_mask=group2)

        else:  # one-sample: add to everyone
            data = add_circular_signal(data, snr, signal_radius, group_mask=None)

    for i in range(n_subj):
        data[i] = gaussian_filter(data[i], sigma=sigma, mode="constant")

    return data


def get_smoothed_truth_mask(nx, ny, sigma, radius, null_boundary=1e-6):
    pure_signal = np.zeros((nx, ny))
    center = (nx // 2, ny // 2)
    rigid_mask = create_circular_mask(nx, ny, center, radius)
    pure_signal[rigid_mask] = 1.0


    if sigma > 0:
        smoothed_signal = gaussian_filter(pure_signal, sigma=sigma, mode="constant")
    else:
        smoothed_signal = pure_signal

    return smoothed_signal > null_boundary


def build_design_matrix(n_subj, labels):
    if labels:
        X = np.zeros((n_subj, 2))
        X[:n_subj // 2, 0] = 1
        X[n_subj // 2:, 1] = 1
        L = np.array([1, -1])
    else:
        X = np.ones((n_subj, 1))
        L = np.array([1])

    df = n_subj - X.shape[1]
    return X, L, df


def parametric_threshold(df, alpha=0.05):
    return t.ppf(1 - alpha / 2, df)


##################

##################

def compute_beta_map(data, X):
    """
    Vectorized GLM fitting
    """
    n_subj, nx, ny = data.shape
    Y = data.reshape(n_subj, -1)  # (n, V) -- Flatten for vectorization (+ speed)

    XtX_inv = np.linalg.inv(X.T @ X)  # (p,p)
    pseudo = XtX_inv @ X.T  # (p,n)

    B_map = pseudo @ Y  # (p, V)

    return B_map.reshape(-1, nx, ny)


def compute_variance_map(data, X, beta):
    n, nx, ny = data.shape
    p = X.shape[1]

    Y = data.reshape(n, -1)  # (n, V)
    B_map = beta.reshape(p, -1)  # (p, V)

    Y_hat = X @ B_map  # (n, V)
    res = Y - Y_hat
    var = np.sum(res ** 2, axis=0) / (n - p)

    return var.reshape(nx, ny)


def compute_t_map(beta, X, L, variance_map):
    contrast = np.tensordot(L, beta, axes=(0, 0))

    XtX_inv = np.linalg.inv(X.T @ X)
    cvar = L @ (XtX_inv @ L)

    return contrast / np.sqrt(variance_map * cvar)


def tmap_from_contrast(data, X, L):
    beta = compute_beta_map(data, X)  # (p, nx, ny)
    var_map = compute_variance_map(data, X, beta)  # (nx, ny)
    tmap = compute_t_map(beta, X, L, var_map)  # (nx, ny)
    return tmap


###################
############################################

def get_max_cluster_size(tmap,
                         thr):  # gets a t statistic map and a cluster forming threshold and finds size of largest cluster

    supra = (np.abs(tmap) > thr).astype(np.int32)  # gives a 0/1 binary mask of supratheshold voxels
    labeled, _ = cc_label(
        supra)  # runs a connected component labeling on the binary image and each clutser gets assigned an interger 1,2,3...

    if labeled.max() == 0:  # the event where no cluster is found
        return 0, labeled, supra.astype(bool)  # returns max cluster size the labeled image and the suprathreshold mask

    props = regionprops(labeled)  # each element in this variable corresponds to a cluster

    sizes = [p.area for p in props]  # makes a list of clustersizes

    return max(sizes), labeled, supra.astype(
        bool)  # returns max cluster size the labeled image and the suprathreshold mask


def clusterwise_permutation_test(data, labels, alpha=0.05, n_perm=1000, cluster_forming_thr=None,
                                 random_state=None):  # takes in data, labels (T/F two sample or not), alpha for FWER level, number of permutations, voxelwise threashold

    rng_local = np.random.default_rng(random_state)  # for reporoducability

    n_subj, nx, ny = data.shape  # gets number of subjects and image dimentions

    X, L, df = build_design_matrix(n_subj, labels)  # get the design matrix, contrast nmatrix and df

    if cluster_forming_thr is None:  # if no cluster forming threshold has been chosen computer a vocelwise t threshold from the t distribution (just defines what voxels are signif)
        cluster_forming_thr = parametric_threshold(df, alpha=alpha)

    max_cluster_sizes = np.zeros(n_perm, dtype=int)  # empty array to store maxcluster size in each permutation

    for p in range(n_perm):  # permutation test begins

        if labels:  # two-sample: permute subjects (rows)
            perm = rng_local.permutation(n_subj)
            data_p = data[perm, :, :]

        else:
            # one-sample: sign flip
            signs = rng_local.choice([-1, 1], size=n_subj)
            data_p = data * signs[:, None, None]

        tmap_p = tmap_from_contrast(data_p, X, L)  # voxelwise t map for this permutatiopn
        max_sz, _, _ = get_max_cluster_size(tmap_p,
                                            cluster_forming_thr)  # gets largest suprathreshold cluster in this permutation
        max_cluster_sizes[p] = max_sz  # stroes itt

    crit_cluster_size = int(np.percentile(max_cluster_sizes, 100 * (
                1 - alpha)))  # calculates critical clutser size from the distribution this is floor alphaN

    # observed
    tmap_obs = tmap_from_contrast(data, X, L)  # obersevred t map
    _, labeled_obs, supra_obs = get_max_cluster_size(tmap_obs, cluster_forming_thr)  # get the clusters

    sig_clusters_mask = np.zeros((nx, ny), dtype=bool)  # signficance map

    if labeled_obs.max() > 0:  # if there are clusters in the map

        props = regionprops(labeled_obs)
        for k, prop in enumerate(props, start=1):
            if prop.area >= crit_cluster_size:  # check if cluster is lkarge enough to be deemed significant
                sig_clusters_mask[labeled_obs == k] = True

    return tmap_obs, labeled_obs, sig_clusters_mask, crit_cluster_size, max_cluster_sizes, cluster_forming_thr


#

#

def run_2d_sweep_clusterwise(n_runs, n_subj, img_side, snr_levels, sigma_levels, alpha, labels=False, signal_radius=6,
                             n_perm=100, null_boundary=1e-6, cluster_forming_thr=None, random_state_base=0):
    # added a cluster forming threshold, labels for two sample and random state for reporducability

    # in italise variables for result
    sens_matrix = np.zeros((len(sigma_levels), len(snr_levels)))
    fwer_matrix = np.zeros((len(sigma_levels), len(snr_levels)))

    for i, sig in enumerate(sigma_levels):  # loopijng for each egiven sigma

        true_mask = get_smoothed_truth_mask(img_side, img_side, sig, signal_radius,
                                            null_boundary)  # boolean mask of where the real signal has been added
        noise_mask = ~true_mask

        for j, snr in enumerate(snr_levels):  # loopiung for each snr level given

            detected_counts = []  # stores sensitiviry value for each run
            fp_events = 0  # counts how many runs had ANY false positive

            for r in range(n_runs):  # going to run repated simulations for estimation
                rs = random_state_base + r  # for reproducibility

                # simuloate data for the run
                data = simulate_data(
                    n_subj=n_subj,
                    img_side=img_side,
                    sigma=sig,
                    snr=snr,
                    signal_radius=signal_radius,
                    labels=labels,
                    random_state= rs
                )

                # run c;lusterwuse permuation test
                _, _, sig_clusters_mask, _, _, used_thr = clusterwise_permutation_test(
                    data=data,
                    labels=labels,
                    alpha=alpha,
                    n_perm=n_perm,
                    cluster_forming_thr=cluster_forming_thr,
                    random_state=rs
                )

                sig_map = sig_clusters_mask  # clusterwise significance map (boolean)

                # Sensitivity (TPR over true region)
                total_true = np.sum(true_mask)
                true_pos = np.sum(sig_map & true_mask)
                sens = (true_pos / total_true) if total_true > 0 else 0.0
                detected_counts.append(sens)

                # FWER event: any significant voxel in noise region
                if np.any(sig_map & noise_mask):
                    fp_events += 1

            sens_matrix[i, j] = np.mean(detected_counts)
            fwer_matrix[i, j] = fp_events / n_runs

    return sens_matrix, fwer_matrix



def run_threshold_sweep_clusterwise(
    n_runs,
    n_subj,
    img_side,
    sigma,
    snr,
    alpha,
    thresholds,
    labels=False,
    signal_radius=6,
    n_perm=1000,
    null_boundary=1e-6,
    random_state_base=0,
):

    # truth mask depends on sigma (because you use smoothed truth)
    true_mask = get_smoothed_truth_mask(img_side, img_side, sigma, signal_radius, null_boundary)
    noise_mask = ~true_mask

    out_thr = []
    out_sens = []
    out_fwer = []
    out_dice = []

    for thr in thresholds:
        sens_runs = []
        dice_runs = []
        fp_events = 0

        for r in range(n_runs):
            rs = random_state_base + r

            data = simulate_data(
                n_subj=n_subj,
                img_side=img_side,
                sigma=sigma,
                snr=snr,
                signal_radius=signal_radius,
                labels=labels,
                random_state=rs,
            )

            _, _, sig_map, _, _, _ = clusterwise_permutation_test(
                data=data,
                labels=labels,
                alpha=alpha,
                n_perm=n_perm,
                cluster_forming_thr=thr,
                random_state=rs,
            )

            # Sensitivity (TPR over true region) - your current metric
            total_true = true_mask.sum()
            true_pos = np.logical_and(sig_map, true_mask).sum()
            sens = (true_pos / total_true) if total_true > 0 else 0.0
            sens_runs.append(sens)

            # Dice (spatial overlap)
            dice_runs.append(dice_coefficient(sig_map, true_mask))

            # FWER event (any false positive in noise)
            if np.any(sig_map & noise_mask):
                fp_events += 1

        out_thr.append(thr)
        out_sens.append(float(np.mean(sens_runs)))
        out_dice.append(float(np.mean(dice_runs)))
        out_fwer.append(fp_events / n_runs)

    return {"thr": np.array(out_thr), "sens": np.array(out_sens), "fwer": np.array(out_fwer), "dice": np.array(out_dice)}
