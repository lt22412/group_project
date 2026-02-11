import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma
from scipy.ndimage import label as label_clusters, generate_binary_structure
from scipy.stats import t as t_dist
from scipy.stats import norm
rng = np.random.default_rng()



# -------------------------------------------------------------------
# 1. DATA SIMULATION
# -------------------------------------------------------------------------

def create_circular_mask(nx, ny, center, radius):
    """Helper to generate a boolean mask of a circle."""
    Y, X = np.ogrid[:nx, :ny]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist_from_center <= radius

def add_circular_signal(data, snr, radius): # SNR = Signal to Noise Ratio
    """
    Injects signal into the data array.
    Used 'between the lines' of noise generation and smoothing.
    """
    n_subj, nx, ny = data.shape
    center = (nx // 2, ny // 2)
    true_mask = create_circular_mask(nx, ny, center, radius)
    
    data[:, true_mask] += snr
    return data, true_mask

def simulate_null_data(n_subj=20, img_side=64, sigma=1.5, snr=0, signal_radius=0, method = "normal"):
    nx = ny = img_side
    
    if method == "normal":
        data = rng.normal(loc=0, scale=1.0, size=(n_subj, nx, ny)) # change this line
    elif method == "cauchy":
        data = rng.standard_cauchy(size=(n_subj, nx, ny))
    elif method == "t-dist":    
        data = rng.standard_t(df=3, size=(n_subj, nx, ny))

    if snr > 0 and signal_radius > 0:
        data, true_mask = add_circular_signal(data, snr, signal_radius)
    
    for i in range(n_subj):
        data[i] = gaussian_filter(data[i], sigma=sigma, mode="constant")# Gaussian smoothing is additive

    return data



def get_smoothed_truth_mask(nx, ny, sigma, radius, null_boundary = 1e-3):
    """
    Generates the Ground Truth by smoothing the pure signal.
    Any voxel > 1e-3 is considered True Signal.
    """
    pure_signal = np.zeros((nx, ny))
    center = (nx // 2, ny // 2)
    rigid_mask = create_circular_mask(nx, ny, center, radius)
    pure_signal[rigid_mask] = 1.0
    
    if sigma > 0:
        smoothed_signal = gaussian_filter(pure_signal, sigma=sigma, mode="constant")
    else:
        smoothed_signal = pure_signal

    return smoothed_signal > null_boundary


def tmap_to_zmap(tmap: np.ndarray, df: int) -> np.ndarray:
    """
    For Clusterwise RFT only
    """
    tmap = np.asarray(tmap, dtype=float)
    p = t_dist.sf(tmap, df)                        # one-sided upper-tail p
    p = np.clip(p, np.finfo(float).tiny, 1.0)
    z = norm.isf(p)

    return z


# ---------------------------------------------------------------------
# 2. MAKING GLM: BETA, CONTRAST, VARIANCE, TMAP
# ------------------------------------------------------------------  

def compute_beta_map(data, X):
    """
    Vectorized GLM fitting
    """
    n_subj, nx, ny = data.shape
    Y = data.reshape(n_subj, -1)        # (n, V) -- Flatten for vectorization (+ speed)

    XtX_inv = np.linalg.inv(X.T @ X)    # (p,p)
    pseudo = XtX_inv @ X.T              # (p,n)

    B_map = pseudo @ Y                      # (p, V)

    return B_map.reshape(-1, nx, ny)


def compute_variance_and_residual_map(data, X, beta):
    n, nx, ny = data.shape
    p = X.shape[1]

    Y = data.reshape(n, -1)             # (n, V) -- Flatten for vectorization (+ speed)
    B_map = beta.reshape(p, -1)             # (p, V)

    Y_hat = X @ B_map                       # (n, V)
    res = Y - Y_hat
    var = np.sum(res**2, axis=0) / (n - p)
    var_map = var.reshape(nx, ny)

    return var_map, res.reshape(n, nx, ny) #also return residual map


def compute_t_map(beta, X, L, variance_map):
    contrast = np.tensordot(L, beta, axes=(0, 0))

    XtX_inv = np.linalg.inv(X.T @ X)
    cvar = L @ (XtX_inv @ L)

    return contrast / np.sqrt(variance_map * cvar)



# ----------------------------------------------------------------------
# 3. DESIGN MATRIX BUILDER
# ---------------------------------------------------------------------

def build_design_matrix(n_subj, labels):
    """
    labels=False  = one-sample test (intercept only)
    labels=True   = two-sample test (group1 vs group2)
    """
    if labels:
        X = np.zeros((n_subj, 2))
        X[:n_subj//2, 0] = 1
        X[n_subj//2:, 1] = 1
        L = np.array([1, -1])
    else:
        X = np.ones((n_subj, 1))
        L = np.array([1])

    df = n_subj - X.shape[1]
    return X, L, df

def simulate_one_tmap(n_subj, img_side_length, smoothing_sigma, snr=0, signal_radius=0):
    data = simulate_null_data(n_subj, img_side_length, smoothing_sigma, snr, signal_radius)
    X, L, df = build_design_matrix(n_subj, labels=False)
    beta = compute_beta_map(data, X)
    var_map, res_map = compute_variance_and_residual_map(data, X, beta)
    tmap = compute_t_map(beta, X, L, var_map)
    fwhm = compute_2D_fwhm(res_map=res_map, voxel_size=1.0)
    return fwhm, tmap, res_map, df

def simulate_one_tmap_test(n_subj, img_side_length, smoothing_sigma, snr=0, signal_radius=0):
    data = simulate_null_data(n_subj, img_side_length, smoothing_sigma, snr, signal_radius)
    true_mask = get_smoothed_truth_mask(img_side_length, img_side_length, smoothing_sigma, signal_radius, null_boundary=1e-3)
    X, L, df = build_design_matrix(n_subj, labels=False)
    beta = compute_beta_map(data, X)
    var_map, res_map = compute_variance_and_residual_map(data, X, beta)
    tmap = compute_t_map(beta, X, L, var_map)
    fwhm = compute_2D_fwhm(res_map=res_map, voxel_size=1.0)
    return fwhm, tmap, res_map, df, true_mask

# --------------------------------------------------------------------
# 4. PARAMETRIC THRESHOLD
# ---------------------------------------------------------------------

def parametric_threshold(df, alpha=0.05):
    return t.ppf(1 - alpha/2, df)

# --------------------------------------------------------------
# 5a. Clusterwise RFT FRAMEWORK
# -------------------------------------------------------------------------

def compute_2D_fwhm(*, res_map,voxel_size=1.0):
    
    """
    Note, we assume here the voxel size is 1 unit per x,y 
    its a 2D simulation in this project s.t we set dof = 2-1 =1
    In reality, it doesnt (e.g. voxel size maybe 2mm and we do 3D, i.e dof =2)
    """
    #### Error Checking Section ###
    dof = 1

    if np.isscalar(voxel_size):
        dx = dy = float(voxel_size)
    else:
        dx, dy = map(float, voxel_size)

    res = np.asarray(res_map)
    if res.ndim != 3:
        raise ValueError("res_map must have shape (n_subj, nx, ny)")

    
    #### Calculate actually dx,dy ###
    _, nx, ny = res.shape
    sigma2_hat = np.var(res, ddof=dof)

    dres_dx = (res[:, 2:nx, 1:ny-1] - res[:, 0:nx-2, 1:ny-1]) / (2.0 * dx)
    dres_dy = (res[:, 1:nx-1, 2:ny] - res[:, 1:nx-1, 0:ny-2]) / (2.0 * dy)

    var_dx = np.var(dres_dx, ddof=dof)
    var_dy = np.var(dres_dy, ddof=dof)

    eps = 1e-12
    var_dx = max(var_dx, eps)
    var_dy = max(var_dy, eps)

    # Equation 1 in worksheet FWHM := root (8ln(2)) * λ
    const = np.sqrt(8.0 * np.log(2.0))
    fwhm_x = const * np.sqrt(sigma2_hat / var_dx)
    fwhm_y = const * np.sqrt(sigma2_hat / var_dy)
    fwhm = float(np.sqrt(fwhm_x * fwhm_y))
    return fwhm



def compute_expected_clusters(voxel_vol, smoothness, threshold, D=2):

    #we calculate E(m), denoted as exp number of clusters above threshold u.
    # when it is 2 dimension E{m} ≈ S/(2π)^(3/2) × W^(-2) × u × exp(-u²/2)
    coeff = voxel_vol / (2*np.pi) ** ((D + 1) / 2)
    smoothness_term = smoothness ** (-D)
    height_term     = threshold ** (D - 1) * np.exp(-threshold**2 / 2)
    
    return coeff * smoothness_term * height_term

def cal_exp_voxels_above_threshold(vol_S, u):
    # we calculate E(n) as expected number of voxels above threshold u.
    # E(N) = S × Φ(-u)
    return vol_S * (1 - norm.cdf(u))

def detect_clusters(tmap, threshold, k_c):
    binary_map = tmap > threshold
    structure = generate_binary_structure(2, 2)  # 8-connectivity
    detect_labeled, detect_num_clusters = label_clusters(binary_map, structure=structure)

    # Get sizes of all detected clusters
    cluster_sizes = get_cluster_sizes(detect_labeled)

    # accept clusters with size greater than k_c
    sig_map = np.zeros_like(tmap, dtype=bool)
    for cid, size in cluster_sizes.items():
        if size >= k_c:
            sig_map[cid == detect_labeled] = True

    sig_labeled, sig_num_clusters = label_clusters(sig_map, structure=structure)

    return detect_labeled, detect_num_clusters, cluster_sizes, sig_labeled, sig_num_clusters

def get_cluster_sizes(labeled_map):

    cluster_ids = np.unique(labeled_map)
    cluster_ids = cluster_ids[cluster_ids > 0]
    
    sizes = {cid: np.sum(labeled_map == cid) for cid in cluster_ids}
    return sizes

def get_max_cluster_size(labeled_map):
    """we get n_max, that is the largest cluster size within the label map"""
    sizes = get_cluster_sizes(labeled_map)
    if len(sizes) == 0:
        return 0
    return max(sizes.values())

def cal_beta_parameter(exp_m, exp_n, D=2):
    # calculate β to identify the distribution of the cluster size
    # see equation 12 in Friston, 213

    beta = (gamma(D/2+1) * exp_m / exp_n) ** (2/D)
    return beta

def cal_expected_cluster_size(exp_N, exp_m):
    #see equation 5 in Friston, 212
    if exp_m == 0:
        return np.inf 
    return exp_N / exp_m

def compute_critical_cluster_size(exp_m, beta, D=2, alpha=0.05):
    """
    Key Aim: Find k_c such tht P(n_max≥ k) ≤ alpha.
    see equation 15 in Friston, page 214
    """
    if exp_m == 0:
        return np.inf

    # k_c ≈ [log(-E(m)/log(1-α)) / β]^(D/2)
    # Note: This can produce negative k_c when E{m} < -log(1-α) ≈ 0.0513
    # In that regime, RFT theory is questionable but we compute anyway
    numerator = np.log(-exp_m/ np.log(1 -alpha))

    if numerator < 0:
        # When E{m} is very small, return minimal cluster size
        return 1.0

    k_c = (numerator / beta) ** (D/2) # crititcal cluster size
    return k_c

def compute_cluster_p_value(k, E_m, beta, D=2):
    """
    Key Aim: Calculate P(n_max ≥ k) from the observed cluster size k.
    see equation 14 in Friston, page 213
    """
    # P(n_max ≥ k) = 1 - exp(-E(m) × exp(-β× k^(2/D)))
    p_val = 1 - np.exp(-E_m * np.exp(-beta * k**(2/D)))
    return p_val

def Rft_clusterwise_indentification(tmap, res_map, threshold, fwer_alpha=0.05):

    vol_size = tmap.size
    D = tmap.ndim
    
    FWHM = compute_2D_fwhm(res_map=res_map, voxel_size=1.0)
    W = FWHM / np.sqrt(8 * np.log(2))

    # Get RFT parameters at this threshold
    exp_N = cal_exp_voxels_above_threshold(vol_size, threshold)
    exp_m = compute_expected_clusters(vol_size, W, threshold, D)
    beta = cal_beta_parameter(exp_m, exp_N, D)
    
    # find critical cluster size
    k_c = compute_critical_cluster_size(exp_m, beta, D, fwer_alpha)
    
    # detect clusters
    detect_labeled, detect_num_clusters, cluster_sizes, sig_labeled, sig_num_clusters = detect_clusters(tmap, threshold, k_c)
    return detect_labeled, detect_num_clusters, cluster_sizes, sig_labeled, sig_num_clusters


def RFT_cluster_verification(sig_labeled, true_mask, overlap_threshold=0.3):
    """
    Checks if any significant cluster is falsely detected.
    A cluster is "false" if >30% of its voxels fall outside the true signal mask.
    Returns True if no false clusters, False otherwise.
    """
    cluster_ids = np.unique(sig_labeled)
    cluster_ids = cluster_ids[cluster_ids > 0]

    false_clusters = 0
    for cid in cluster_ids:
        cluster_mask = sig_labeled == cid
        total_voxels = np.sum(cluster_mask)
        not_overlapped = np.sum(cluster_mask & ~true_mask)

        if not_overlapped > overlap_threshold * total_voxels:
            false_clusters += 1

    return false_clusters == 0


def signal_detected(sig_labeled, true_mask, overlap_threshold=0.3):
    """
    Checks if the true signal was detected by any significant cluster.
    A cluster "detects" the signal if >overlap_threshold of its voxels
    fall inside the true signal mask.
    Returns True if at least one significant cluster overlaps the signal.
    """
    cluster_ids = np.unique(sig_labeled)
    cluster_ids = cluster_ids[cluster_ids > 0]

    for cid in cluster_ids:
        cluster_mask = sig_labeled == cid
        total_voxels = np.sum(cluster_mask)
        overlapped = np.sum(cluster_mask & true_mask)

        if overlapped > overlap_threshold * total_voxels:
            return True

    return False


def clusterwise_RFT_Full_Test(n_sims, n_subj_list, sm_sigma_list, snr_list,
                               threshold_u_list, img_side_length=64,
                               signal_radius=6, fwer_alpha=0.05,
                               overlap_threshold=0.3, seed=2026):
    """
    Full parameter sweep for clusterwise RFT.

    For each parameter combination:
      - FWER: run null simulations (snr=0) and count false positives
      - Sensitivity: run signal simulations and count detection rate
        (simulation-level: did any sig cluster overlap the true signal?)

    Returns a pandas DataFrame with columns:
      sm_sigma, snr, n_subj, threshold_u, method, fwer, sensitivity
    """
    import pandas as pd
    from tqdm import tqdm

    global rng
    rng = np.random.default_rng(seed=seed)

    total = len(sm_sigma_list) * len(snr_list) * len(n_subj_list) * len(threshold_u_list)
    rows = []

    pbar = tqdm(total=total, desc="Clusterwise RFT Sweep")

    for sm_sigma in sm_sigma_list:
        for snr in snr_list:
            for n_subj in n_subj_list:
                for u in threshold_u_list:
                    # --- FWER: null simulations (snr=0) ---
                    null_fp = 0
                    for _ in range(n_sims):
                        fwhm, tmap, res_map, df = simulate_one_tmap(
                            n_subj, img_side_length, sm_sigma, snr=0, signal_radius=0)
                        zmap = tmap_to_zmap(tmap, df)
                        _, _, _, sig_labeled, sig_num = Rft_clusterwise_indentification(
                            zmap, res_map, u, fwer_alpha)
                        if sig_num > 0:
                            null_fp += 1

                    fwer = null_fp / n_sims

                    # --- Sensitivity: signal simulations ---
                    detected_count = 0
                    for _ in range(n_sims):
                        fwhm, tmap, res_map, df, true_mask = simulate_one_tmap_test(
                            n_subj, img_side_length, sm_sigma, snr=snr, signal_radius=signal_radius)
                        zmap = tmap_to_zmap(tmap, df)
                        _, _, _, sig_labeled, sig_num = Rft_clusterwise_indentification(
                            zmap, res_map, u, fwer_alpha)
                        if signal_detected(sig_labeled, true_mask, overlap_threshold):
                            detected_count += 1

                    sensitivity = detected_count / n_sims

                    rows.append({
                        'sm_sigma': sm_sigma,
                        'snr': snr,
                        'n_subj': n_subj,
                        'threshold_u': u,
                        'method': 'clusterwise_RFT',
                        'fwer': fwer,
                        'sensitivity': sensitivity
                    })

                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


# --------------------------------------------------------------
# 5b. PERMUTATION FRAMEWORK
# -------------------------------------------------------------------------

def permute_one_sample_signs(data):
    """
    One-sample permutation: apply random +-1 sign flips to each subject.
    Under H0 (EXpectation = 0), flipping signs keeps the null distribution.
    """
    signs = np.random.choice([-1, 1], size=data.shape[0])

    return data * signs[:, None, None]


def permute_two_sample_labels(n_subj):
    """Randomly assign subjects to groups 1 and 2"""
    perm = np.random.permutation(n_subj)
    X = np.zeros((n_subj, 2))
    X[perm[:n_subj//2], 0] = 1
    X[perm[n_subj//2:], 1] = 1
    L = np.array([1, -1])

    return X, L


def compute_max_t_vals_for_permutations(data, labels, n_perm):
    """
    Generate null distribution of max(|t|)
    """
    n_subj = data.shape[0]
    max_t = np.zeros(n_perm)

    for p in range(n_perm):

        if labels:
            Xp, Lp = permute_two_sample_labels(n_subj)
            beta = compute_beta_map(data, Xp)
            var  = compute_variance_map(data, Xp, beta)
            tmap = compute_t_map(beta, Xp, Lp, var)
        else:
            data_w_permuted_signs = permute_one_sample_signs(data)
            X, L, _ = build_design_matrix(n_subj, labels)
            beta = compute_beta_map(data_w_permuted_signs, X)
            var  = compute_variance_map(data_w_permuted_signs, X, beta)
            tmap = compute_t_map(beta, X, L, var)

        max_t[p] = np.max(np.abs(tmap))

    return max_t


def permutation_threshold(data, labels, alpha, n_perm):
    max_t = compute_max_t_vals_for_permutations(data, labels, n_perm)
    return np.percentile(max_t, 100 * (1 - alpha))



# --------------------------------------------------------------------
# 6. FWER ESTIMATOR (parametric or permutation)
# -------------------------------------------------------------------

def estimate_fwer(n_runs,
                  n_subj,
                  img_side,
                  sigma,
                  alpha,
                  labels,
                  n_perm=0):

    any_sig = np.zeros(n_runs, dtype=bool)

    for r in range(n_runs):

        data = simulate_null_data(n_subj, img_side, sigma)

        n_subj = data.shape[0]
        X, L, df = build_design_matrix(n_subj, labels)
        beta = compute_beta_map(data, X)
        var  = compute_variance_map(data, X, beta)
        tmap = compute_t_map(beta, X, L, var)

        if n_perm > 0:
            thr = permutation_threshold(data, labels, alpha, n_perm)
        else:
            thr = parametric_threshold(df, alpha)

        sig_map = np.abs(tmap) >thr
        any_sig[r] = np.any(sig_map)

    return np.mean(any_sig)



def run_2d_sweep(n_runs, n_subj, img_side, snr_levels, sigma_levels, alpha, signal_radius=6, n_perm=100, null_boundary=1e-3):
    """
    Runs a simulation grid over SNR levels and Sigma levels
    """
    
    # Initialize 2D arrays to store results
    sens_matrix = np.zeros((len(sigma_levels), len(snr_levels)))
    fwer_matrix = np.zeros((len(sigma_levels), len(snr_levels)))
    
    print(f"Starting 3D Sweep: {len(sigma_levels)} Sigmas x {len(snr_levels)} SNRs")
    
    for i, sig in enumerate(sigma_levels):
        print(f"  > Processing Sigma = {sig}...")
        
        true_mask = get_smoothed_truth_mask(img_side, img_side, sig, signal_radius, null_boundary)
        noise_mask = ~true_mask 

        for j, snr in enumerate(snr_levels):
            
            detected_counts = []
            fp_events = 0
            
            for r in range(n_runs):
                data = simulate_null_data(n_subj, img_side, sigma=sig, snr=snr, signal_radius=signal_radius)
                
                thr = permutation_threshold(data, labels=False, alpha=alpha, n_perm=n_perm)
                
                X, L, df = build_design_matrix(n_subj, labels=False)
                beta = compute_beta_map(data, X)
                var  = compute_variance_map(data, X, beta)
                tmap = compute_t_map(beta, X, L, var)
                
                sig_map = np.abs(tmap) > thr
                
                true_pos_count = np.sum(sig_map & true_mask)
                total_true_pixels = np.sum(true_mask)
                
                if total_true_pixels > 0:
                    sens = true_pos_count / total_true_pixels
                else:
                    sens = 0.0
                detected_counts.append(sens)
                
                false_pos_count = np.sum(sig_map & noise_mask)
                if false_pos_count > 0:
                    fp_events += 1
            
            sens_matrix[i, j] = np.mean(detected_counts)
            fwer_matrix[i, j] = fp_events / n_runs
            
    print("Sweep Complete.")
    return sens_matrix, fwer_matrix