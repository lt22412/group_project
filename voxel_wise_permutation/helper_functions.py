import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import t
import matplotlib.pyplot as plt

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
    mask = create_circular_mask(nx, ny, center, radius)
    
    data[:, mask] += snr
    return data

def simulate_null_data(n_subj=20, img_side=64, sigma=1.5, snr=0, signal_radius=0):
    nx = ny = img_side
    
    data = rng.normal(loc=0, scale=1.0, size=(n_subj, nx, ny)) 
    
    if snr > 0 and signal_radius > 0:
        data = add_circular_signal(data, snr, signal_radius)

    for i in range(n_subj):
        data[i] = gaussian_filter(data[i], sigma=sigma, mode="constant")# Gaussian smoothing is additive

    return data



def get_smoothed_truth_mask(nx, ny, sigma, radius, null_boundary = 1e-6):
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


def compute_variance_map(data, X, beta):
    n, nx, ny = data.shape
    p = X.shape[1]

    Y = data.reshape(n, -1)             # (n, V) -- Flatten for vectorization (+ speed)
    B_map = beta.reshape(p, -1)             # (p, V)

    Y_hat = X @ B_map                       # (n, V)
    res = Y - Y_hat
    var = np.sum(res**2, axis=0) / (n - p)

    return var.reshape(nx, ny)


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



# --------------------------------------------------------------------
# 4. PARAMETRIC THRESHOLD
# ---------------------------------------------------------------------

def parametric_threshold(df, alpha=0.05):
    return t.ppf(1 - alpha/2, df)



# --------------------------------------------------------------
# 5. PERMUTATION FRAMEWORK
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



def run_2d_sweep(n_runs, n_subj, img_side, snr_levels, sigma_levels, alpha, signal_radius=6, n_perm=100, null_boundary=1e-6):
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