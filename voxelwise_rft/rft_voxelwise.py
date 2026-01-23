import numpy as np
from scipy.interpolate import UnivariateSpline
from itertools import product
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, t
rng = np.random.default_rng()

# 1. DATA SIMULATION

def parse_spatial_shape(img_size, ndim):
    """ 
    Determine spatial shape from img_size input. 
    """
    if isinstance(img_size, int):
        return (img_size,) * ndim
    else:
        return tuple(img_size)

def center_mask(spatial_shape, radius):
    """
    Create a centered circular (2D) or spherical (3D) boolean mask.
    spatial_shape: (nx, ny) or (nx, ny, nz)
    """
    d = len(spatial_shape)

    center = tuple(s//2 for s in spatial_shape)

    if d == 2:
        nx, ny = spatial_shape
        x = np.arange(nx)[:, None]
        y = np.arange(ny)[None, :]
        dist2 = (x - center[0])**2 + (y - center[1])**2
        return dist2 <= radius**2
    else:
        # d == 3
        nx, ny, nz = spatial_shape
        x = np.arange(nx)[:, None, None]
        y = np.arange(ny)[None, :, None]
        z = np.arange(nz)[None, None, :]
        dist2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        return dist2 <= radius**2

def simulate_null_data(n_subj=20, img_size=64, sigma=1.5, ndim=2, snr=0, signal_radius=0):
    """ 
    Simulate null data as smoothed Gaussian noise fields.
    Optionally insert signal of given SNR and radius at the center.
    Returns: data array of shape (n_subj, *spatial_shape)
    """
    # Determine spatial shape: int or tuple
    spatial_shape = parse_spatial_shape(img_size, ndim)

    # (1) Generate i.i.d. Gaussian noise fields
    data = rng.normal(size=(n_subj, *spatial_shape))

    # (2) Optional signal insertion (when snr and radius > 0)
    if (snr > 0) and (signal_radius > 0):
        mask = center_mask(spatial_shape, signal_radius)
        data[:, mask] += float(snr)

     # (3) Apply smoothing via Gaussian filter
    for i in range(n_subj):
        data[i] = gaussian_filter(data[i], sigma=sigma, mode='constant')
    
    return data

def get_smoothed_truth_mask(spatial_shape, sigma, radius, null_boundary = 1e-6):
    """
    Generate the Ground Truth by smoothing the pure signal.
    Any voxel > 1e-3 is considered True Signal.
    """
    spatial_shape = tuple(spatial_shape)
    pure_signal = np.zeros(spatial_shape, dtype=float)
    rigid_mask = center_mask(spatial_shape, radius)
    pure_signal[rigid_mask] = 1.0
    
    if sigma > 0:
        smoothed_signal = gaussian_filter(pure_signal, sigma=sigma, mode="constant")
    else:
        smoothed_signal = pure_signal

    return smoothed_signal > null_boundary


#2. STANDARDISE DATA

def standardise(data):
    """ 
    Standardise each field to zero mean and unit std. 
    Done before computing EC(u) curves so that thresholds u are comparable across fields.
    """
    data = np.asarray(data, dtype=float)
    n_subj = data.shape[0]
    std_data = np.zeros_like(data, dtype=float)

    for i in range(n_subj):
        field = data[i]
        std_data[i] = (field - np.mean(field)) / (np.std(field) + 1e-12)

    return std_data


# 3. GLM + T-MAP

def compute_beta_map(data, X):
    """
    Compute beta estimates voxelwise: beta = (X^T X)^(-1) X^T Y
    Returns: beta map of shape (p, *spatial_shape), where p = number of regressors
    """
    data = np.asarray(data, dtype=float)
    n_subj = data.shape[0]
    spatial = data.shape[1:]
    V = int(np.prod(spatial))

    Y = data.reshape(n_subj, -1)
    XtX_inv = np.linalg.inv(X.T @ X)
    pseudo  = XtX_inv @ X.T
    B_map = pseudo @ Y

    p = X.shape[1]
    return B_map.reshape(p, *spatial)

def compute_variance_map(data, X, beta):

    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    spatial = data.shape[1:]
    V = int(np.prod(spatial))
    p = X.shape[1]

    Y = data.reshape(n, V)
    B = beta.reshape(p, V)

    Yhat = X @ B
    res  = Y - Yhat
    var  = np.sum(res**2, axis=0) / (n - p)

    return var.reshape(*spatial)

def compute_t_map(beta, X, L, variance_map):

    beta = np.asarray(beta, dtype=float)

    # Contrast image c^T beta at each voxel
    contrast = np.tensordot(L, beta, axes=(0, 0))

    XtX_inv = np.linalg.inv(X.T @ X)
    cvar = L @ (XtX_inv @ L)

    return contrast / np.sqrt(variance_map * cvar + 1e-12)


# 4. DESIGN MATRIX

def build_design_matrix(n_subj, labels):
    """
    labels=False: one-sample test
        X = [1, 1, ..., 1]^T, L = [1]
    labels=True: two-sample test
        X has two columns of group indicators, L = [1, -1]
    """
    if labels:
        X = np.zeros((n_subj, 2))
        X[:n_subj//2, 0] = 1
        X[n_subj//2:, 1] = 1
        L  = np.array([1, -1])
    else:
        X = np.ones((n_subj, 1))
        L = np.array([1])

    df = n_subj - X.shape[1]
    return X, L, df


# 5. THE EULER CHARACTERISTIC

def compute_ec(binary_img):
    """
    Compute Euler characteristic of binary excursion set in 2D or 3D.
    2D: EC = V - E + F
    3D: EC = V - E + F - C
    where V = vertices, E = edges, F = faces, C = cubes.
    """
    A = np.asarray(binary_img, dtype=bool)
    d = A.ndim
    
    # If no pixel/voxel exceeds threshold, EC = 0
    if not A.any():
        return 0
    
    # Count d-cells: pixels (2D) or voxels (3D)
    count_d = float(np.count_nonzero(A))

    # Count (d-1)-cells by adjacency
    count_d_minus_1 = 0.0
    for axis in range(d):
        slicer1 = [slice(None)] * d
        slicer2 = [slice(None)] * d
        slicer1[axis] = slice(0, -1)
        slicer2[axis] = slice(1, None)
        count_d_minus_1 += float(np.count_nonzero(A[tuple(slicer1)] & A[tuple(slicer2)]))

    # Count vertices (0-cells): grid corners touched by active pixels/voxels
    vshape = tuple(n + 1 for n in A.shape)
    Vmask = np.zeros(vshape, dtype=bool)
    for shift in product((0, 1), repeat=d):
        vslice = tuple(slice(s, s + n) for s, n in zip(shift, A.shape))
        Vmask[vslice] |= A
    V = float(np.count_nonzero(Vmask))

    if d == 2:
        # E = shared edges, F = pixels
        E = count_d_minus_1
        F = count_d
        EC = V - E + F
        return EC
    
    # 3D: need edges (1-cells) in addition to vertices, faces, and cubes
    vol = A
    F = count_d_minus_1 # faces = axis adjacencies of voxels
    C = count_d  # cubes = active voxels

    # Count edges along each axis
    E_x = float(np.count_nonzero(vol[:, :-1, :-1] & vol[:, 1:, :-1] & vol[:, :-1, 1:] & vol[:, 1:, 1:]))  # edges along x-axis
    E_y = float(np.count_nonzero(vol[:-1, :, :-1] & vol[1:, :, :-1] & vol[:-1, :, 1:] & vol[1:, :, 1:]))  # edges along y-axis
    E_z = float(np.count_nonzero(vol[:-1, :-1, :] & vol[1:, :-1, :] & vol[:-1, 1:, :] & vol[1:, 1:, :]))  # edges along z-axis
    E = E_x + E_y + E_z

    EC = V - E + F - C
    return EC


# 6. GKF COMPONENTS

def hermite_poly(n, x):
    """
    Probabilists' Hermite polynomial H_n(x) using recurrence, with special case n=-1.
    """
    if n == -1:
        return np.sqrt(2*np.pi) * np.exp(x**2/2) * (1-norm.cdf(x))
    elif n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        Hn_2 = np.ones_like(x)
        Hn_1 = x
        for i in range(2, n+1):
            Hn = x * Hn_1 - (i - 1) * Hn_2
            Hn_2 = Hn_1
            Hn_1 = Hn
        return Hn_1

def compute_rho(j, u):
    """
    Compute the EC density component rho_j(u).
    """
    H = hermite_poly(j-1, u)
    rho = (2*np.pi)**(-(j+1)/2) * H * np.exp(-u**2/2)
    return rho


# 7. SMOOTHED DIAGONAL METHOD FOR LKC ESTIMATION

def smooth_diag(us, EC_var, min_points=5):
    """
    Smooth the diagonal of the EC variance using a quadratic spline to avoid unstable estimates in noisy or low-EC regions.
    """
    valid = EC_var > 1e-10 # non-zero variance points
    # If too few valid points, skip smoothing and return raw variance
    if valid.sum() < min_points:
        return EC_var 
    
    # Log-transform to prevent negative values
    log_var = np.log(EC_var[valid] + 1e-10)
    us_valid = us[valid]

    try:
        # Fit quadratic spline with 10% smoothing ("local quadratic smoothing with 10% nearest neighbours")
        spline = UnivariateSpline(us_valid, log_var, s=len(us_valid)*0.1, k=2)
        smooth_log_var = spline(us)
        return np.exp(smooth_log_var)
    except:
        # If smoothing fails, fill gaps with mean
        smooth_var = EC_var.copy()
        smooth_var[~valid] = EC_var[valid].mean()
        return smooth_var


# 8. ESTIMATING LKCs FROM RESIDUAL FIELDS

def estimate_lkc(residuals, n_thresh=50):
    """
    Estimate LKCs from residual fields using smoothed diagonal method:
    (1) Standardise residuals
    (2) Pick thresholds u in upper tail
    (3) Compute EC(u) for each residual field
    (4) Regress EC means on GKF components to estimate LKCs
    """
    n_subj = residuals.shape[0]
    residuals_std  = standardise(residuals)
    spatial_shape = residuals_std.shape[1:]  # (nx, ny) or (nx, ny, nz)
    d = len(spatial_shape)  # spatial dimension

    # Upper-tail thresholds (EC approximation more accurate for large u)
    u_low = np.quantile(residuals_std, 0.9)
    u_high = np.quantile(residuals_std, 0.99)
    us = np.linspace(u_low, u_high, n_thresh)

    # Compute EC for each subject and threshold u
    ECs = np.zeros((n_subj, n_thresh))
    for i in range(n_subj):
        for j, u in enumerate(us):
            # Excursion sets where the field exceeds threshold u
            excursion_set = residuals_std[i] > u
            if excursion_set.any():
                ECs[i, j] = compute_ec(excursion_set)

    # Average ECs across subjects / residual fields
    EC_mean = ECs.mean(axis=0)
    EC_var = ECs.var(axis=0, ddof=1) / n_subj

    # Build regression matrix
    R = np.vstack([compute_rho(j, us) for j in range(d + 1)]).T  # (n_thresh x 3)

    smooth_var = smooth_diag(us, EC_var)
    weights = 1.0 / (smooth_var + 1e-10)

    # Fix L0 (= EC of domain) to 1
    L0 = 1.0

    # Solve for L1,...,Ld using weighted least squares:
    # EC_mean(u) - L0*rho0(u) ≈ sum_{j=1..d} Lj * rho_j(u)
    y = EC_mean - L0 * R[:, 0]
    A = R[:, 1:]  # rho_1..rho_d

    Wsqrt = np.sqrt(weights)
    Aw = A * Wsqrt[:, None]
    yw = y * Wsqrt

    """try:
        L_rest, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    except np.linalg.LinAlgError:
        L_rest, *_ = np.linalg.lstsq(A, y, rcond=None)"""
    
    L_rest, *_ = np.linalg.lstsq(Aw, yw, rcond=None)

    LKCs = np.concatenate([[L0], L_rest])
    
    return LKCs, us, EC_mean, ECs


# 9. COMPUTE RFT THRESHOLD

def rft_threshold(LKCs, alpha=0.05, u_min=1.0, u_max=8.0, n_grid=4000):
    """
    Compute RFT threshold u such that EEC(u) ≈ alpha.
    """
    LKCs = np.asarray(LKCs, dtype=float)
    d = len(LKCs) - 1  

    u_grid = np.linspace(0.0, u_max, n_grid)

    EEC = np.zeros_like(u_grid, dtype=float)
    for j in range(d + 1):
        EEC += LKCs[j] * compute_rho(j, u_grid)

    # Only consider u >= 1 with positive EEC values
    valid = (u_grid >= u_min) & (EEC > 0)
    u_valid, eec_valid = u_grid[valid], EEC[valid]
    
    # Find first crossing below alpha
    crosses = np.where(eec_valid <= alpha)[0]

    if len(crosses) == 0 or valid.sum() < 2:
        # Return highest threshold if none found or not enough valid points
        if len(u_valid) > 0:
            return u_valid[-1]  
        # Return max grid value if no valid points
        else:
            return u_grid[-1]  
    
    idx = crosses[0]
    # Return min threshold if first point crosses
    if idx == 0:
        return u_valid[0]  
    
    # Linear interpolation to find more precise threshold
    u1, u2 = u_valid[idx-1], u_valid[idx]
    eec1, eec2 = eec_valid[idx-1], eec_valid[idx]
    found_u = u1 + (alpha - eec1) * (u2 - u1) / (eec2 - eec1)

    return found_u


# 10. COMPUTE MAX-T SIGNIFICANCE

def voxelwise_rft_threshold(data, labels, alpha=0.05, n_thresh=50):
    """
    Compute voxelwise RFT threshold for given data.
    Returns: t-map and threshold value.
    """
    data = np.asarray(data, dtype=float)
    n_subj = data.shape[0]
    spatial_shape = data.shape[1:]
    V = int(np.prod(spatial_shape))

    X, L, df = build_design_matrix(n_subj, labels)

    # (1) GLM -> t-map
    beta = compute_beta_map(data, X)
    var  = compute_variance_map(data, X, beta)
    tmap = compute_t_map(beta, X, L, var)

    # (2) Residual fields for LKC regression
    Y = data.reshape(n_subj, -1)
    B = beta.reshape(X.shape[1], -1)
    Yhat = (X @ B).reshape(n_subj, *spatial_shape)
    residuals = data - Yhat

    # (3) Estimate LKCs from residuals
    LKCs, us, EC_mean, ECs = estimate_lkc(residuals, n_thresh=n_thresh)

    # (4) Compute RFT threshold
    alpha_adj = alpha/2.0  # two-tailed
    u_crit = rft_threshold(LKCs, alpha=alpha_adj)

    # (5) Convert to t-threshold
    p_tail = 1.0 - norm.cdf(u_crit)
    thr = float(t.ppf(1.0 - p_tail, df))

    """print("LKCs:", LKCs)
    print("u_crit (z):", u_crit)
    print("thr (t):", thr)"""

    return tmap, thr


# 11. FWER ESTIMATION AND SIGNAL SWEEP

def estimate_fwer_rft(n_runs,
                      n_subj,
                      img_size,
                      sigma,
                      alpha,
                      labels,
                      ndim=2,
                      n_thresh=50):
    """
    Estimate voxelwise RFT FWER under the global null: 
    FWER ≈ P(any voxel exceeds threshold | no signal anywhere).
    """

    any_sig = np.zeros(n_runs, dtype=bool)

    for r in range(n_runs):
        data = simulate_null_data(n_subj, img_size, sigma, ndim)

        tmap, thr = voxelwise_rft_threshold(data, labels=labels, alpha=alpha, n_thresh=n_thresh)
        sig_map = np.abs(tmap) > thr
        any_sig[r] = np.any(sig_map)

    return np.mean(any_sig)

def run_2d_sweep(
    n_runs,
    n_subj,
    img_size,
    snr_levels,
    sigma_levels,
    alpha,
    signal_radius=6,
    labels=False,
    ndim=2,
    n_thresh=50,
    null_boundary=1e-6,
    verbose=True,
):
    """
    Signal-present sweep (detection power and false-positive event rate).

    For each (sigma, snr):
      (1) Build sigma-dependent true signal mask
      (2) Simulate data with given SNR, then smooth
      (2) Compute voxel-wise RFT threshold (|t|)
      (3) Compute sensitivity = fraction of true signal detected
      (4) Compute fp-event rate
    """
    # Determine spatial shape
    spatial_shape = parse_spatial_shape(img_size, ndim)

    # Allocate output matrices
    sens_matrix = np.zeros((len(sigma_levels), len(snr_levels)), dtype=float)
    fwer_matrix = np.zeros((len(sigma_levels), len(snr_levels)), dtype=float)

    if verbose:
        print(f"Starting Sweep (RFT): {len(sigma_levels)} sigmas x {len(snr_levels)} SNRs")

    # Loop over smoothing sigma
    for i, sig in enumerate(sigma_levels):
        if verbose:
            print(f"  > Processing Sigma = {sig}...")

        # Sigma-dependent truth + noise regions
        true_mask = get_smoothed_truth_mask(spatial_shape, sig, signal_radius, null_boundary)
        noise_mask = ~true_mask

        # Loop over SNR levels
        for j, snr in enumerate(snr_levels):
            detected_counts = [] # sensitivity counts
            fp_events = 0 # false-positive event counts

            # Loop over Monte Carlo runs
            for r in range(n_runs):
                data = simulate_null_data(n_subj=n_subj, img_size=img_size, sigma=sig, ndim=ndim, snr=snr, signal_radius=signal_radius)
                tmap, thr = voxelwise_rft_threshold(data, labels=labels, alpha=alpha, n_thresh=n_thresh)
                
                # Binary map of significant voxels
                sig_map = np.abs(tmap) > thr

                # Calculate sensitivity
                total_true = float(np.sum(true_mask))
                if total_true > 0:
                    sens = float(np.sum(sig_map & true_mask)) / total_true
                else:
                    sens = 0.0
                detected_counts.append(sens)

                # Count false-positive events (if at least one voxel outside true region detected as significant) in noise region
                if np.any(sig_map & noise_mask):
                    fp_events += 1

            sens_matrix[i, j] = float(np.mean(detected_counts))
            fwer_matrix[i, j] = fp_events / float(n_runs)

    if verbose:
        print("Sweep Complete.")

    return sens_matrix, fwer_matrix
