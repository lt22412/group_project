import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import helper_functions as hf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_2d(img):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="viridis", origin='lower')
    plt.colorbar(label="Intensity")
    plt.title("2D image")
    plt.tight_layout()
    plt.show()



def plot_3d(img, title="title"):
    nx, ny = img.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, img, 
                           cmap='coolwarm', 
                           edgecolor='none', 
                           antialiased=True,
                           linewidth=0)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Intensity")

    try:
        ax.set_box_aspect((1, 1, 0.5)) 
    except AttributeError:
        pass 

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Voxel X")
    ax.set_ylabel("Voxel Y")
    ax.set_zlabel("Intensity")

    plt.tight_layout()
    plt.show()

def plot_3d_surfaces(snrs, sigmas, sens_matrix, fwer_matrix, elev=30, azim=-135, extra_title = ""):
    """
    Plots Sensitivity and FWER.
    """
    X, Y = np.meshgrid(snrs, sigmas)
    
    # Changed from (14, 6) to (8, 12) for vertical stacking
    fig = plt.figure(figsize=(8, 12))
    
    # Changed from (1, 2, 1) to (2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, sens_matrix, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax1.set_title("Sensitivity" + extra_title)
    ax1.set_xlabel("SNR", labelpad=10)
    ax1.set_ylabel("Smoothing Sigma", labelpad=10)
    ax1.set_zlabel("TPR", labelpad=10)
    ax1.set_zlim(0, 1)

    ax1.view_init(elev=elev, azim=azim)
    
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    
    # --- Plot 2: FWER ---
    # Changed from (1, 2, 2) to (2, 1, 2)
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    
    cmap_custom = mcolors.ListedColormap(['cornflowerblue', 'firebrick'])
    norm_custom = mcolors.BoundaryNorm([0, 0.05, 1.0], cmap_custom.N)
    
    surf2 = ax2.plot_surface(X, Y, fwer_matrix, cmap=cmap_custom, norm=norm_custom, 
                             edgecolor='none', alpha=0.8)

    z_plane = np.full_like(X, 0.05)
    ax2.plot_surface(X, Y, z_plane, color='black', alpha=0.1)

    legend_elements = [
        mlines.Line2D([], [], color='firebrick', marker='s', linestyle='None', label='Inflated (>0.05)'),
        mlines.Line2D([], [], color='cornflowerblue', marker='s', linestyle='None', label='Safe (<=0.05)')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize='small')

    ax2.set_title("FWER (False Positive Rate)" + extra_title)
    ax2.set_xlabel("SNR", labelpad=10)
    ax2.set_ylabel("Smoothing Sigma", labelpad=10)
    ax2.set_zlabel("Prob(FP >= 1)", labelpad=10)
    ax2.set_zlim(0, 1)
    
    ax2.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    plt.show()


def generate_multi_snr_viz(snr_list=[0.5, 1.5, 3.0], null_boundary=0.01):
    """
    Visualizes Ground Truth vs Observed Data for multiple SNR levels.
    """
    img_side = 64
    center = (32, 32)
    radius_signal = 6
    sigma = 1.5
    
    unit_signal = np.zeros((img_side, img_side))
    mask_rigid = hf.create_circular_mask(img_side, img_side, center, radius_signal)
    unit_signal[mask_rigid] = 1.0
    
    unit_smoothed = hf.gaussian_filter(unit_signal, sigma=sigma, mode="constant")
    mask_boolean = unit_smoothed > null_boundary

    n_snr = len(snr_list)
    # Changed to vertical stacking: 2 plots per SNR level, one above another
    fig, axes = plt.subplots(2 * n_snr, 1, figsize=(8, 6 * n_snr))

    for i, snr in enumerate(snr_list):
        ax_top = axes[2 * i]
        ax_bottom = axes[2 * i + 1]
        
        truth_scaled = unit_smoothed * snr
        
        observed_data = hf.simulate_null_data(n_subj=1, img_side=img_side, 
                                              sigma=sigma, snr=snr, 
                                              signal_radius=radius_signal)
        observed_slice = observed_data[0]

        im1 = ax_top.imshow(truth_scaled, cmap='hot', origin='lower', vmin=0, vmax=max(snr, 1))
        ax_top.set_title(f"Ground Truth (SNR={snr})\nTarget Area Constant", fontsize=11)
        ax_top.set_ylabel("Voxel Y")
        if i == n_snr - 1: ax_bottom.set_xlabel("Voxel X")
        else: ax_top.set_xticks([]) 
            
        ax_top.contour(mask_boolean, levels=[0.5], colors='red', linewidths=2, linestyles='--')
        fig.colorbar(im1, ax=ax_top, fraction=0.046, pad=0.04)

        im2 = ax_bottom.imshow(observed_slice, cmap='viridis', origin='lower')
        ax_bottom.set_title(f"Observed Data (SNR={snr})\nSmoothed ($\sigma$={sigma}) + Noise", fontsize=11)
        ax_bottom.set_ylabel("Voxel Y")
        
        ax_bottom.contour(mask_boolean, levels=[0.5], colors='red', linewidths=2, linestyles='--')
        
        if i == 0:
            proxy_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, label=f'Boundary (>{null_boundary})')
            ax_bottom.legend(handles=[proxy_line], loc='upper right', fontsize='small')
            
        fig.colorbar(im2, ax=ax_bottom, fraction=0.046, pad=0.04)
        if i != n_snr - 1: ax_bottom.set_xticks([])

    plt.tight_layout()
    plt.show()


def plot_fwer_stability(snrs, sigmas, fwer_matrix, original_sigmas_list, original_snrs_list, extra_title = ""):
    # Changed from (1, 2) to (2, 1) and adjusted figsize
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    def _pick_targets(preferred, available):
        picked = [v for v in preferred if np.any(np.isclose(available, v))]
        if picked:
            return picked
        if len(available) <= 3:
            return list(available)
        mid = len(available) // 2
        return [available[0], available[mid], available[-1]]

    target_sigmas = _pick_targets([0.5, 1.5, 2.5], original_sigmas_list)
    colors = ['purple', 'teal', 'gold']

    for k, target in enumerate(target_sigmas):
        idx = np.where(np.isclose(original_sigmas_list, target))[0][0]

        axes[0].plot(original_snrs_list, fwer_matrix[idx, :], 
                     marker='s', markersize=5, linewidth=2, 
                     color=colors[k % len(colors)], label=f'$\sigma={target}$')

    axes[0].axhline(0.05, color='red', linestyle='--', linewidth=2, label='Nominal 0.05')
    axes[0].set_title("A. FWER Stability vs Signal Strength" + extra_title, fontsize=12)
    axes[0].set_xlabel("Signal-to-Noise Ratio (SNR)")
    axes[0].set_ylabel("False Positive Rate")
    axes[0].set_ylim(0, 0.15)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    target_snrs = _pick_targets([0.0, 1.5, 3.0], original_snrs_list)
    colors_snr = ['gray', 'blue', 'firebrick']

    for k, target in enumerate(target_snrs):
        idx = np.where(np.isclose(original_snrs_list, target))[0][0]
        
        axes[1].plot(original_sigmas_list, fwer_matrix[:, idx], 
                     marker='o', markersize=5, linewidth=2, 
                     color=colors_snr[k % len(colors_snr)], label=f'SNR={target}')

    axes[1].axhline(0.05, color='red', linestyle='--', linewidth=2, label='Nominal 0.05')
    axes[1].set_title("B. FWER Stability vs Smoothing" + extra_title, fontsize=12)
    axes[1].set_xlabel("Smoothing Sigma ($\sigma$)")
    axes[1].set_ylabel("False Positive Rate")
    axes[1].set_ylim(0, 0.15)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()




def plot_sensitivity_analysis(snrs, sigmas, sens_matrix, extra_title = ""):
    # Changed from (1, 2) to (2, 1) and adjusted figsize
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    n_sigmas = len(sigmas)
    n_snrs = len(snrs)
    
    colors_sig = plt.cm.viridis(np.linspace(0, 1, n_sigmas))
    colors_snr = plt.cm.plasma(np.linspace(0, 1, n_snrs))

    ax1 = axes[0]
    
    for i in range(n_sigmas):
        sig_val = sigmas[i]
        ax1.plot(snrs, sens_matrix[i, :], marker='o', markersize=5, 
                 label=f'$\sigma$={sig_val:.1f}', color=colors_sig[i], linewidth=2)
        
    ax1.set_title("A. Sensitivity vs Signal-to-Noise Ratio (SNR)" + extra_title, fontsize=14)
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    ax1.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax1.legend(title="Smoothing Level", fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax2 = axes[1]

    for j in range(n_snrs):
        snr_val = snrs[j]
        ax2.plot(sigmas, sens_matrix[:, j], marker='o', markersize=5, 
                 label=f'SNR={snr_val:.1f}', color=colors_snr[j], linewidth=2)

    ax2.set_title("B. Sensitivity vs Smoothing Sigma" + extra_title, fontsize=14)
    ax2.set_xlabel("Smoothing Sigma ($\sigma$)", fontsize=12)
    ax2.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax2.legend(title="Signal Strength", fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()



def _distribution_label(df):
    if "distribution" not in df.columns:
        return "unknown"
    vals = df["distribution"].dropna().unique().tolist()
    if not vals:
        return "unknown"
    if len(vals) == 1:
        return str(vals[0])
    return ",".join(str(v) for v in vals)


def _plot_metric_vs_axis(
    df,
    metric,
    x_col,
    fixed_filters,
    xlabel,
    ylabel,
    title,
    add_fwer_reference=True,
    ax=None
):
    mask = np.ones(len(df), dtype=bool)
    for col, val in fixed_filters.items():
        if col in {"sm_sigma", "snr"}:
            mask &= np.isclose(df[col], val)
        else:
            mask &= (df[col] == val)

    subset = df[mask]
    if subset.empty:
        raise ValueError(f"No rows found for filters: {fixed_filters}")

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(6, 4))

    for method, g in subset.groupby("method"):
        g = g.sort_values(x_col)
        ax.plot(
            g[x_col],
            g[metric],
            marker="o",
            linewidth=2,
            label=method
        )

    if metric == "fwer" and add_fwer_reference:
        ax.axhline(0.05, linestyle="--", color="red", label="FWER=0.05")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_sensitivity_vs_snr(df, sigma=1.5, n_val=20, ax=None):
    dist = _distribution_label(df)
    _plot_metric_vs_axis(
        df=df,
        metric="sensitivity",
        x_col="snr",
        fixed_filters={"sm_sigma": sigma, "n": n_val},
        xlabel="SNR",
        ylabel="Sensitivity",
        title=f"Sensitivity vs SNR (sigma={sigma}, n={n_val}, distribution={dist})",
        ax=ax
    )


def plot_fwer_vs_snr(df, sigma=1.5, n_val=20, ax=None):
    dist = _distribution_label(df)
    _plot_metric_vs_axis(
        df=df,
        metric="fwer",
        x_col="snr",
        fixed_filters={"sm_sigma": sigma, "n": n_val},
        xlabel="SNR",
        ylabel="FWER",
        title=f"FWER vs SNR (sigma={sigma}, n={n_val}, distribution={dist})",
        add_fwer_reference=True,
        ax=ax
    )


def plot_sensitivity_vs_sigma(df, snr_val=2.0, n_val=20, ax=None):
    dist = _distribution_label(df)
    _plot_metric_vs_axis(
        df=df,
        metric="sensitivity",
        x_col="sm_sigma",
        fixed_filters={"snr": snr_val, "n": n_val},
        xlabel="Smoothing sigma",
        ylabel="Sensitivity",
        title=f"Sensitivity vs smoothing sigma (SNR={snr_val}, n={n_val}, distribution={dist})",
        ax=ax
    )


def plot_fwer_vs_sigma(df, snr_val=2.0, n_val=20, ax=None):
    dist = _distribution_label(df)
    _plot_metric_vs_axis(
        df=df,
        metric="fwer",
        x_col="sm_sigma",
        fixed_filters={"snr": snr_val, "n": n_val},
        xlabel="Smoothing sigma",
        ylabel="FWER",
        title=f"FWER vs smoothing sigma (SNR={snr_val}, n={n_val}, distribution={dist})",
        add_fwer_reference=True,
        ax=ax
    )


def plot_sensitivity_vs_n(df, sigma=1.5, snr_val=2.0, ax=None):
    dist = _distribution_label(df)
    _plot_metric_vs_axis(
        df=df,
        metric="sensitivity",
        x_col="n",
        fixed_filters={"sm_sigma": sigma, "snr": snr_val},
        xlabel="Sample size (n)",
        ylabel="Sensitivity",
        title=f"Sensitivity vs n (sigma={sigma}, SNR={snr_val}, distribution={dist})",
        ax=ax
    )


def plot_fwer_vs_n(df, sigma=1.5, snr_val=2.0, ax=None):
    dist = _distribution_label(df)
    _plot_metric_vs_axis(
        df=df,
        metric="fwer",
        x_col="n",
        fixed_filters={"sm_sigma": sigma, "snr": snr_val},
        xlabel="Sample size (n)",
        ylabel="FWER",
        title=f"FWER vs n (sigma={sigma}, SNR={snr_val}, distribution={dist})",
        add_fwer_reference=True,
        ax=ax
    )


def plot_all_method_curves(df, sigma=1.5, snr_val=2.0, n_val=20, figsize=(18, 10)):
    """
    Creates 6 plots in one figure:
    - Sensitivity vs SNR, sigma, n
    - FWER vs SNR, sigma, n
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axs = axes.ravel()

    plot_sensitivity_vs_snr(df, sigma=sigma, n_val=n_val, ax=axs[0])
    plot_fwer_vs_snr(df, sigma=sigma, n_val=n_val, ax=axs[1])

    plot_sensitivity_vs_sigma(df, snr_val=snr_val, n_val=n_val, ax=axs[2])
    plot_fwer_vs_sigma(df, snr_val=snr_val, n_val=n_val, ax=axs[3])

    plot_sensitivity_vs_n(df, sigma=sigma, snr_val=snr_val, ax=axs[4])
    plot_fwer_vs_n(df, sigma=sigma, snr_val=snr_val, ax=axs[5])

    plt.tight_layout()
    plt.show()
