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

def plot_2d_with_numbers(img, title="2D image", fmt=".1f", fontsize=6):
    fig, ax = plt.subplots(figsize=(max(6, img.shape[1] * 0.6), max(5, img.shape[0] * 0.6)))
    im = ax.imshow(img, cmap="viridis", origin='lower')

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = img[y, x]
            color = "white" if val < (img.max() + img.min()) / 2 else "black"
            ax.text(x, y, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=fontsize, color=color)

    plt.colorbar(im, ax=ax, label="Value")
    ax.set_title(title)
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

def plot_3d_surfaces(snrs, sigmas, sens_matrix, fwer_matrix, elev=30, azim=-135):
    """
    Plots Sensitivity and FWER.
    """
    X, Y = np.meshgrid(snrs, sigmas)
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, sens_matrix, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax1.set_title("Sensitivity")
    ax1.set_xlabel("SNR", labelpad=10)
    ax1.set_ylabel("Smoothing Sigma", labelpad=10)
    ax1.set_zlabel("TPR", labelpad=10)
    ax1.set_zlim(0, 1)

    ax1.view_init(elev=elev, azim=azim)
    
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    
    # --- Plot 2: FWER ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
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

    ax2.set_title("FWER (False Positive Rate)")
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

    n_rows = len(snr_list)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))

    if n_rows == 1:
        axes = [axes]

    for i, snr in enumerate(snr_list):
        ax_left = axes[i][0]
        ax_right = axes[i][1]
        
        truth_scaled = unit_smoothed * snr
        
        observed_data = hf.simulate_null_data(n_subj=1, img_side=img_side, 
                                              sigma=sigma, snr=snr, 
                                              signal_radius=radius_signal)
        observed_slice = observed_data[0]

        im1 = ax_left.imshow(truth_scaled, cmap='hot', origin='lower', vmin=0, vmax=max(snr, 1))
        ax_left.set_title(f"Ground Truth (SNR={snr})\nTarget Area Constant", fontsize=11)
        ax_left.set_ylabel("Voxel Y")
        if i == n_rows - 1: ax_left.set_xlabel("Voxel X")
        else: ax_left.set_xticks([]) 
            
        ax_left.contour(mask_boolean, levels=[0.5], colors='red', linewidths=2, linestyles='--')
        fig.colorbar(im1, ax=ax_left, fraction=0.046, pad=0.04)

        im2 = ax_right.imshow(observed_slice, cmap='viridis', origin='lower')
        ax_right.set_title(f"Observed Data (SNR={snr})\nSmoothed ($\sigma$={sigma}) + Noise", fontsize=11)
        
        ax_right.contour(mask_boolean, levels=[0.5], colors='red', linewidths=2, linestyles='--')
        
        if i == 0:
            proxy_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, label=f'Boundary (>{null_boundary})')
            ax_right.legend(handles=[proxy_line], loc='upper right', fontsize='small')
            
        fig.colorbar(im2, ax=ax_right, fraction=0.046, pad=0.04)
        if i == n_rows - 1: ax_right.set_xlabel("Voxel X")
        else: ax_right.set_xticks([])

    plt.tight_layout()
    plt.show()


def plot_fwer_stability(snrs, sigmas, fwer_matrix, original_sigmas_list, original_snrs_list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    target_sigmas = [0.5, 1.5, 2.5] 
    colors = ['purple', 'teal', 'gold']
    
    for k, target in enumerate(target_sigmas):
        idx = np.where(np.isclose(original_sigmas_list, target))[0][0]
        
        axes[0].plot(original_snrs_list, fwer_matrix[idx, :], 
                     marker='s', markersize=5, linewidth=2, 
                     color=colors[k], label=f'$\sigma={target}$')

    axes[0].axhline(0.05, color='red', linestyle='--', linewidth=2, label='Nominal 0.05')
    axes[0].set_title("A. FWER Stability vs Signal Strength", fontsize=12)
    axes[0].set_xlabel("Signal-to-Noise Ratio (SNR)")
    axes[0].set_ylabel("False Positive Rate")
    axes[0].set_ylim(0, 0.15)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    target_snrs = [0.0, 1.5, 3.0]
    colors_snr = ['gray', 'blue', 'firebrick']

    for k, target in enumerate(target_snrs):
        idx = np.where(np.isclose(original_snrs_list, target))[0][0]
        
        axes[1].plot(original_sigmas_list, fwer_matrix[:, idx], 
                     marker='o', markersize=5, linewidth=2, 
                     color=colors_snr[k], label=f'SNR={target}')

    axes[1].axhline(0.05, color='red', linestyle='--', linewidth=2, label='Nominal 0.05')
    axes[1].set_title("B. FWER Stability vs Smoothing", fontsize=12)
    axes[1].set_xlabel("Smoothing Sigma ($\sigma$)")
    axes[1].set_ylabel("False Positive Rate")
    axes[1].set_ylim(0, 0.15)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()




def plot_sensitivity_analysis(snrs, sigmas, sens_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    n_sigmas = len(sigmas)
    n_snrs = len(snrs)
    
    colors_sig = plt.cm.viridis(np.linspace(0, 1, n_sigmas))
    colors_snr = plt.cm.plasma(np.linspace(0, 1, n_snrs))

    ax1 = axes[0]
    
    for i in range(n_sigmas):
        sig_val = sigmas[i]
        ax1.plot(snrs, sens_matrix[i, :], marker='o', markersize=5, 
                 label=f'$\sigma$={sig_val:.1f}', color=colors_sig[i], linewidth=2)
        
    ax1.set_title("A. Sensitivity vs Signal-to-Noise Ratio (SNR)", fontsize=14)
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    ax1.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax1.legend(title="Smoothing Level", fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax2 = axes[1]

    for j in range(n_snrs):
        snr_val = snrs[j]
        ax2.plot(sigmas, sens_matrix[:, j], marker='o', markersize=5, 
                 label=f'SNR={snr_val:.1f}', color=colors_snr[j], linewidth=2)

    ax2.set_title("B. Sensitivity vs Smoothing Sigma", fontsize=14)
    ax2.set_xlabel("Smoothing Sigma ($\sigma$)", fontsize=12)
    ax2.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax2.legend(title="Signal Strength", fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()