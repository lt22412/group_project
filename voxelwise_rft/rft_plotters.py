import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def plot_null_fwer_heatmap(n_subj_levels, sigma_levels, fwer_matrix, alpha=0.05,
                           title="Voxel-wise RFT: Global-null FWER",
                           save_path=None):

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(fwer_matrix, origin="lower", aspect="auto", 
                   cmap='RdYlGn_r', vmin=0, vmax=alpha*2)
    
    cbar = plt.colorbar(im, ax=ax, label="Observed FWER P(any FP)")
    
    # Add target line
    cbar.ax.axhline(alpha, color='blue', linestyle='--', linewidth=2)
    cbar.ax.text(0.5, alpha, f' α={alpha}', va='center', color='blue', fontweight='bold')
    
    # Set tick labels
    ax.set_xticks(range(len(n_subj_levels)))
    ax.set_xticklabels(n_subj_levels)
    ax.set_yticks(range(len(sigma_levels)))
    ax.set_yticklabels([f'{s:.2f}' for s in sigma_levels])
    
    ax.set_xlabel("Number of subjects (n_subj)", fontsize=12)
    ax.set_ylabel("Smoothing σ", fontsize=12)
    ax.set_title(f"{title}\n(target α={alpha})", fontsize=14, fontweight='bold')
    
    # Add text annotations for each cell
    for i in range(len(sigma_levels)):
        for j in range(len(n_subj_levels)):
            text_color = 'white' if fwer_matrix[i, j] > alpha else 'black'
            ax.text(j, i, f'{fwer_matrix[i, j]:.3f}',
                   ha="center", va="center", color=text_color, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_signal_heatmaps(snr_levels, sigma_levels, sens_matrix, fp_event_matrix,
                         alpha=0.05,
                         title1="Sensitivity (TPR in true region)",
                         title2="FP-event rate (outside signal)",
                         save_path=None):
  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sensitivity heatmap
    im1 = axes[0].imshow(sens_matrix, origin="lower", aspect="auto",
                        cmap='YlGn', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(snr_levels)))
    axes[0].set_xticklabels([f'{s:.2f}' for s in snr_levels])
    axes[0].set_yticks(range(len(sigma_levels)))
    axes[0].set_yticklabels([f'{s:.2f}' for s in sigma_levels])
    axes[0].set_xlabel("SNR", fontsize=12)
    axes[0].set_ylabel("Smoothing σ", fontsize=12)
    axes[0].set_title(title1, fontsize=13, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label("Sensitivity (0-1)", fontsize=11)

    # FP event rate heatmap
    im2 = axes[1].imshow(fp_event_matrix, origin="lower", aspect="auto",
                        cmap='RdYlGn_r', vmin=0, vmax=min(alpha*2, 1))
    axes[1].set_xticks(range(len(snr_levels)))
    axes[1].set_xticklabels([f'{s:.2f}' for s in snr_levels])
    axes[1].set_yticks(range(len(sigma_levels)))
    axes[1].set_yticklabels([f'{s:.2f}' for s in sigma_levels])
    axes[1].set_xlabel("SNR", fontsize=12)
    axes[1].set_ylabel("Smoothing σ", fontsize=12)
    axes[1].set_title(title2, fontsize=13, fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label("FP Event Rate", fontsize=11)
    
    # Add alpha reference line
    cbar2.ax.axhline(alpha, color='blue', linestyle='--', linewidth=2)
    cbar2.ax.text(0.5, alpha, f' α={alpha}', va='center', color='blue', fontweight='bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_3d_surface(field2d, title="", save_path=None, cmap='viridis'):

    A = np.asarray(field2d, dtype=float)
    if A.ndim != 2:
        raise ValueError("plot_3d_surface expects a 2D array.")

    nx, ny = A.shape
    X = np.arange(nx)
    Y = np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    surf = ax.plot_surface(XX, YY, A, cmap=cmap, linewidth=0, 
                          antialiased=True, alpha=0.9)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("X (voxels)", fontsize=11)
    ax.set_ylabel("Y (voxels)", fontsize=11)
    ax.set_zlabel("Intensity", fontsize=11)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_fwer_vs_sigma(sigma_levels, fwer_vals, alpha=0.05, 
                       title="FWER vs Smoothing Sigma",
                       save_path=None):

    sig = np.asarray(sigma_levels, dtype=float)
    fwer = np.asarray(fwer_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Plot FWER
    ax.plot(sig, fwer, marker="o", markersize=8, linewidth=2, 
           label='Empirical FWER', color='steelblue')
    
    # Add target line
    ax.axhline(alpha, linestyle="--", color='red', linewidth=2,
              label=f'Target α = {alpha}')
    
    # Add acceptable range (±0.01)
    ax.axhline(alpha - 0.01, linestyle=":", color='red', linewidth=1, alpha=0.5)
    ax.axhline(alpha + 0.01, linestyle=":", color='red', linewidth=1, alpha=0.5)
    ax.fill_between(sig, alpha - 0.01, alpha + 0.01, 
                    color='red', alpha=0.1, label='±0.01 tolerance')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Smoothing σ", fontsize=12)
    ax.set_ylabel("FWER", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(alpha * 3, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_performance_landscape(snr_levels, sigma_levels, sens_matrix, fp_event_matrix,
                               alpha=0.05,
                               title_left="Sensitivity",
                               title_right="FP Event Rate",
                               save_path=None):

    snr = np.asarray(snr_levels, dtype=float)
    sig = np.asarray(sigma_levels, dtype=float)

    SNR, SIG = np.meshgrid(snr, sig, indexing="xy")
    SENS = np.asarray(sens_matrix, dtype=float)
    FP = np.asarray(fp_event_matrix, dtype=float)

    fig = plt.figure(figsize=(16, 6))
    
    # Sensitivity surface
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(SNR, SIG, SENS, cmap='YlGn', 
                            linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_title(title_left, fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("SNR", fontsize=11)
    ax1.set_ylabel("Smoothing σ", fontsize=11)
    ax1.set_zlabel("Sensitivity", fontsize=11)
    ax1.set_zlim(0, 1)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # FP event rate surface
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(SNR, SIG, FP, cmap='RdYlGn_r',
                            linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_title(title_right, fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("SNR", fontsize=11)
    ax2.set_ylabel("Smoothing σ", fontsize=11)
    ax2.set_zlabel("FP Event Rate", fontsize=11)
    
    # Add alpha reference plane
    alpha_plane = np.full_like(FP, alpha)
    ax2.plot_surface(SNR, SIG, alpha_plane, alpha=0.3, color='blue',
                    linewidth=0, antialiased=True)
    
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, (ax1, ax2)


def plot_signal_and_data(true_mask, observed_field2d, signal_radius,
                         title_left="Ground Truth Signal Mask",
                         title_right="Observed Data (Smoothed + Noise)",
                         save_path=None):

    M = np.asarray(true_mask, dtype=bool)
    A = np.asarray(observed_field2d, dtype=float)

    if M.ndim != 2 or A.ndim != 2:
        raise ValueError("plot_signal_and_data expects 2D arrays.")
    if M.shape != A.shape:
        raise ValueError("true_mask and observed_field2d must have the same shape.")

    nx, ny = A.shape
    cx, cy = nx // 2, ny // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Ground truth mask
    im1 = ax1.imshow(M.astype(float), origin="lower", cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_title(title_left, fontsize=13, fontweight='bold')
    ax1.set_xlabel("Voxel X", fontsize=11)
    ax1.set_ylabel("Voxel Y", fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right: Observed data with circles
    im2 = ax2.imshow(A, origin="lower", cmap='viridis')
    ax2.set_title(title_right, fontsize=13, fontweight='bold')
    ax2.set_xlabel("Voxel X", fontsize=11)
    ax2.set_ylabel("Voxel Y", fontsize=11)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Draw signal region circle
    theta = np.linspace(0, 2*np.pi, 400)
    
    # Original signal circle (before smoothing)
    x_sig = cx + signal_radius * np.cos(theta)
    y_sig = cy + signal_radius * np.sin(theta)
    ax2.plot(y_sig, x_sig, 'r-', linewidth=2, 
            label=f'Original Signal Region (R={signal_radius})')

    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, (ax1, ax2)


def plot_diagnostic_summary(fwer_vals, sens_matrix, fp_matrix, 
                           sigma_levels, snr_levels, alpha=0.05,
                           save_path=None):
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: FWER vs sigma
    ax = axes[0, 0]
    ax.plot(sigma_levels, fwer_vals, 'o-', linewidth=2, markersize=8)
    ax.axhline(alpha, color='red', linestyle='--', linewidth=2)
    ax.fill_between(sigma_levels, alpha-0.01, alpha+0.01, alpha=0.2, color='red')
    ax.set_xlabel('Smoothing σ', fontsize=11)
    ax.set_ylabel('FWER', fontsize=11)
    ax.set_title('FWER Control (Null Case)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Sensitivity heatmap
    ax = axes[0, 1]
    im = ax.imshow(sens_matrix, origin='lower', aspect='auto', cmap='YlGn')
    ax.set_xticks(range(len(snr_levels)))
    ax.set_xticklabels([f'{s:.1f}' for s in snr_levels])
    ax.set_yticks(range(len(sigma_levels)))
    ax.set_yticklabels([f'{s:.1f}' for s in sigma_levels])
    ax.set_xlabel('SNR', fontsize=11)
    ax.set_ylabel('Smoothing σ', fontsize=11)
    ax.set_title('Sensitivity (TPR)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Panel 3: FP rate heatmap
    ax = axes[1, 0]
    im = ax.imshow(fp_matrix, origin='lower', aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(snr_levels)))
    ax.set_xticklabels([f'{s:.1f}' for s in snr_levels])
    ax.set_yticks(range(len(sigma_levels)))
    ax.set_yticklabels([f'{s:.1f}' for s in sigma_levels])
    ax.set_xlabel('SNR', fontsize=11)
    ax.set_ylabel('Smoothing σ', fontsize=11)
    ax.set_title('FP Event Rate', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Panel 4: Tradeoff curve (sensitivity vs FP rate)
    ax = axes[1, 1]
    # For each sigma, plot sensitivity vs FP rate across SNR
    for i, sig in enumerate(sigma_levels):
        ax.plot(fp_matrix[i, :], sens_matrix[i, :], 'o-', 
               label=f'σ={sig:.1f}', alpha=0.7)
    ax.axvline(alpha, color='red', linestyle='--', linewidth=2, label=f'α={alpha}')
    ax.set_xlabel('FP Event Rate', fontsize=11)
    ax.set_ylabel('Sensitivity', fontsize=11)
    ax.set_title('Power vs FP Rate Tradeoff', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(alpha*3, 1))
    ax.set_ylim(0, 1)
    
    plt.suptitle('RFT Method Diagnostic Summary', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes
