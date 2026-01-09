import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import helper_functions as hf

def plot_2d(img):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="viridis", origin='lower')
    plt.colorbar(label="Intensity")
    plt.title("2D image")
    plt.tight_layout()
    plt.show()



def plot_3d(img, title = "title"):
    nx, ny = img.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, img, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensity")

    plt.tight_layout()
    plt.show()


def plot_3d_surfaces(snrs, sigmas, sens_matrix, fwer_matrix, elev=30, azim=-60):
    """
    Plots Sensitivity and FWER.
    """
    X, Y = np.meshgrid(snrs, sigmas)
    
    fig = plt.figure(figsize=(14, 6))
    
    # --- Plot 1: Sensitivity (Standard Viridis) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, sens_matrix, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title("Sensitivity (Power)")
    ax1.set_xlabel("SNR")
    ax1.set_ylabel("Smoothing Sigma")
    ax1.set_zlabel("TPR")
    ax1.set_zlim(0, 1)
    ax1.view_init(elev=elev, azim=azim)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    cmap_custom = mcolors.ListedColormap(['cornflowerblue', 'firebrick'])
    norm_custom = mcolors.BoundaryNorm([0, 0.05, 1.0], cmap_custom.N)
    
    # 2. Plot the Surface using this colormap
    surf2 = ax2.plot_surface(X, Y, fwer_matrix, cmap=cmap_custom, norm=norm_custom, 
                             edgecolor='none', alpha=0.8)

    # 4. Add a transparent Plane for visual reference (Optional, but looks nice)
    z_plane = np.full_like(X, 0.05)
    ax2.plot_surface(X, Y, z_plane, color='black', alpha=0.1)

    # Legend
    legend_elements = [
        mlines.Line2D([], [], color='firebrick', marker='s', linestyle='None', label='Inflated (>0.05)'),
        mlines.Line2D([], [], color='cornflowerblue', marker='s', linestyle='None', label='Safe (<=0.05)'),
        mlines.Line2D([], [], color='lime', linewidth=3, label='Threshold Line')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize='small')

    ax2.set_title("FWER (False Positive Rate)\n(Visualizing the 0.05 Breach)")
    ax2.set_xlabel("SNR")
    ax2.set_ylabel("Smoothing Sigma")
    ax2.set_zlabel("Prob(FP >= 1)")
    ax2.set_zlim(0, 1)
    ax2.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    plt.show()




def plot_net_performance_3d(snrs, sigmas, sens_matrix, fwer_matrix):
    """
    Plots (Sensitivity - FWER) as a 3D surface.
    Range: -1 (Worst) to +1 (Best).
    """
    # 1. Calculate the Net Metric
    net_matrix = sens_matrix - fwer_matrix
    
    X, Y = np.meshgrid(snrs, sigmas)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, net_matrix, cmap='RdYlGn', edgecolor='none', alpha=0.9)
    
    ax.set_title("Net Performance Metric\n(Sensitivity minus FWER)")
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Smoothing Sigma")
    ax.set_zlabel("Net Score")
    
    ax.set_zlim(-1, 1)
    
    zero_plane = np.zeros_like(X)
    ax.plot_surface(X, Y, zero_plane, color='gray', alpha=0.2)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Score (1=Perfect, 0=Useless, <0=Noise)")
    
    plt.tight_layout()
    plt.show()



def generate_method_viz():
    img_side = 64
    center = (32, 32)
    radius_signal = 6
    radius_buffer = 8
    snr = 2.0
    sigma = 1.5
    
    truth_img = np.zeros((img_side, img_side))
    mask = hf.create_circular_mask(img_side, img_side, center, radius_signal)
    truth_img[mask] = snr
    
    # 2. Generate the "Observed Data" (Post-smoothing, Post-noise)
    observed_data = hf.simulate_null_data(n_subj=1, img_side=img_side, sigma=sigma, snr=snr, signal_radius=radius_signal)
    observed_slice = observed_data[0] 
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PLOT 1: GROUND TRUTH
    im1 = axes[0].imshow(truth_img, cmap='gray', origin='lower', vmin=0, vmax=snr*1.2)
    axes[0].set_title("1. Ground Truth (Neuronal Input)\nDiscrete Cylinder (Radius=6)", fontsize=12)
    axes[0].set_xlabel("Voxel X")
    axes[0].set_ylabel("Voxel Y")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # PLOT 2: OBSERVED DATA
    im2 = axes[1].imshow(observed_slice, cmap='viridis', origin='lower')
    axes[1].set_title(f"2. Observed Data (Scanner Output)\nSmoothed ($\sigma$={sigma}) + Noise", fontsize=12)
    axes[1].set_xlabel("Voxel X")
    
    circle_signal = patches.Circle(center, radius_signal, linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
    axes[1].add_patch(circle_signal)
    
    # Ring 2: The Buffer Boundary (White/Dashed)
    circle_buffer = patches.Circle(center, radius_buffer, linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    axes[1].add_patch(circle_buffer)
    
    axes[1].legend([circle_signal, circle_buffer], 
                   ['True Signal (R=6)', 'Buffer Limit (R=10)'], 
                   loc='upper right', fontsize='small', framealpha=0.9)

    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()