import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

def plot_2d(img):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="viridis", origin='lower')
    plt.colorbar(label="Intensity")
    plt.title("2D image")
    plt.tight_layout()
    plt.show()



def plot_3d(img):
    nx, ny = img.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, img, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_title("3D surface")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensity")

    plt.tight_layout()
    plt.show()


def plot_3d_surfaces(snrs, sigmas, sens_matrix, fwer_matrix, elev=30, azim=-60):
    """
    Plots Sensitivity and FWER.
    Uses a 'Water Level' colormap to show the intersection at 0.05 perfectly.
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
    
    # 2. Setup Plot
    X, Y = np.meshgrid(snrs, sigmas)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. Plot Surface
    surf = ax.plot_surface(X, Y, net_matrix, cmap='RdYlGn', edgecolor='none', alpha=0.9)
    
    # 4. Labels
    ax.set_title("Net Performance Metric\n(Sensitivity minus FWER)")
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Smoothing Sigma")
    ax.set_zlabel("Net Score")
    
    ax.set_zlim(-1, 1)
    
    # 5. Add a transparent plane at Z=0 (The "Break-even" point)
    zero_plane = np.zeros_like(X)
    ax.plot_surface(X, Y, zero_plane, color='gray', alpha=0.2)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Score (1=Perfect, 0=Useless, <0=Noise)")
    
    plt.tight_layout()
    plt.show()