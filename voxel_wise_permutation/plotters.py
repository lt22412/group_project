import numpy as np

import matplotlib.pyplot as plt



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

