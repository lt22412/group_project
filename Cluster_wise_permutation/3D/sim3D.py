import numpy as np
from scipy.ndimage import label, gaussian_filter

def simulate_3d_data(
        n_subjects=20,
        shape=(64, 64, 32),
        sigma=1.5,
        mode="null",  # "null" or "group"
        signal_region=(slice(20, 30), slice(20, 30), slice(12, 20)),
        signal_strength=1.5,
        random_state=None):


    rng = np.random.default_rng(random_state)
    X, Y, Z = shape

    #we generate the random data
    data = rng.standard_normal((n_subjects, X, Y, Z))

    #we smooth it as fmri data is more like this as voxels close to eachother will have similar variance
    for i in range(n_subjects):
        data[i] = gaussian_filter(data[i], sigma=sigma, mode="constant")

    #split the 2 groups
    n1 = n_subjects // 2
    n2 = n_subjects - n1
    group = np.array([0] * n1 + [1] * n2)

    #intercept of all ones
    intercept = np.ones(n_subjects)

    design = np.column_stack([intercept, group])

    #we add a signal at the specified area
    if mode == "group":
        data[group == 1,signal_region[0],signal_region[1],signal_region[2]] += signal_strength

    return data, design


def fit_glm(design, Y):
    # get dimention of design matix
    n = design.shape[0]

    # keep the original shape of Y and then flatten
    shape = Y.shape[1:]
    Y_flat = Y.reshape(n, -1)

    # comute (XTX)**-1*XTY = Beta
    XtX_inv = np.linalg.inv(design.T @ design)
    betas = XtX_inv @ design.T @ Y_flat

    # calculate residuals
    resid = Y_flat - design @ betas

    # want to keep original shape of data so we reshape to original shape
    betas = betas.reshape(betas.shape[0], *shape)
    resid = resid.reshape((n,) + shape)

    return betas, resid