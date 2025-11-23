import numpy as np
from numpy.linalg import pinv
from scipy.ndimage import label
from skimage.measure import regionprops

def fit_glm(design, Y):

    n = design.shape[0]
    shape = Y.shape[1:]

    #flatten so easier to deal with
    Y_flat = Y.reshape(n, -1)

    #beta = (XTX)^-1 XTY
    XtX_inv = np.linalg.inv(design.T @ design)
    betas = XtX_inv @ design.T @ Y_flat

    # Residuals
    resid = Y_flat - design @ betas

    # Restore shapes
    betas = betas.reshape(betas.shape[0], *shape)
    resid = resid.reshape((n,) + shape)

    return betas, resid
# going to return a beata for every voxel


def tmap_from_L(design, betas, resid, L):

    n, p = design.shape
    df = n - p                       #degrees of freedom
    L = np.asarray(L).reshape(p,)    # contrast vector L

    #flatten residuals
    resid_flat = resid.reshape(n, -1)

    #estimate variance using the rss/df
    rss = np.sum(resid_flat ** 2, axis=0)
    sigma2 = rss / df

    XtX_inv = pinv(design.T @ design)     #(XTX)^-1
    var_L = float(L @ XtX_inv @ L.T)         # L(XTX)^-1LT

    betas_flat = betas.reshape(p, -1)
    effect = L @ betas_flat                  #numerator of t stat LB

    se = np.sqrt(sigma2 * var_L)             # dinominator of t stat sqrt(sigma^2L(XTX)^-1LT)

    #calculate t stat first bit stops program from breaking if inf
    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = effect / se                  # (V,)

    # Reshape back to (X, Y)
    return tvals.reshape(betas.shape[1:])

def get_clusters_and_mask(tmap, threshold):

    #check if any of the t values in the t map pass the threshold and
    mask = (tmap > threshold).astype(np.int32)# astype change changes the boolean into bianry 0,1

    #each connected componenet gets a uniqe number
    labeled, _ = label(mask)

    #calculate the number of voxels in each cluster
    props = regionprops(labeled)

    #create an array which stores the cluster sizes
    cluster_sizes = [p.area for p in props]

    #output clustersizes a boolean map and a labeld t map
    return cluster_sizes, mask.astype(bool), labeled



def permutation_cluster_test(Y, design, L, threshold, n_permutations=1000, alpha=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    n_subjects = design.shape[0]

    #initalise array to stroe max cluster size for each permutaion to build dist
    max_cluster_sizes = []

    # generate permutations
    for i in range(n_permutations):

        P = np.eye(n_subjects)[rng.permutation(n_subjects)]  # create permutation matrix
        permuted_labels = P @ design[:, 1]  #permute the labels ( multiplying by design)
        permuted_design = np.column_stack([design[:, 0], permuted_labels])

        # fit GLM
        betas, resid = fit_glm(permuted_design, Y)
        tmap = tmap_from_L(permuted_design, betas, resid, L)

        # get clusters
        cluster_sizes, _, _ = get_clusters_and_mask(tmap, threshold)
        if cluster_sizes:
            max_cluster_sizes.append(max(cluster_sizes))
        else:
            max_cluster_sizes.append(0)

    max_cluster_sizes = np.array(max_cluster_sizes)

    # determine critical cluster size
    c = int(np.floor(alpha * n_permutations)) #floor alphaN
    crit_cluster_size = np.sort(max_cluster_sizes)[-c - 1]

    # compute observed clusters
    betas_obs, resid_obs = fit_glm(design, Y)
    tmap_obs = tmap_from_L(design, betas_obs, resid_obs, L)
    cluster_sizes_obs, mask_obs, labeled_obs = get_clusters_and_mask(tmap_obs, threshold)

    # determine which clusters are significant
    sig_clusters_mask = np.zeros_like(labeled_obs, dtype=bool)
    for i, size in enumerate(cluster_sizes_obs):
        if size >= crit_cluster_size:
            sig_clusters_mask[labeled_obs == (i + 1)] = True

    return tmap_obs, labeled_obs, sig_clusters_mask, crit_cluster_size, max_cluster_sizes
