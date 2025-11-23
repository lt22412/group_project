import numpy as np
from scipy.ndimage import gaussian_filter

def simulate_2d_data(
        n_subjects=20,
        shape=(64, 64),
        sigma=1.5,
        mode="null",
        signal_region=(slice(20, 30), slice(20, 30)),
        signal_strength=1.5,
        random_state=None):

    rng = np.random.default_rng(random_state)
    X, Y = shape

    #random data
    data = rng.standard_normal((n_subjects, X, Y))

    #assign groups
    group = np.zeros(n_subjects, dtype=int)
    group[n_subjects // 2:] = 1

    #add signal
    if mode == "group":
        idx = np.ix_(np.where(group == 1)[0],
            range(signal_region[0].start, signal_region[0].stop),
            range(signal_region[1].start, signal_region[1].stop),
        )
        data[idx] += signal_strength

    #smooth as fmri data is typically like this as neighboring voxels will be similar variance
    for i in range(n_subjects):
        data[i] = gaussian_filter(data[i], sigma=sigma, mode="constant")

    design = np.column_stack([np.ones(n_subjects), group])

    return data, design
