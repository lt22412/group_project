import helper_functions as hf
import importlib
importlib.reload(hf)
from mpi4py import MPI
import time

start = time.perf_counter()
results = hf.clusterwise_RFT_Full_Test_3D_speedup(
    #sm_sigma_list    = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    #snr_list         = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    sm_sigma_list    = [1.5],
    snr_list         = [3],
    n_subj_list      = [2, 5, 7, 10, 15, 20, 30, 45, 60],
    threshold_u=4.4,
    overlap_threshold=0,
    img_side_length=32,
    no_simulation=100,
    method="cauchy"
)

elapsed = time.perf_counter() - start


if MPI.COMM_WORLD.Get_rank() == 0:
    print(results)
    results.to_csv("results_3d_mpi.csv", index=False)
    print(f"Total runtime: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
