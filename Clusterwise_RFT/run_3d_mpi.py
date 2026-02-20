import helper_functions as hf
import importlib
importlib.reload(hf)
from mpi4py import MPI
import time
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--threshold_u",      type=float, default=4.4)
parser.add_argument("--overlap_threshold",type=float, default=0)
parser.add_argument("--img_side_length",  type=int,   default=32)
parser.add_argument("--no_simulation",    type=int,   default=100)
parser.add_argument("--method",           type=str,   default="cauchy")
args = parser.parse_args()

# ── Logging (rank 0 only) ─────────────────────────────────────────────────────
class _Tee:
    """Mirrors writes to two streams simultaneously."""
    def __init__(self, primary, secondary):
        self._primary   = primary
        self._secondary = secondary
    def write(self, data):
        self._primary.write(data)
        self._secondary.write(data)
        self._secondary.flush()
    def flush(self):
        self._primary.flush()
        self._secondary.flush()
    # Delegate everything else (e.g. fileno) to the primary stream
    def __getattr__(self, name):
        return getattr(self._primary, name)

_log_file = None
if MPI.COMM_WORLD.Get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    log_path = (f"logs/3D/run_3d_overlap_t{args.overlap_threshold}"
                f"_s{args.no_simulation}"
                f"_u{args.threshold_u}"
                f"_{args.method}.log")
    _log_file = open(log_path, "w", buffering=1)   # line-buffered
    sys.stderr = _Tee(sys.__stderr__, _log_file)    # tqdm → stderr + log
    sys.stdout = _Tee(sys.__stdout__, _log_file)    # print → stdout + log
    print(f"Log: {log_path}")
    print(f"Parameters: threshold_u={args.threshold_u}, overlap_threshold={args.overlap_threshold}, "
          f"img_side_length={args.img_side_length}, no_simulation={args.no_simulation}, method={args.method}")
# ─────────────────────────────────────────────────────────────────────────────

start = time.perf_counter()
results = hf.clusterwise_RFT_Full_Test_3D_speedup(
    # sm_sigma_list    = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    # snr_list         = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    sm_sigma_list    = [1.5],
    snr_list         = [3],
    n_subj_list      = [2, 5, 7, 10, 15, 20, 30, 45, 60],
    threshold_u      = args.threshold_u,
    overlap_threshold= args.overlap_threshold,
    img_side_length  = args.img_side_length,
    no_simulation    = args.no_simulation,
    method           = args.method
)

elapsed = time.perf_counter() - start

if MPI.COMM_WORLD.Get_rank() == 0:
    print(results)
    fname = (f"./Output/3D/results_3d_overlap_t{args.overlap_threshold}"
             f"_s{args.no_simulation}"
             f"_u{args.threshold_u}"
             f"_{args.method}_df.csv")
    results.to_csv(fname, index=False)
    print(f"Saved to {fname}")
    print(f"Total runtime: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    if _log_file:
        _log_file.close()
