import numpy as np
import helper_functions as hf

# Test: Does computed FWHM match true FWHM?
np.random.seed(42)

test_sigmas = [1.0, 2.0, 3.0, 4.0]
n_subj = 50
img_side = 128

print("FWHM Diagnostic Test")
print("=" * 70)
print(f"{'Smoothing Sigma':<20} {'True FWHM':<15} {'Computed FWHM':<15} {'Ratio':<10}")
print("-" * 70)

for sigma in test_sigmas:
    # True FWHM from Gaussian kernel
    true_fwhm = np.sqrt(8 * np.log(2)) * sigma

    # Computed FWHM from simulated data
    fwhm, tmap, res_map, df = hf.simulate_one_tmap(
        n_subj=n_subj,
        img_side_length=img_side,
        smoothing_sigma=sigma,
        snr=0,
        signal_radius=0
    )

    ratio = fwhm / true_fwhm

    print(f"{sigma:<20.2f} {true_fwhm:<15.3f} {fwhm:<15.3f} {ratio:<10.3f}")

print("=" * 70)
print("\nExpected ratio: ~1.0 if FWHM computation is correct")
print("If ratio > 1: Computed FWHM is overestimated")
print("If ratio < 1: Computed FWHM is underestimated")
