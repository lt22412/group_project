import numpy as np
import matplotlib.pyplot as plt
from rft_voxelwise import simulate_null_data, voxelwise_rft_threshold, estimate_fwer_rft
from scipy import stats


def test_fwer_single_condition(n_runs=100, n_subj=20, img_size=64, sigma=1.5,
                                alpha=0.05, labels=False, ndim=2, n_thresh=50,
                                noise="normal"):
    
    print(f"Testing: n={n_subj}, sigma={sigma}, alpha={alpha}, {n_runs} runs...")

    any_sig = np.zeros(n_runs, dtype=bool)
    max_t_values = np.zeros(n_runs)
    thresholds = np.zeros(n_runs)

    for r in range(n_runs):
        data = simulate_null_data(
            n_subj=n_subj, img_size=img_size,
            sigma=sigma, ndim=ndim, snr=0, signal_radius=0, noise=noise
        )

        tmap, thr = voxelwise_rft_threshold(data, labels=labels, alpha=alpha, n_thresh=n_thresh)

        max_t_values[r] = np.abs(tmap).max()
        thresholds[r] = thr
        any_sig[r] = np.any(np.abs(tmap) > thr)

    empirical_fwer = np.mean(any_sig)

    print(f"FWER: {empirical_fwer:.4f} (target: {alpha:.4f})\n")

    return empirical_fwer, max_t_values, thresholds


def test_fwer_across_alphas(n_runs=100, n_subj=20, img_size=64, sigma=1.5,
                            alphas=[0.01, 0.05, 0.10], labels=False, ndim=2):
    
    results = {}

    for alpha in alphas:
        empirical_fwer, max_t, thrs = test_fwer_single_condition(
            n_runs=n_runs, n_subj=n_subj, img_size=img_size,
            sigma=sigma, alpha=alpha, labels=labels, ndim=ndim
        )
        results[alpha] = {
            'empirical_fwer': empirical_fwer,
            'max_t': max_t,
            'thresholds': thrs,
        }

    return results


def test_fwer_across_conditions(n_runs=100, n_subj=20, alpha=0.05,
                                labels=False, ndim=2):
    
    conditions = [
        {'name': 'Small image, low smoothing',   'img_size': 32, 'sigma': 1.0, 'n_subj': 20},
        {'name': 'Medium image, medium smoothing','img_size': 64, 'sigma': 1.5, 'n_subj': 20},
        {'name': 'Large image, high smoothing',  'img_size': 96, 'sigma': 2.0, 'n_subj': 20},
        {'name': 'Medium image, small sample',   'img_size': 64, 'sigma': 1.5, 'n_subj': 10},
        {'name': 'Medium image, large sample',   'img_size': 64, 'sigma': 1.5, 'n_subj': 40},
    ]

    results = {}

    for condition in conditions:
        print(f"Condition: {condition['name']}")

        empirical_fwer, max_t, thrs = test_fwer_single_condition(
            n_runs=n_runs,
            n_subj=condition['n_subj'],
            img_size=condition['img_size'],
            sigma=condition['sigma'],
            alpha=alpha,
            labels=labels,
            ndim=ndim
        )

        results[condition['name']] = {
            'empirical_fwer': empirical_fwer,
            'max_t': max_t,
            'thresholds': thrs,
            'params': condition,
        }

    return results


def plot_fwer_results(results_dict, alpha=0.05, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(results_dict.keys())
    fwers = [results_dict[name]['empirical_fwer'] for name in names]
    max_ts = [results_dict[name]['max_t'] for name in names]
    thresholds = [results_dict[name]['thresholds'] for name in names]

    ax = axes[0, 0]
    x_pos = np.arange(len(names))
    ax.bar(x_pos, fwers, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'Nominal α = {alpha}')
    ax.axhline(y=alpha - 0.01, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=alpha + 0.01, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('Empirical FWER', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title('FWER Control Across Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    positions = np.arange(len(names))
    bp = ax.boxplot(max_ts, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Max |t| value', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title('Distribution of Maximum |t| Values', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    bp2 = ax.boxplot(thresholds, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax.set_ylabel('RFT Threshold', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title('RFT Thresholds Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        ax.scatter(thresholds[i], max_ts[i], alpha=0.5, s=30, color=colors[i], label=name)

    all_thrs = np.concatenate(thresholds)
    all_maxts = np.concatenate(max_ts)
    lim_min = min(all_thrs.min(), all_maxts.min())
    lim_max = max(all_thrs.max(), all_maxts.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, label='Perfect calibration')
    ax.set_xlabel('RFT Threshold', fontsize=12)
    ax.set_ylabel('Max |t|', fontsize=12)
    ax.set_title('Max |t| vs RFT Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_qq_plot(max_t_values, thresholds, alpha=0.05, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mean_thr = thresholds.mean()

    ax = axes[0]
    ax.hist(max_t_values, bins=30, density=True, alpha=0.7, label='Empirical')
    ax.axvline(mean_thr, color='red', linestyle='--', linewidth=2,
               label=f'Mean threshold = {mean_thr:.2f}')
    ax.axvline(np.percentile(max_t_values, (1-alpha)*100), color='orange',
               linestyle='--', linewidth=2,
               label=f'Empirical {(1-alpha)*100:.0f}% quantile')
    ax.set_xlabel('Max |t| value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Max |t| Under Null', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    sorted_max_t = np.sort(max_t_values)
    exceedance_prob = 1 - np.arange(1, len(sorted_max_t) + 1) / len(sorted_max_t)
    ax.plot(sorted_max_t, exceedance_prob, 'o-', alpha=0.6, label='Empirical')
    ax.axhline(alpha, color='red', linestyle='--', linewidth=2, label=f'Target α = {alpha}')
    ax.axvline(mean_thr, color='orange', linestyle='--', linewidth=2,
               label=f'Mean threshold = {mean_thr:.2f}')
    ax.set_xlabel('Max |t| threshold', fontsize=12)
    ax.set_ylabel('P(Max |t| > threshold)', fontsize=12)
    ax.set_title('Exceedance Probability', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def compute_confidence_interval(fwer, n_runs, confidence=0.95):
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n_runs
    center = (fwer + z**2 / (2 * n_runs)) / denominator
    margin = z * np.sqrt(fwer * (1 - fwer) / n_runs + z**2 / (4 * n_runs**2)) / denominator
    return center - margin, center + margin


def print_summary_table(results_dict, alpha=0.05, n_runs=100):
    print(f"\n{'='*90}")
    print(f"FWER SUMMARY (α = {alpha})")
    print(f"{'='*90}")
    print(f"{'Condition':<35} {'FWER':>8} {'95% CI':>20} {'Threshold':>10} {'Status':>10}")
    print(f"{'-'*90}")

    for name, data in results_dict.items():
        fwer = data['empirical_fwer']
        ci_lower, ci_upper = compute_confidence_interval(fwer, n_runs)
        mean_thr = data['thresholds'].mean()
        within_ci = ci_lower <= alpha <= ci_upper
        status = "PASS" if within_ci else "FAIL"
        print(f"{name:<35} {fwer:>8.4f} [{ci_lower:>6.4f}, {ci_upper:>6.4f}] {mean_thr:>10.3f} {status:>10}")

    print(f"{'='*90}\n")


def run_quick_test():
    return test_fwer_single_condition(
        n_runs=20, n_subj=20, img_size=32, sigma=1.5, alpha=0.05, labels=False, ndim=2
    )


def run_standard_test():
    fwer, max_t, thrs = test_fwer_single_condition(
        n_runs=100, n_subj=20, img_size=64, sigma=1.5, alpha=0.05, labels=False, ndim=2
    )

    results = {'Standard test': {'empirical_fwer': fwer, 'max_t': max_t, 'thresholds': thrs}}
    plot_fwer_results(results, alpha=0.05, save_path="fwer_barplot.png")
    plot_qq_plot(max_t, thrs, alpha=0.05, save_path="fwer_qqplot.png")

    return fwer, max_t, thrs


def run_comprehensive_test():
    results = test_fwer_across_conditions(n_runs=100, n_subj=20, alpha=0.05, labels=False, ndim=2)
    print_summary_table(results, alpha=0.05, n_runs=100)
    plot_fwer_results(results, alpha=0.05, save_path="fwer_comprehensive.png")

    return results


def run_alpha_sweep_test():
    results = test_fwer_across_alphas(
        n_runs=100, n_subj=20, img_size=64, sigma=1.5, alphas=[0.01, 0.05, 0.10], labels=False, ndim=2
    )

    print(f"\nAlpha Sweep Results:")
    print(f"{'Nominal α':>12} {'FWER':>16} {'Threshold':>10} {'Diff':>12}")
    print("-"*52)
    for alpha, data in results.items():
        fwer = data['empirical_fwer']
        thr = data['thresholds'].mean()
        print(f"{alpha:>12.3f} {fwer:>16.4f} {thr:>10.4f} {fwer - alpha:>12.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    alphas_list = list(results.keys())
    fwers = [results[a]['empirical_fwer'] for a in alphas_list]
    ax.plot(alphas_list, alphas_list, 'r--', linewidth=2, label='Perfect control')
    ax.plot(alphas_list, fwers, 'o-', markersize=10, linewidth=2, label='Empirical FWER')
    ax.fill_between(alphas_list, [a - 0.01 for a in alphas_list],
                    [a + 0.01 for a in alphas_list],
                    alpha=0.2, color='red', label='±0.01 tolerance')
    ax.set_xlabel('Nominal α', fontsize=14)
    ax.set_ylabel('Empirical FWER', fontsize=14)
    ax.set_title('FWER Control Across Alpha Levels', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fwer_alpha_sweep.png", dpi=300, bbox_inches='tight')

    return results


if __name__ == "__main__":
    run_standard_test()
