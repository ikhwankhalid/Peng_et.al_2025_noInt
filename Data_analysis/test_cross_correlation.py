"""
Quick test script to verify the updated cross-correlation function works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy import stats

def compute_cross_correlation(x, y, max_lag=10):
    """
    Compute cross-correlation between two time series at multiple lags.
    """
    from scipy.signal import correlate

    x = np.array(x)
    y = np.array(y)

    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < max_lag * 2:
        return None, None, None

    # Standardize the signals (zero mean, unit variance)
    x_std = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # Compute cross-correlation using scipy
    cross_corr = correlate(y_std, x_std, mode='full', method='auto')

    # Normalize by the number of overlapping points at each lag
    n = len(x_std)
    lags_full = np.arange(-n + 1, n)

    # Compute normalization factor
    normalization = np.zeros_like(cross_corr)
    for i, lag in enumerate(lags_full):
        if lag < 0:
            overlap = min(n, n + lag)
        else:
            overlap = min(n, n - lag)
        normalization[i] = overlap

    # Normalize to get correlation coefficients
    correlations_full = cross_corr / (normalization + 1e-10)

    # Extract the desired lag range
    center_idx = len(lags_full) // 2
    lag_indices = np.arange(center_idx - max_lag, center_idx + max_lag + 1)
    lag_indices = lag_indices[(lag_indices >= 0) & (lag_indices < len(lags_full))]

    lags = lags_full[lag_indices]
    correlations = correlations_full[lag_indices]

    # Compute p-values using Fisher z-transformation
    p_values = np.zeros(len(correlations))
    for i, (lag, corr) in enumerate(zip(lags, correlations)):
        if lag < 0:
            n_eff = min(n, n + lag)
        else:
            n_eff = min(n, n - lag)

        if n_eff > 3 and not np.isnan(corr):
            if np.abs(corr) < 0.9999:
                z = 0.5 * np.log((1 + corr) / (1 - corr))
                se = 1 / np.sqrt(n_eff - 3)
                z_stat = z / se
                p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
                p_values[i] = p_val
            else:
                p_values[i] = 0.0
        else:
            p_values[i] = np.nan

    return lags, correlations, p_values


if __name__ == '__main__':
    print("Testing cross-correlation function...\n")

    # Test 1: Autocorrelation (x with itself) - should be 1 at lag 0
    print("Test 1: Autocorrelation")
    np.random.seed(42)
    x = np.random.randn(100)
    lags, corrs, pvals = compute_cross_correlation(x, x, max_lag=10)
    zero_lag_idx = np.where(lags == 0)[0][0]
    print(f"  Correlation at lag 0: {corrs[zero_lag_idx]:.4f} (expected: ~1.0)")
    print(f"  Test 1: {'PASS' if abs(corrs[zero_lag_idx] - 1.0) < 0.01 else 'FAIL'}\n")

    # Test 2: Uncorrelated signals - should be near 0 at all lags
    print("Test 2: Uncorrelated signals")
    x = np.random.randn(100)
    y = np.random.randn(100)
    lags, corrs, pvals = compute_cross_correlation(x, y, max_lag=10)
    max_corr = np.max(np.abs(corrs))
    print(f"  Maximum absolute correlation: {max_corr:.4f} (expected: <0.3)")
    print(f"  Test 2: {'PASS' if max_corr < 0.3 else 'FAIL'}\n")

    # Test 3: Lagged signal - should have peak at specific lag
    print("Test 3: Lagged signal (y is x shifted by 3 steps)")
    x = np.random.randn(100)
    lag_amount = 3
    y = np.roll(x, lag_amount)
    lags, corrs, pvals = compute_cross_correlation(x, y, max_lag=10)
    peak_lag = lags[np.argmax(corrs)]
    print(f"  Peak correlation at lag: {peak_lag} (expected: {lag_amount})")
    print(f"  Correlation at peak: {corrs[np.argmax(corrs)]:.4f}")
    print(f"  Test 3: {'PASS' if abs(peak_lag - lag_amount) <= 1 else 'FAIL'}\n")

    # Test 4: Negatively correlated with lag
    print("Test 4: Negatively correlated signals")
    x = np.random.randn(100)
    y = -x + np.random.randn(100) * 0.1  # Negative correlation with small noise
    lags, corrs, pvals = compute_cross_correlation(x, y, max_lag=10)
    zero_lag_idx = np.where(lags == 0)[0][0]
    print(f"  Correlation at lag 0: {corrs[zero_lag_idx]:.4f} (expected: <-0.8)")
    print(f"  Test 4: {'PASS' if corrs[zero_lag_idx] < -0.8 else 'FAIL'}\n")

    # Visual test: Plot example
    print("Test 5: Visual verification")
    t = np.linspace(0, 10, 200)
    x = np.sin(t)
    y = np.sin(t - 1)  # Lagged sine wave
    lags, corrs, pvals = compute_cross_correlation(x, y, max_lag=20)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, label='x (sin)', alpha=0.7)
    plt.plot(t, y, label='y (lagged sin)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Input Signals')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(lags, corrs, 'o-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title('Cross-correlation Function')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_cross_correlation_output.png', dpi=150)
    print("  Visual test plot saved to: test_cross_correlation_output.png")
    plt.close()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
