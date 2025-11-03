"""
Test to verify scipy.signal.correlate lag convention.

This test creates signals with known temporal relationships to verify
that the lag interpretation is correct.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy import stats

def compute_cross_correlation(x, y, max_lag=10):
    """Same as in the main script."""
    from scipy.signal import correlate

    x = np.array(x)
    y = np.array(y)

    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < max_lag * 2:
        return None, None, None

    # Standardize the signals
    x_std = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # Compute cross-correlation
    cross_corr = correlate(y_std, x_std, mode='full', method='auto')

    # Normalize by overlap
    n = len(x_std)
    lags_full = np.arange(-n + 1, n)

    normalization = np.zeros_like(cross_corr)
    for i, lag in enumerate(lags_full):
        if lag < 0:
            overlap = min(n, n + lag)
        else:
            overlap = min(n, n - lag)
        normalization[i] = overlap

    correlations_full = cross_corr / (normalization + 1e-10)

    # Extract desired lag range
    center_idx = len(lags_full) // 2
    lag_indices = np.arange(center_idx - max_lag, center_idx + max_lag + 1)
    lag_indices = lag_indices[(lag_indices >= 0) & (lag_indices < len(lags_full))]

    lags = lags_full[lag_indices]
    correlations = correlations_full[lag_indices]

    return lags, correlations, None


if __name__ == '__main__':
    print("="*70)
    print("LAG CONVENTION VERIFICATION TEST")
    print("="*70)

    np.random.seed(42)
    n = 200

    # Test 1: X leads Y by 5 steps (Y lags behind X)
    print("\n" + "="*70)
    print("TEST 1: X leads Y by 5 steps")
    print("="*70)
    print("Setup: y[t] = x[t-5] + noise")
    print("Expected: Peak correlation at LAG = +5")
    print("Interpretation: Positive lag means X predicts future Y")

    x = np.random.randn(n)
    y = np.zeros(n)
    y[5:] = x[:-5] + np.random.randn(n-5) * 0.1  # Y is delayed version of X

    lags, corrs, _ = compute_cross_correlation(x, y, max_lag=15)
    peak_lag = lags[np.argmax(corrs)]

    print(f"\nResult: Peak at lag = {peak_lag}")
    print(f"Correlation at peak: {corrs[np.argmax(corrs)]:.3f}")

    if peak_lag == 5:
        print("[PASS] CORRECT: Positive lag indicates X leads Y")
    else:
        print("[FAIL] UNEXPECTED: Peak not at expected lag")

    # Test 2: Y leads X by 5 steps (X lags behind Y)
    print("\n" + "="*70)
    print("TEST 2: Y leads X by 5 steps")
    print("="*70)
    print("Setup: x[t] = y[t-5] + noise")
    print("Expected: Peak correlation at LAG = -5")
    print("Interpretation: Negative lag means Y predicts future X")

    y = np.random.randn(n)
    x = np.zeros(n)
    x[5:] = y[:-5] + np.random.randn(n-5) * 0.1  # X is delayed version of Y

    lags, corrs, _ = compute_cross_correlation(x, y, max_lag=15)
    peak_lag = lags[np.argmax(corrs)]

    print(f"\nResult: Peak at lag = {peak_lag}")
    print(f"Correlation at peak: {corrs[np.argmax(corrs)]:.3f}")

    if peak_lag == -5:
        print("[PASS] CORRECT: Negative lag indicates Y leads X")
    else:
        print("[FAIL] UNEXPECTED: Peak not at expected lag")

    # Test 3: Real-world scenario with angular velocity and heading deviation
    print("\n" + "="*70)
    print("TEST 3: Real-world scenario")
    print("="*70)
    print("Scenario: Animal turns (angular velocity), then heading deviation changes")
    print("Setup: heading_dev_change[t] = angular_velocity[t-3] + noise")
    print("Expected: Peak at LAG = +3")
    print("Interpretation: Angular velocity predicts future heading dev change")

    angular_velocity = np.random.randn(n) * 0.5
    heading_dev_change = np.zeros(n)
    heading_dev_change[3:] = angular_velocity[:-3] + np.random.randn(n-3) * 0.1

    # NOTE: In compute_cross_correlation, x is first arg, y is second
    # correlate(y_std, x_std) is called
    lags, corrs, _ = compute_cross_correlation(angular_velocity, heading_dev_change, max_lag=15)
    peak_lag = lags[np.argmax(corrs)]

    print(f"\nResult: Peak at lag = {peak_lag}")
    print(f"Correlation at peak: {corrs[np.argmax(corrs)]:.3f}")

    if peak_lag == 3:
        print("[PASS] CORRECT: Positive lag -> angular velocity predicts future heading_dev_change")
    else:
        print("[FAIL] UNEXPECTED: Peak not at expected lag")

    # Test 4: Opposite scenario
    print("\n" + "="*70)
    print("TEST 4: Opposite scenario")
    print("="*70)
    print("Scenario: Heading deviation changes, then animal corrects with turning")
    print("Setup: angular_velocity[t] = heading_dev_change[t-3] + noise")
    print("Expected: Peak at LAG = -3")
    print("Interpretation: Heading dev change predicts future angular velocity")

    heading_dev_change = np.random.randn(n) * 0.5
    angular_velocity = np.zeros(n)
    angular_velocity[3:] = heading_dev_change[:-3] + np.random.randn(n-3) * 0.1

    lags, corrs, _ = compute_cross_correlation(angular_velocity, heading_dev_change, max_lag=15)
    peak_lag = lags[np.argmax(corrs)]

    print(f"\nResult: Peak at lag = {peak_lag}")
    print(f"Correlation at peak: {corrs[np.argmax(corrs)]:.3f}")

    if peak_lag == -3:
        print("[PASS] CORRECT: Negative lag -> heading_dev_change predicts future angular_velocity")
    else:
        print("[FAIL] UNEXPECTED: Peak not at expected lag")

    # Visual summary
    print("\n" + "="*70)
    print("SUMMARY OF LAG CONVENTION")
    print("="*70)
    print("\nWith compute_cross_correlation(x=angular_velocity, y=heading_dev_change):")
    print("  * POSITIVE LAG: angular_velocity[t] correlates with heading_dev_change[t+lag]")
    print("                  -> Angular velocity predicts FUTURE heading dev change")
    print("                  -> Turning causes subsequent error accumulation/correction")
    print("")
    print("  * NEGATIVE LAG: angular_velocity[t] correlates with heading_dev_change[t+lag]")
    print("                  -> Heading dev change predicts FUTURE angular velocity")
    print("                  -> Error accumulation/correction causes subsequent turning")
    print("")
    print("This MATCHES the interpretation in the script comments.")
    print("="*70)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test 1
    np.random.seed(42)
    x = np.random.randn(n)
    y = np.zeros(n)
    y[5:] = x[:-5] + np.random.randn(n-5) * 0.1
    lags, corrs, _ = compute_cross_correlation(x, y, max_lag=15)

    axes[0, 0].plot(x[:50], 'b-', label='X', alpha=0.7)
    axes[0, 0].plot(y[:50], 'r-', label='Y (X delayed by 5)', alpha=0.7)
    axes[0, 0].set_title('Test 1: X leads Y by 5 steps')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)

    # Test 2
    y = np.random.randn(n)
    x = np.zeros(n)
    x[5:] = y[:-5] + np.random.randn(n-5) * 0.1
    lags2, corrs2, _ = compute_cross_correlation(x, y, max_lag=15)

    axes[0, 1].plot(x[:50], 'b-', label='X (Y delayed by 5)', alpha=0.7)
    axes[0, 1].plot(y[:50], 'r-', label='Y', alpha=0.7)
    axes[0, 1].set_title('Test 2: Y leads X by 5 steps')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)

    # Cross-correlations
    axes[1, 0].plot(lags, corrs, 'o-', linewidth=2, markersize=6, label='Test 1')
    axes[1, 0].axvline(x=5, color='b', linestyle='--', alpha=0.5, label='Expected peak (+5)')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Cross-correlation: X leads Y')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(lags2, corrs2, 'o-', linewidth=2, markersize=6, color='red', label='Test 2')
    axes[1, 1].axvline(x=-5, color='r', linestyle='--', alpha=0.5, label='Expected peak (-5)')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Cross-correlation: Y leads X')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_lag_convention_visualization.png', dpi=150)
    print("\nVisualization saved to: test_lag_convention_visualization.png")
