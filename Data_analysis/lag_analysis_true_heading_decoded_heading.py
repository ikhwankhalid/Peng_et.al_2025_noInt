"""
Lag Analysis: True Heading and Decoded Heading Temporal Relationships (Circular Correlation)

This script performs comprehensive lag analysis to test temporal/causal relationships
between the animal's true heading direction and the neural network decoded heading
direction using circular correlation methods.

Key Questions:
1. Does the decoded heading at time t predict true heading at t+Δt?
   (Does neural representation lead behavioral heading?)
2. Does true heading at time t predict decoded heading at t+Δt?
   (Does behavioral heading lead neural representation?)
3. What are the optimal time lags for these relationships?

Key Difference from Other Scripts:
- This analyzes absolute heading directions (circular variables)
- Uses circular correlation (Jammalamadaka-Sarma method)
- Previous scripts analyzed rates of change (linear variables with standard correlation)

Variables:
- True heading: hdPose (animal's actual movement direction from position data)
- Decoded heading: hdPose - mvtDirError (movement direction inferred from neural activity)
- Both are circular variables (angles in radians, -π to π)

Analysis Methods:
- Circular-circular correlation at multiple time lags (Jammalamadaka-Sarma method)
- Component-wise residualization for speed covariate control
- Permutation testing for significance thresholds
- Per-trial optimal lag analysis

Statistical Approach:
- Uses Jammalamadaka-Sarma circular correlation coefficient
- Properly handles circular/periodic nature of angular data
- Speed control via residualization of sin/cos components

Data Source:
- Uses reconstruction data from `reconstuctionDFAutoPI.csv`
- Analyzes within-trial temporal dynamics

References:
- Jammalamadaka, S. R., & Sarma, Y. (1988). A correlation coefficient for angular variables.
- Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.ndimage
from scipy import stats
from tqdm import tqdm
import warnings

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style('whitegrid')
except (ImportError, AttributeError) as e:
    HAS_SEABORN = False
    print(f"Warning: Seaborn not available ({e}). Using matplotlib defaults.")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


# ============================================================================
# CONFIGURATION
# ============================================================================

# Setup paths
PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'

# Sessions to use
useAble = [
    'jp486-19032023-0108', 'jp486-18032023-0108',
    'jp3269-28112022-0108', 'jp486-16032023-0108',
    'jp452-25112022-0110', 'jp486-24032023-0108',
    'jp486-22032023-0108', 'jp452-24112022-0109',
    'jp486-15032023-0108', 'jp3120-25052022-0107',
    'jp3120-26052022-0107', 'jp451-28102022-0108',
    'jp486-20032023-0108', 'jp486-06032023-0108',
    'jp486-26032023-0108', 'jp486-17032023-0108',
    'jp451-29102022-0108', 'jp451-30102022-0108',
    'jp486-10032023-0108', 'jp486-05032023-0108',
    'jp3269-29112022-0108', 'mn8578-17122021-0107',
    'jp452-23112022-0108', 'jp1686-26042022-0108'
]

# Lag analysis parameters
MAX_LAG = 80  # Maximum lag in timesteps
MIN_TRIAL_LENGTH = 50  # Minimum trial length for analysis
N_PERMUTATIONS = 1000  # Number of permutations for significance testing

# Speed control parameters
CONTROL_FOR_SPEED = True  # If True, use residualization to control for speed effects
MIN_VALID_POINTS = 50  # Minimum number of valid points after residualization


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_speed(x, y, time):
    """Calculate instantaneous speed from position and time data."""
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(time)
    dt[dt == 0] = np.nan
    distance = np.sqrt(dx**2 + dy**2)
    speed_vals = distance / dt
    speed = np.full(len(x), np.nan)
    speed[1:] = speed_vals
    return speed


def circular_mean(angles):
    """
    Calculate circular mean of angles.

    Parameters
    ----------
    angles : array-like
        Angles in radians

    Returns
    -------
    mean_angle : float
        Circular mean in radians
    """
    angles = np.array(angles)
    valid = ~np.isnan(angles)
    if np.sum(valid) == 0:
        return np.nan
    angles = angles[valid]
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def jammalamadaka_correlation(alpha, beta):
    """
    Compute Jammalamadaka-Sarma circular-circular correlation coefficient.

    This is the appropriate correlation measure for two circular variables.
    It ranges from -1 to 1, similar to Pearson correlation.

    Parameters
    ----------
    alpha : array-like
        First circular variable in radians
    beta : array-like
        Second circular variable in radians

    Returns
    -------
    rho : float
        Circular correlation coefficient (-1 to 1)
    p_value : float
        Approximate p-value using permutation-based estimate

    References
    ----------
    Jammalamadaka, S. R., & Sarma, Y. (1988). A correlation coefficient
    for angular variables. Statistical Theory and Data Analysis II, 349-364.
    """
    alpha = np.array(alpha)
    beta = np.array(beta)

    # Remove NaN values
    valid = ~(np.isnan(alpha) | np.isnan(beta))
    alpha = alpha[valid]
    beta = beta[valid]

    if len(alpha) < 10:
        return np.nan, np.nan

    # Circular means
    alpha_bar = np.arctan2(np.mean(np.sin(alpha)), np.mean(np.cos(alpha)))
    beta_bar = np.arctan2(np.mean(np.sin(beta)), np.mean(np.cos(beta)))

    # Angular deviations from circular means
    sin_alpha = np.sin(alpha - alpha_bar)
    sin_beta = np.sin(beta - beta_bar)

    # Correlation coefficient
    numerator = np.sum(sin_alpha * sin_beta)
    denominator = np.sqrt(np.sum(sin_alpha**2) * np.sum(sin_beta**2))

    if denominator < 1e-10:
        return np.nan, np.nan

    rho = numerator / denominator

    # Approximate p-value using Fisher z-transformation
    # (This is an approximation; for circular data it's less rigorous)
    n = len(alpha)
    if n > 3 and not np.isnan(rho) and np.abs(rho) < 0.9999:
        z = 0.5 * np.log((1 + rho) / (1 - rho))
        se = 1 / np.sqrt(n - 3)
        z_stat = z / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    else:
        p_value = np.nan

    return rho, p_value


def residualize_variable(y, covariate, min_valid_points=10):
    """
    Remove the linear effect of a covariate from a variable using regression residuals.

    This function regresses y on the covariate and returns the residuals, which
    represent the variation in y that cannot be explained by the covariate.

    Parameters
    ----------
    y : array-like
        Variable to residualize
    covariate : array-like
        Covariate to control for (e.g., speed)
    min_valid_points : int
        Minimum number of valid points required for regression

    Returns
    -------
    residuals : ndarray
        Residualized variable (same length as input, NaN where input was NaN)
    """
    y = np.array(y)
    covariate = np.array(covariate)

    # Find valid data points (non-NaN in both)
    valid_mask = ~(np.isnan(y) | np.isnan(covariate))

    # Initialize output with NaN
    residuals = np.full(len(y), np.nan)

    # Check if we have enough valid points
    if np.sum(valid_mask) < min_valid_points:
        return residuals

    # Extract valid data
    y_valid = y[valid_mask]
    covariate_valid = covariate[valid_mask]

    # Check if covariate has sufficient variance
    if np.std(covariate_valid) < 1e-10:
        # If covariate is essentially constant, return centered y
        residuals[valid_mask] = y_valid - np.mean(y_valid)
        return residuals

    # Fit linear regression: y ~ covariate
    try:
        # Fit: y = a * covariate + b
        coeffs = np.polyfit(covariate_valid, y_valid, deg=1)
        y_predicted = np.polyval(coeffs, covariate_valid)

        # Residuals = observed - predicted
        residuals[valid_mask] = y_valid - y_predicted

    except:
        # If regression fails, return centered y
        residuals[valid_mask] = y_valid - np.mean(y_valid)

    return residuals


def residualize_circular_variable(angles, covariate, min_valid_points=10):
    """
    Residualize circular variable by residualizing sin/cos components separately.

    This preserves circular structure while removing linear covariate effects.
    The approach is to:
    1. Convert angles to sin/cos components
    2. Residualize each component with respect to covariate
    3. Reconstruct angles from residualized components

    Parameters
    ----------
    angles : array-like
        Circular variable in radians
    covariate : array-like
        Covariate to control for (e.g., speed)
    min_valid_points : int
        Minimum number of valid points required

    Returns
    -------
    angles_resid : ndarray
        Residualized circular variable in radians
    """
    angles = np.array(angles)
    covariate = np.array(covariate)

    # Convert to components
    sin_component = np.sin(angles)
    cos_component = np.cos(angles)

    # Residualize each component
    sin_resid = residualize_variable(sin_component, covariate, min_valid_points)
    cos_resid = residualize_variable(cos_component, covariate, min_valid_points)

    # Reconstruct angle from residualized components
    angles_resid = np.arctan2(sin_resid, cos_resid)

    return angles_resid


# ============================================================================
# LAG ANALYSIS FUNCTIONS
# ============================================================================

def compute_circular_cross_correlation(heading1, heading2, max_lag=10):
    """
    Compute circular cross-correlation between two heading time series at multiple lags.

    Uses Jammalamadaka-Sarma circular correlation at each lag separately.

    Parameters
    ----------
    heading1 : array-like
        First heading time series (radians)
    heading2 : array-like
        Second heading time series (radians)
    max_lag : int
        Maximum lag to test (both positive and negative)

    Returns
    -------
    lags : ndarray
        Array of lag values
    correlations : ndarray
        Circular correlation coefficients at each lag
    p_values : ndarray
        P-values for each correlation
    """
    heading1 = np.array(heading1)
    heading2 = np.array(heading2)

    # Remove NaN values
    valid_mask = ~(np.isnan(heading1) | np.isnan(heading2))
    heading1 = heading1[valid_mask]
    heading2 = heading2[valid_mask]

    if len(heading1) < max_lag * 2:
        return None, None, None

    # Create lag array
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros(len(lags))
    p_values = np.zeros(len(lags))

    # Compute circular correlation at each lag
    for i, lag in enumerate(lags):
        if lag > 0:
            # Positive lag: heading1 predicts future heading2
            h1 = heading1[:-lag]
            h2 = heading2[lag:]
        elif lag < 0:
            # Negative lag: heading2 predicts future heading1
            h1 = heading1[-lag:]
            h2 = heading2[:lag]
        else:
            # Zero lag
            h1 = heading1
            h2 = heading2

        # Compute circular correlation
        rho, p_val = jammalamadaka_correlation(h1, h2)
        correlations[i] = rho
        p_values[i] = p_val

    return lags, correlations, p_values


def compute_trial_circular_cross_correlations(trial_data, max_lag=10):
    """
    Compute circular cross-correlations for a single trial.

    Parameters
    ----------
    trial_data : DataFrame
        Trial data with true_heading and decoded_heading columns
    max_lag : int
        Maximum lag to test

    Returns
    -------
    results : dict
        Dictionary containing lag analysis results
    """
    true_heading = trial_data['true_heading'].values
    decoded_heading = trial_data['decoded_heading'].values

    # Remove NaN values
    valid_mask = ~(np.isnan(true_heading) | np.isnan(decoded_heading))
    true_heading = true_heading[valid_mask]
    decoded_heading = decoded_heading[valid_mask]

    if len(true_heading) < max_lag * 2:
        return None

    lags, corrs, pvals = compute_circular_cross_correlation(true_heading, decoded_heading, max_lag)

    if lags is None:
        return None

    return {
        'lags': lags,
        'correlations': corrs,
        'p_values': pvals,
        'n_points': len(true_heading)
    }


def permutation_test_circular_correlation(heading1, heading2, max_lag=10, n_permutations=1000):
    """
    Perform permutation test for circular cross-correlation significance.

    Parameters
    ----------
    heading1 : array-like
        First heading time series
    heading2 : array-like
        Second heading time series
    max_lag : int
        Maximum lag to test
    n_permutations : int
        Number of permutations

    Returns
    -------
    null_distribution : ndarray
        Null distribution of maximum absolute correlations
    p_value : float
        Permutation p-value for observed maximum correlation
    """
    heading1 = np.array(heading1)
    heading2 = np.array(heading2)

    # Remove NaN values
    valid_mask = ~(np.isnan(heading1) | np.isnan(heading2))
    heading1 = heading1[valid_mask]
    heading2 = heading2[valid_mask]

    # Compute observed correlation
    _, obs_corrs, _ = compute_circular_cross_correlation(heading1, heading2, max_lag)
    obs_max_corr = np.max(np.abs(obs_corrs))

    # Permutation test
    null_max_corrs = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle heading2 while preserving heading1
        heading2_shuffled = np.random.permutation(heading2)
        _, perm_corrs, _ = compute_circular_cross_correlation(heading1, heading2_shuffled, max_lag)
        null_max_corrs[i] = np.max(np.abs(perm_corrs))

    # Calculate p-value
    p_value = np.mean(null_max_corrs >= obs_max_corr)

    return null_max_corrs, p_value


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_condition_for_circular_lag_analysis(df, condition, max_lag=10, control_for_speed=True,
                                                 min_valid_points=50):
    """
    Process a condition for circular lag analysis with speed as a covariate.

    This function analyzes the relationship between true heading and decoded heading
    while controlling for the effect of movement speed using component-wise residualization.

    Parameters
    ----------
    df : DataFrame
        Input dataframe
    condition : str
        Condition name
    max_lag : int
        Maximum lag to test
    control_for_speed : bool
        If True, residualize headings with respect to speed
        If False, use raw headings (no covariate control)
    min_valid_points : int
        Minimum number of valid (non-NaN) points required for analysis

    Returns
    -------
    results : dict
        Lag analysis results for the condition
    """
    print(f"\nProcessing {condition}...")
    if control_for_speed:
        print("  Using speed as covariate (component-wise residualization)")
    else:
        print("  Using raw headings (no covariate control)")

    condition_df = df[df.condition == condition].copy()
    condition_df['session_trial'] = condition_df['session'] + '_T' + condition_df['trial'].astype(str)
    unique_trials = condition_df['session_trial'].unique()
    print(f"Found {len(unique_trials)} trials")

    trial_results = []
    optimal_lags = []
    optimal_corrs = []

    for trial_id in tqdm(unique_trials):
        trial_data = condition_df[condition_df['session_trial'] == trial_id].copy()
        trial_data = trial_data.sort_values('recTime')

        if len(trial_data) < MIN_TRIAL_LENGTH:
            continue

        # Extract headings
        # True heading from pose data
        true_heading = trial_data['hdPose'].values

        # Decoded heading = true heading - movement direction error
        # (using circular subtraction)
        mvt_dir_error = trial_data['mvtDirError'].values
        decoded_heading = np.arctan2(
            np.sin(true_heading - mvt_dir_error),
            np.cos(true_heading - mvt_dir_error)
        )

        # Calculate speed
        speed = calculate_speed(
            trial_data['xPose'].values,
            trial_data['yPose'].values,
            trial_data['recTime'].values
        )

        # Control for speed using component-wise residualization
        if control_for_speed:
            # Residualize true heading
            true_heading_resid = residualize_circular_variable(true_heading, speed)
            # Residualize decoded heading
            decoded_heading_resid = residualize_circular_variable(decoded_heading, speed)

            # Use residualized headings for analysis
            trial_data['true_heading'] = true_heading_resid
            trial_data['decoded_heading'] = decoded_heading_resid
        else:
            # Use raw headings
            trial_data['true_heading'] = true_heading
            trial_data['decoded_heading'] = decoded_heading

        # Check for sufficient valid data points
        valid_mask = ~(np.isnan(trial_data['true_heading']) |
                      np.isnan(trial_data['decoded_heading']))

        if np.sum(valid_mask) < min_valid_points:
            continue

        # Compute circular cross-correlations for this trial
        trial_cc = compute_trial_circular_cross_correlations(trial_data, max_lag)

        if trial_cc is not None:
            trial_results.append(trial_cc)

            # Calculate optimal lag for this trial (lag with maximum absolute correlation)
            lags = trial_cc['lags']
            corrs = trial_cc['correlations']
            max_idx = np.argmax(np.abs(corrs))
            optimal_lag = lags[max_idx]
            optimal_corr = corrs[max_idx]

            optimal_lags.append(optimal_lag)
            optimal_corrs.append(optimal_corr)

    if len(trial_results) == 0:
        return None

    # Aggregate across trials
    all_lags = trial_results[0]['lags']
    all_corrs = np.array([r['correlations'] for r in trial_results])

    mean_corrs = np.nanmean(all_corrs, axis=0)
    sem_corrs = np.nanstd(all_corrs, axis=0) / np.sqrt(len(trial_results))

    return {
        'condition': condition,
        'lags': all_lags,
        'mean_correlations': mean_corrs,
        'sem_correlations': sem_corrs,
        'trial_correlations': all_corrs,
        'n_trials': len(trial_results),
        'optimal_lags_per_trial': np.array(optimal_lags),
        'optimal_corrs_per_trial': np.array(optimal_corrs)
    }


def aggregate_circular_lag_analysis(df, conditions, max_lag=10, control_for_speed=True):
    """
    Perform circular lag analysis across all conditions.

    Parameters
    ----------
    df : DataFrame
        Input dataframe
    conditions : list
        List of conditions to analyze
    max_lag : int
        Maximum lag to test
    control_for_speed : bool
        If True, control for speed using component-wise residualization
        If False, use raw headings without covariate control

    Returns
    -------
    results : dict
        Dictionary mapping conditions to lag analysis results
    """
    results = {}

    for condition in conditions:
        result = process_condition_for_circular_lag_analysis(df, condition, max_lag,
                                                              control_for_speed=control_for_speed)
        if result is not None:
            results[condition] = result

    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def calculate_median_timestep(df, n_samples=1000):
    """
    Calculate the median timestep duration from the data.

    Parameters
    ----------
    df : DataFrame
        Input dataframe with recTime column
    n_samples : int
        Number of random trials to sample for estimation

    Returns
    -------
    median_dt : float
        Median timestep duration in seconds
    """
    # Sample random trials
    df_copy = df.copy()
    df_copy['session_trial'] = df_copy['session'] + '_T' + df_copy['trial'].astype(str)
    unique_trials = df_copy['session_trial'].unique()

    if len(unique_trials) > n_samples:
        sampled_trials = np.random.choice(unique_trials, n_samples, replace=False)
    else:
        sampled_trials = unique_trials

    all_dts = []
    for trial_id in sampled_trials:
        trial_data = df_copy[df_copy['session_trial'] == trial_id].copy()
        trial_data = trial_data.sort_values('recTime')
        if len(trial_data) > 1:
            dt = np.diff(trial_data['recTime'].values)
            all_dts.extend(dt[dt > 0])  # Only positive time differences

    median_dt = np.median(all_dts)
    print(f"Median timestep duration: {median_dt:.4f} seconds ({1/median_dt:.2f} Hz)")

    return median_dt


def plot_circular_cross_correlation_function(results, condition, timestep_duration=1.0):
    """
    Plot circular cross-correlation function for a condition.

    Parameters
    ----------
    results : dict
        Lag analysis results
    condition : str
        Condition name
    timestep_duration : float
        Duration of one timestep in seconds

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lags = results['lags']
    lags_sec = lags * timestep_duration  # Convert to seconds
    mean_corrs = results['mean_correlations']
    sem_corrs = results['sem_correlations']

    # Plot correlation function
    ax.plot(lags_sec, mean_corrs, 'o-', linewidth=2, markersize=8, label='Mean correlation')
    ax.fill_between(lags_sec, mean_corrs - sem_corrs, mean_corrs + sem_corrs,
                     alpha=0.3, label='SEM')

    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # Formatting
    ax.set_xlabel('Lag (seconds)', fontsize=12)
    ax.set_ylabel('Circular Correlation Coefficient', fontsize=12)
    ax.set_title(f'Circular Cross-Correlation: True Heading ↔ Decoded Heading\n{condition}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    textstr = 'Negative lag: Decoded heading predicts future true heading\n'
    textstr += 'Positive lag: True heading predicts future decoded heading'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_lag_heatmap(all_results, conditions, timestep_duration=1.0):
    """
    Create heatmap of correlations across conditions and lags.

    Parameters
    ----------
    all_results : dict
        Dictionary of lag analysis results for all conditions
    conditions : list
        List of condition names
    timestep_duration : float
        Duration of one timestep in seconds

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    # Prepare data matrix
    lags = all_results[conditions[0]]['lags']
    lags_sec = lags * timestep_duration  # Convert to seconds
    corr_matrix = np.zeros((len(conditions), len(lags)))

    for i, condition in enumerate(conditions):
        if condition in all_results:
            corr_matrix[i, :] = all_results[condition]['mean_correlations']
        else:
            corr_matrix[i, :] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-0.3, vmax=0.3, interpolation='nearest')

    # Set ticks - show every few lags to avoid crowding
    tick_indices = np.arange(0, len(lags), max(1, len(lags) // 10))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{lags_sec[i]:.2f}' for i in tick_indices])
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(conditions)

    # Labels
    ax.set_xlabel('Lag (seconds)', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    ax.set_title('Circular Cross-Correlation Heatmap: True Heading ↔ Decoded Heading',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Circular Correlation Coefficient', fontsize=11)

    # Add vertical line at lag=0
    ax.axvline(x=len(lags)//2, color='black', linestyle='--', linewidth=2, alpha=0.7)

    plt.tight_layout()
    return fig


def plot_optimal_lags(all_results, conditions, timestep_duration=1.0):
    """
    Plot optimal lags (maximum absolute correlation) for each condition.

    Parameters
    ----------
    all_results : dict
        Dictionary of lag analysis results
    conditions : list
        List of condition names
    timestep_duration : float
        Duration of one timestep in seconds

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    optimal_lags = []
    max_corrs = []
    condition_labels = []

    for condition in conditions:
        if condition in all_results:
            result = all_results[condition]
            lags = result['lags']
            corrs = result['mean_correlations']

            # Find lag with maximum absolute correlation
            max_idx = np.argmax(np.abs(corrs))
            optimal_lag = lags[max_idx] * timestep_duration  # Convert to seconds
            max_corr = corrs[max_idx]

            optimal_lags.append(optimal_lag)
            max_corrs.append(max_corr)
            condition_labels.append(condition)

    # Plot 1: Optimal lag values
    colors = ['blue' if lag < 0 else 'red' if lag > 0 else 'gray' for lag in optimal_lags]
    ax1.barh(range(len(condition_labels)), optimal_lags, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(condition_labels)))
    ax1.set_yticklabels(condition_labels)
    ax1.set_xlabel('Optimal Lag (seconds)', fontsize=12)
    ax1.set_title('Optimal Lag for Each Condition', fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Maximum correlation values
    colors = ['blue' if corr < 0 else 'red' for corr in max_corrs]
    ax2.barh(range(len(condition_labels)), max_corrs, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(condition_labels)))
    ax2.set_yticklabels(condition_labels)
    ax2.set_xlabel('Maximum |Correlation|', fontsize=12)
    ax2.set_title('Maximum Correlation at Optimal Lag', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_individual_trials(results, condition, n_trials=5, timestep_duration=1.0):
    """
    Plot circular cross-correlations for individual trials.

    Parameters
    ----------
    results : dict
        Lag analysis results
    condition : str
        Condition name
    n_trials : int
        Number of trials to plot
    timestep_duration : float
        Duration of one timestep in seconds

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lags = results['lags']
    lags_sec = lags * timestep_duration  # Convert to seconds
    trial_corrs = results['trial_correlations']

    # Plot individual trials
    n_plot = min(n_trials, len(trial_corrs))
    for i in range(n_plot):
        ax.plot(lags_sec, trial_corrs[i], alpha=0.3, linewidth=1)

    # Plot mean
    mean_corrs = results['mean_correlations']
    ax.plot(lags_sec, mean_corrs, 'k-', linewidth=3, label='Mean')

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # Formatting
    ax.set_xlabel('Lag (seconds)', fontsize=12)
    ax.set_ylabel('Circular Correlation Coefficient', fontsize=12)
    ax.set_title(f'Individual Trial Circular Cross-Correlations\n{condition}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_optimal_lags_violin(all_results, conditions, timestep_duration=1.0):
    """
    Create violin plot showing the distribution of optimal lags per trial for each condition.

    Parameters
    ----------
    all_results : dict
        Dictionary of lag analysis results
    conditions : list
        List of condition names
    timestep_duration : float
        Duration of one timestep in seconds

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    # Prepare data for plotting
    data_by_condition = {}

    for condition in conditions:
        if condition in all_results:
            result = all_results[condition]
            if 'optimal_lags_per_trial' in result:
                optimal_lags = result['optimal_lags_per_trial']
                # Convert to seconds
                optimal_lags_sec = optimal_lags * timestep_duration
                data_by_condition[condition] = optimal_lags_sec

    if len(data_by_condition) == 0:
        print("Warning: No data available for violin plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    if HAS_SEABORN:
        # Use seaborn if available
        data_for_plot = []
        for condition, lags in data_by_condition.items():
            for lag in lags:
                data_for_plot.append({
                    'Condition': condition,
                    'Optimal Lag (s)': lag
                })
        plot_df = pd.DataFrame(data_for_plot)

        sns.violinplot(data=plot_df, x='Condition', y='Optimal Lag (s)',
                       ax=ax, palette='Set2', inner='box')
    else:
        # Use matplotlib box plot as fallback
        positions = range(len(data_by_condition))
        data_list = [data_by_condition[cond] for cond in data_by_condition.keys()]

        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True)

        # Color the boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(data_list)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(data_by_condition.keys())
        ax.set_ylabel('Optimal Lag (seconds)', fontsize=12, fontweight='bold')

    # Add reference line at lag=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero lag')

    # Formatting
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimal Lag (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Optimal Lags Per Trial\nTrue Heading ↔ Decoded Heading',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add interpretation text
    textstr = 'Negative lag: Decoded heading predicts future true heading\n'
    textstr += 'Positive lag: True heading predicts future decoded heading'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

def generate_statistical_summary(all_results, conditions):
    """
    Generate statistical summary table.

    Parameters
    ----------
    all_results : dict
        Dictionary of lag analysis results
    conditions : list
        List of condition names

    Returns
    -------
    summary_df : DataFrame
        Summary statistics table
    """
    summary_data = []

    for condition in conditions:
        if condition in all_results:
            result = all_results[condition]
            lags = result['lags']
            corrs = result['mean_correlations']

            # Find optimal lag
            max_idx = np.argmax(np.abs(corrs))
            optimal_lag = lags[max_idx]
            max_corr = corrs[max_idx]

            # Zero-lag correlation
            zero_lag_idx = len(lags) // 2
            zero_lag_corr = corrs[zero_lag_idx]

            # Mean absolute correlation
            mean_abs_corr = np.mean(np.abs(corrs))

            summary_data.append({
                'Condition': condition,
                'N_Trials': result['n_trials'],
                'Optimal_Lag': optimal_lag,
                'Max_Correlation': max_corr,
                'Zero_Lag_Corr': zero_lag_corr,
                'Mean_Abs_Corr': mean_abs_corr
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Load data
    print("="*80)
    print("LAG ANALYSIS: True Heading ↔ Decoded Heading (Circular Correlation)")
    print("="*80)
    print("\nLoading reconstruction data...")
    fn = os.path.join(PROJECT_DATA_PATH, "results", "reconstuctionDFAutoPI.csv")
    dfAutoPI = pd.read_csv(fn)
    print(f"Loaded {len(dfAutoPI)} rows")

    # Filter for useable sessions
    dfAutoPI = dfAutoPI[dfAutoPI.session.isin(useAble)]
    print(f"After filtering: {len(dfAutoPI)} rows")
    print(f"Sessions: {dfAutoPI.session.nunique()}")

    # Calculate median timestep duration
    print("\n" + "="*80)
    print("CALCULATING TIMESTEP DURATION")
    print("="*80)
    timestep_duration = calculate_median_timestep(dfAutoPI, n_samples=1000)

    # Define conditions
    conditions = [
        'all_light',
        'all_dark',
        'searchToLeverPath_light',
        'searchToLeverPath_dark',
        'homingFromLeavingLever_light',
        'homingFromLeavingLever_dark',
        'atLever_light',
        'atLever_dark'
    ]

    # Perform circular lag analysis
    print("\n" + "="*80)
    print("PERFORMING CIRCULAR LAG ANALYSIS")
    if CONTROL_FOR_SPEED:
        print("Method: Speed covariate control (component-wise residualization)")
        print("  - Preserves temporal continuity")
        print("  - Controls for linear effect of speed on sin/cos components")
    else:
        print("Method: Raw headings (no covariate control)")
    print("="*80)
    all_results = aggregate_circular_lag_analysis(dfAutoPI, conditions, max_lag=MAX_LAG,
                                                   control_for_speed=CONTROL_FOR_SPEED)

    # Generate statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    summary_df = generate_statistical_summary(all_results, conditions)
    print("\n", summary_df.to_string(index=False))

    # Save summary
    summary_path = os.path.join(PROJECT_DATA_PATH, 'results', 'lag_analysis_true_decoded_heading_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    results_dir = os.path.join(PROJECT_DATA_PATH, 'results')

    # 1. Individual circular cross-correlation plots
    for condition in conditions:
        if condition in all_results:
            print(f"\nGenerating circular cross-correlation plot for {condition}...")
            fig = plot_circular_cross_correlation_function(all_results[condition], condition, timestep_duration)
            fig.savefig(os.path.join(results_dir, f'lag_heading_circular_ccf_{condition}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

    # 2. Heatmap across conditions
    print("\nGenerating heatmap...")
    fig = plot_lag_heatmap(all_results, conditions, timestep_duration)
    fig.savefig(os.path.join(results_dir, 'lag_heading_circular_heatmap_all_conditions.png'),
               dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Optimal lags plot
    print("\nGenerating optimal lags plot...")
    fig = plot_optimal_lags(all_results, conditions, timestep_duration)
    fig.savefig(os.path.join(results_dir, 'lag_heading_circular_optimal_lags.png'),
               dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4. Individual trial examples
    for condition in ['all_light', 'all_dark']:
        if condition in all_results:
            print(f"\nGenerating individual trials plot for {condition}...")
            fig = plot_individual_trials(all_results[condition], condition, n_trials=10, timestep_duration=timestep_duration)
            fig.savefig(os.path.join(results_dir, f'lag_heading_circular_individual_trials_{condition}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

    # 5. Violin plot of optimal lags per trial
    print("\nGenerating violin plot of optimal lags per trial...")
    fig = plot_optimal_lags_violin(all_results, conditions, timestep_duration)
    if fig is not None:
        fig.savefig(os.path.join(results_dir, 'lag_heading_circular_optimal_lags_violin.png'),
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Save per-trial optimal lags data
    print("\nSaving per-trial optimal lags data...")
    optimal_lags_data = []
    for condition in conditions:
        if condition in all_results:
            result = all_results[condition]
            if 'optimal_lags_per_trial' in result:
                optimal_lags = result['optimal_lags_per_trial']
                optimal_corrs = result['optimal_corrs_per_trial']
                optimal_lags_sec = optimal_lags * timestep_duration

                for i, (lag, corr) in enumerate(zip(optimal_lags_sec, optimal_corrs)):
                    optimal_lags_data.append({
                        'condition': condition,
                        'trial_index': i,
                        'optimal_lag_steps': optimal_lags[i],
                        'optimal_lag_seconds': lag,
                        'optimal_correlation': corr
                    })

    optimal_lags_df = pd.DataFrame(optimal_lags_data)
    optimal_lags_path = os.path.join(results_dir, 'lag_heading_circular_optimal_lags_per_trial.csv')
    optimal_lags_df.to_csv(optimal_lags_path, index=False)
    print(f"Per-trial optimal lags saved to: {optimal_lags_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nInterpretation Guide:")
    print("- Negative lag: Decoded heading at time t predicts true heading at t+Δt")
    print("  (Neural representation leads behavior)")
    print("- Positive lag: True heading at time t predicts decoded heading at t+Δt")
    print("  (Behavior leads neural representation)")
    print("- Zero lag: Simultaneous relationship between variables")
    print("\nMethodological Notes:")
    print("- Uses circular correlation (Jammalamadaka-Sarma method) for angular data")
    print("- Analyzes absolute heading directions (not rates of change)")
    print("- Decoded heading = true heading - movement direction error")
    if CONTROL_FOR_SPEED:
        print("- Speed control: Component-wise residualization of sin/cos components")
        print("- Preserves temporal continuity and circular structure")
    else:
        print("- Speed control: None (raw headings used)")
    print("\nNew Outputs:")
    print("- Violin plot showing per-trial optimal lag distributions by condition")
    print("- Per-trial optimal lags saved to CSV for further analysis")
    print("\nResults saved to:", results_dir)
