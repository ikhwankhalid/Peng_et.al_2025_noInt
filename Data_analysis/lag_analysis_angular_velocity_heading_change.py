"""
Lag Analysis: Angular Velocity and Heading Deviation CHANGE Temporal Relationships

This script performs comprehensive lag analysis to test temporal/causal relationships
between angular velocity (turning rate) and the CHANGE in heading deviation (rate of
navigation error accumulation/correction).

UPDATES:
- Modified to use scipy.signal.correlate for proper cross-correlation computation
- Added per-trial optimal lag calculation and violin plot visualization
- Changed from speed filtering to speed covariate control (residualization)
  * Preserves temporal continuity of time series
  * Controls for speed effects without creating gaps in data
  * Lags now measured in actual time steps, not filtered samples

Key Questions:
1. Does the change in heading deviation at time t predict angular velocity at t+Δt?
   (Does the rate of error accumulation/correction predict turning?)
2. Does angular velocity at time t predict change in heading deviation at t+Δt?
   (Does turning predict the rate of subsequent error accumulation/correction?)
3. What are the optimal time lags for these relationships?

Key Difference from Original Script:
- Original: angular velocity ↔ heading deviation (absolute error)
- This script: angular velocity ↔ CHANGE in heading deviation (rate of error change)

Analysis Methods:
- Cross-correlation analysis at multiple time lags using scipy.signal.correlate
  (both angular velocity and change in heading deviation are linear variables)
- Granger causality tests for directional prediction
- Lagged regression models with statistical controls
- Permutation testing for significance thresholds
- Autocorrelation-corrected statistics

Statistical Approach:
- Uses normalized cross-correlation (scipy.signal.correlate)
- Angular velocity is the rate of turning (rad/s)
- Change in heading deviation is the rate of error change (rad/s)
- Signals are standardized before correlation computation

Data Source:
- Uses reconstruction data from `reconstuctionDFAutoPI.csv`
- Analyzes within-trial temporal dynamics

References:
- Granger, C. W. (1969). Investigating causal relations by econometric models.
- Box, G. E. P., & Jenkins, G. M. (1976). Time series analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.ndimage
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# Set plotting style
sns.set_style('whitegrid')
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
                          # If False, use raw variables (old approach with filtering)
MIN_VALID_POINTS = 50  # Minimum number of valid points after residualization


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_heading_from_position(x, y, smooth_window=0.1):
    """Calculate instantaneous heading from position data."""
    if smooth_window > 1:
        x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma=smooth_window/3)
        y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma=smooth_window/3)
    else:
        x_smooth, y_smooth = x, y

    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    heading = np.full(len(x), np.nan)
    heading[1:] = np.arctan2(dy, dx)
    return heading


def calculate_angular_velocity(heading, time, smooth_window=1.):
    """Calculate angular velocity from heading and time."""
    dheading = np.diff(heading)
    dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
    dt = np.diff(time)
    dt[dt == 0] = np.nan
    angular_vel = dheading / dt
    angular_velocity = np.full(len(heading), np.nan)
    angular_velocity[1:] = angular_vel

    if smooth_window > 0:
        valid_mask = ~np.isnan(angular_velocity)
        if np.sum(valid_mask) > 0:
            angular_velocity[valid_mask] = scipy.ndimage.gaussian_filter1d(
                angular_velocity[valid_mask], sigma=smooth_window)
    return angular_velocity


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


def calculate_heading_deviation_change(heading_deviation, time, smooth_window=0.5):
    """
    Calculate the change (derivative) in heading deviation over time.

    This represents the rate at which navigation error is accumulating or
    being corrected. Like angular velocity, this is a linear variable representing
    a rate of change in an angular quantity.

    Parameters
    ----------
    heading_deviation : array-like
        Heading deviation in radians (circular variable)
    time : array-like
        Time values
    smooth_window : float
        Smoothing window for Gaussian filter (0 for no smoothing)

    Returns
    -------
    heading_dev_change : ndarray
        Rate of change in heading deviation (rad/s)
    """
    heading_deviation = np.array(heading_deviation)
    time = np.array(time)

    # Calculate angular difference between consecutive heading deviations
    # Use arctan2 to handle circular wraparound properly
    d_heading_dev = np.diff(heading_deviation)
    d_heading_dev = np.arctan2(np.sin(d_heading_dev), np.cos(d_heading_dev))

    # Calculate time differences
    dt = np.diff(time)
    dt[dt == 0] = np.nan

    # Calculate rate of change
    heading_dev_change_vals = d_heading_dev / dt

    # Create output array with same length as input
    heading_dev_change = np.full(len(heading_deviation), np.nan)
    heading_dev_change[1:] = heading_dev_change_vals

    # Optional smoothing
    if smooth_window > 0:
        valid_mask = ~np.isnan(heading_dev_change)
        if np.sum(valid_mask) > 0:
            heading_dev_change[valid_mask] = scipy.ndimage.gaussian_filter1d(
                heading_dev_change[valid_mask], sigma=smooth_window)

    return heading_dev_change


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
    # Using numpy polyfit for simple linear regression
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


# ============================================================================
# LAG ANALYSIS FUNCTIONS
# ============================================================================

def compute_cross_correlation(x, y, max_lag=10):
    """
    Compute cross-correlation between two time series at multiple lags.

    Uses scipy.signal.correlate for efficient computation of normalized
    cross-correlation coefficients.

    Parameters
    ----------
    x : array-like
        First time series (angular velocity)
    y : array-like
        Second time series (change in heading deviation)
    max_lag : int
        Maximum lag to test (both positive and negative)

    Returns
    -------
    lags : ndarray
        Array of lag values
    correlations : ndarray
        Normalized cross-correlation coefficients at each lag
    p_values : ndarray
        P-values for each correlation (using Fisher z-transformation)
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
    # mode='full' gives all possible overlaps
    cross_corr = correlate(y_std, x_std, mode='full', method='auto')

    # Normalize by the number of overlapping points at each lag
    n = len(x_std)
    lags_full = np.arange(-n + 1, n)

    # Compute normalization factor (number of overlapping samples at each lag)
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

    # Make sure indices are valid
    lag_indices = lag_indices[(lag_indices >= 0) & (lag_indices < len(lags_full))]

    lags = lags_full[lag_indices]
    correlations = correlations_full[lag_indices]

    # Compute p-values using Fisher z-transformation
    # For cross-correlation, approximate degrees of freedom
    p_values = np.zeros(len(correlations))
    for i, (lag, corr) in enumerate(zip(lags, correlations)):
        if lag < 0:
            n_eff = min(n, n + lag)
        else:
            n_eff = min(n, n - lag)

        if n_eff > 3 and not np.isnan(corr):
            # Fisher z-transformation
            if np.abs(corr) < 0.9999:  # Avoid log(0)
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


def compute_trial_cross_correlations(trial_data, max_lag=10):
    """
    Compute cross-correlations for a single trial.

    Parameters
    ----------
    trial_data : DataFrame
        Trial data with angular_velocity and heading_dev_change columns
    max_lag : int
        Maximum lag to test

    Returns
    -------
    results : dict
        Dictionary containing lag analysis results
    """
    ang_vel = trial_data['angular_velocity'].values
    heading_dev_change = trial_data['heading_dev_change'].values

    # Remove NaN values
    valid_mask = ~(np.isnan(ang_vel) | np.isnan(heading_dev_change))
    ang_vel = ang_vel[valid_mask]
    heading_dev_change = heading_dev_change[valid_mask]

    if len(ang_vel) < max_lag * 2:
        return None

    lags, corrs, pvals = compute_cross_correlation(ang_vel, heading_dev_change, max_lag)

    if lags is None:
        return None

    return {
        'lags': lags,
        'correlations': corrs,
        'p_values': pvals,
        'n_points': len(ang_vel)
    }


def permutation_test_cross_correlation(x, y, max_lag=10, n_permutations=1000):
    """
    Perform permutation test for cross-correlation significance.

    Parameters
    ----------
    x : array-like
        First time series
    y : array-like
        Second time series
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
    x = np.array(x)
    y = np.array(y)

    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    # Compute observed correlation
    _, obs_corrs, _ = compute_cross_correlation(x, y, max_lag)
    obs_max_corr = np.max(np.abs(obs_corrs))

    # Permutation test
    null_max_corrs = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle y while preserving x
        y_shuffled = np.random.permutation(y)
        _, perm_corrs, _ = compute_cross_correlation(x, y_shuffled, max_lag)
        null_max_corrs[i] = np.max(np.abs(perm_corrs))

    # Calculate p-value
    p_value = np.mean(null_max_corrs >= obs_max_corr)

    return null_max_corrs, p_value


def lagged_regression_analysis(x, y, lag, add_controls=True):
    """
    Perform lagged regression analysis.

    Parameters
    ----------
    x : array-like
        Predictor variable
    y : array-like
        Outcome variable
    lag : int
        Lag to apply (positive means x predicts future y)
    add_controls : bool
        Whether to add autoregressive controls

    Returns
    -------
    results : dict
        Regression results including coefficients and statistics
    """
    x = np.array(x)
    y = np.array(y)

    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    if lag < 0:
        # Negative lag: x leads y
        x_lagged = x[:lag]
        y_lagged = y[-lag:]
    elif lag > 0:
        # Positive lag: y leads x
        x_lagged = x[lag:]
        y_lagged = y[:-lag]
    else:
        # Zero lag
        x_lagged = x
        y_lagged = y

    if len(x_lagged) < 10:
        return None

    # Prepare regression
    X = x_lagged.reshape(-1, 1)

    if add_controls and lag != 0:
        # Add autoregressive term for y
        if lag > 0:
            y_ar = y[:-lag]
        else:
            y_ar = y[-lag:]
        X = np.column_stack([X, y_ar[:-1] if lag > 0 else y_ar[1:]])

    X = add_constant(X)

    # Fit model
    try:
        model = OLS(y_lagged, X).fit()

        return {
            'lag': lag,
            'coef': model.params[1],  # Coefficient for x
            'se': model.bse[1],
            't_stat': model.tvalues[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared,
            'n_obs': len(y_lagged)
        }
    except:
        return None


def granger_causality_analysis(x, y, max_lag=5):
    """
    Perform Granger causality test.

    Parameters
    ----------
    x : array-like
        First time series
    y : array-like
        Second time series
    max_lag : int
        Maximum lag to test

    Returns
    -------
    results : dict
        Granger causality test results
    """
    # Prepare data
    data = pd.DataFrame({'x': x, 'y': y})
    data = data.dropna()

    if len(data) < max_lag * 10:
        return None

    try:
        # Test if x Granger-causes y
        gc_results = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)

        # Extract p-values for each lag
        results = {}
        for lag in range(1, max_lag + 1):
            # Use F-test p-value
            p_value = gc_results[lag][0]['ssr_ftest'][1]
            results[f'lag_{lag}'] = p_value

        return results
    except:
        return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_condition_for_lag_analysis(df, condition, max_lag=10, control_for_speed=True,
                                       min_valid_points=50):
    """
    Process a condition for lag analysis with speed as a covariate.

    This function analyzes the relationship between angular velocity and heading
    deviation change while controlling for the effect of movement speed. Instead
    of filtering out low-speed samples (which creates gaps in the time series),
    it uses residualization to remove the linear effect of speed from both variables.

    Parameters
    ----------
    df : DataFrame
        Input dataframe
    condition : str
        Condition name
    max_lag : int
        Maximum lag to test
    control_for_speed : bool
        If True, residualize angular velocity and heading dev change with respect to speed
        If False, use raw variables (no covariate control)
    min_valid_points : int
        Minimum number of valid (non-NaN) points required for analysis

    Returns
    -------
    results : dict
        Lag analysis results for the condition
    """
    print(f"\nProcessing {condition}...")
    if control_for_speed:
        print("  Using speed as covariate (residualization approach)")
    else:
        print("  Using raw variables (no covariate control)")

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

        # Calculate variables
        heading = calculate_heading_from_position(trial_data['x'].values, trial_data['y'].values)
        angular_velocity = calculate_angular_velocity(heading, trial_data['recTime'].values)
        speed = calculate_speed(trial_data['x'].values, trial_data['y'].values, trial_data['recTime'].values)

        # Calculate change in heading deviation
        heading_dev_change = calculate_heading_deviation_change(
            trial_data['mvtDirError'].values,
            trial_data['recTime'].values
        )

        trial_data['heading'] = heading
        trial_data['speed'] = speed

        # Control for speed using residualization
        if control_for_speed:
            # Residualize angular velocity: remove linear effect of speed
            angular_velocity_resid = residualize_variable(angular_velocity, speed)
            # Residualize heading dev change: remove linear effect of speed
            heading_dev_change_resid = residualize_variable(heading_dev_change, speed)

            # Use residualized variables for analysis
            trial_data['angular_velocity'] = angular_velocity_resid
            trial_data['heading_dev_change'] = heading_dev_change_resid
        else:
            # Use raw variables
            trial_data['angular_velocity'] = angular_velocity
            trial_data['heading_dev_change'] = heading_dev_change

        # Check for sufficient valid data points
        # NaN filtering will happen in compute_trial_cross_correlations
        valid_mask = ~(np.isnan(trial_data['angular_velocity']) |
                      np.isnan(trial_data['heading_dev_change']))

        if np.sum(valid_mask) < min_valid_points:
            continue

        # Compute cross-correlations for this trial (uses all data, not filtered by speed)
        trial_cc = compute_trial_cross_correlations(trial_data, max_lag)

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


def aggregate_lag_analysis(df, conditions, max_lag=10, control_for_speed=True):
    """
    Perform lag analysis across all conditions.

    Parameters
    ----------
    df : DataFrame
        Input dataframe
    conditions : list
        List of conditions to analyze
    max_lag : int
        Maximum lag to test
    control_for_speed : bool
        If True, control for speed using residualization (default: True)
        If False, use raw variables without covariate control

    Returns
    -------
    results : dict
        Dictionary mapping conditions to lag analysis results
    """
    results = {}

    for condition in conditions:
        result = process_condition_for_lag_analysis(df, condition, max_lag,
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


def plot_cross_correlation_function(results, condition, timestep_duration=1.0):
    """
    Plot cross-correlation function for a condition.

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
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title(f'Cross-Correlation: Angular Velocity ↔ Change in Heading Deviation\n{condition}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    textstr = 'Negative lag: Change in heading dev. predicts future angular velocity\n'
    textstr += 'Positive lag: Angular velocity predicts future change in heading dev.'
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
    ax.set_title('Cross-Correlation Heatmap: Angular Velocity ↔ Change in Heading Deviation',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11)

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
    Plot cross-correlations for individual trials.

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
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title(f'Individual Trial Cross-Correlations\n{condition}',
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
    # Prepare data for violin plot
    data_for_plot = []

    for condition in conditions:
        if condition in all_results:
            result = all_results[condition]
            if 'optimal_lags_per_trial' in result:
                optimal_lags = result['optimal_lags_per_trial']
                # Convert to seconds
                optimal_lags_sec = optimal_lags * timestep_duration

                # Add to plotting data
                for lag in optimal_lags_sec:
                    data_for_plot.append({
                        'Condition': condition,
                        'Optimal Lag (s)': lag
                    })

    # Create DataFrame for seaborn
    plot_df = pd.DataFrame(data_for_plot)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create violin plot
    sns.violinplot(data=plot_df, x='Condition', y='Optimal Lag (s)',
                   ax=ax, palette='Set2', inner='box')

    # Add reference line at lag=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero lag')

    # Formatting
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimal Lag (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Optimal Lags Per Trial\nAngular Velocity ↔ Change in Heading Deviation',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add interpretation text
    textstr = 'Negative lag: Change in heading dev. predicts future angular velocity\n'
    textstr += 'Positive lag: Angular velocity predicts future change in heading dev.'
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
    print("LAG ANALYSIS: Angular Velocity ↔ CHANGE in Heading Deviation")
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

    # Perform lag analysis
    print("\n" + "="*80)
    print("PERFORMING LAG ANALYSIS")
    if CONTROL_FOR_SPEED:
        print("Method: Speed covariate control (residualization)")
        print("  - Preserves temporal continuity")
        print("  - Controls for linear effect of speed")
    else:
        print("Method: Raw variables (no covariate control)")
    print("="*80)
    all_results = aggregate_lag_analysis(dfAutoPI, conditions, max_lag=MAX_LAG,
                                         control_for_speed=CONTROL_FOR_SPEED)

    # Generate statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    summary_df = generate_statistical_summary(all_results, conditions)
    print("\n", summary_df.to_string(index=False))

    # Save summary
    summary_path = os.path.join(PROJECT_DATA_PATH, 'results', 'lag_analysis_heading_change_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    results_dir = os.path.join(PROJECT_DATA_PATH, 'results')

    # 1. Individual cross-correlation plots
    for condition in conditions:
        if condition in all_results:
            print(f"\nGenerating cross-correlation plot for {condition}...")
            fig = plot_cross_correlation_function(all_results[condition], condition, timestep_duration)
            fig.savefig(os.path.join(results_dir, f'lag_heading_change_ccf_{condition}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

    # 2. Heatmap across conditions
    print("\nGenerating heatmap...")
    fig = plot_lag_heatmap(all_results, conditions, timestep_duration)
    fig.savefig(os.path.join(results_dir, 'lag_heading_change_heatmap_all_conditions.png'),
               dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Optimal lags plot
    print("\nGenerating optimal lags plot...")
    fig = plot_optimal_lags(all_results, conditions, timestep_duration)
    fig.savefig(os.path.join(results_dir, 'lag_heading_change_optimal_lags.png'),
               dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4. Individual trial examples
    for condition in ['all_light', 'all_dark']:
        if condition in all_results:
            print(f"\nGenerating individual trials plot for {condition}...")
            fig = plot_individual_trials(all_results[condition], condition, n_trials=10, timestep_duration=timestep_duration)
            fig.savefig(os.path.join(results_dir, f'lag_heading_change_individual_trials_{condition}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

    # 5. Violin plot of optimal lags per trial
    print("\nGenerating violin plot of optimal lags per trial...")
    fig = plot_optimal_lags_violin(all_results, conditions, timestep_duration)
    fig.savefig(os.path.join(results_dir, 'lag_heading_change_optimal_lags_violin.png'),
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
    optimal_lags_path = os.path.join(results_dir, 'lag_heading_change_optimal_lags_per_trial.csv')
    optimal_lags_df.to_csv(optimal_lags_path, index=False)
    print(f"Per-trial optimal lags saved to: {optimal_lags_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nInterpretation Guide:")
    print("- Negative lag: Change in heading dev. at time t predicts angular velocity at t+Δt")
    print("  (Rate of error accumulation/correction causes turning)")
    print("- Positive lag: Angular velocity at time t predicts change in heading dev. at t+Δt")
    print("  (Turning causes subsequent rate of error accumulation/correction)")
    print("- Zero lag: Simultaneous relationship between variables")
    print("\nMethodological Updates:")
    print("- This analysis examines the RATE of change in navigation error")
    print("- Original script examined the ABSOLUTE navigation error")
    if CONTROL_FOR_SPEED:
        print("- Speed control: Residualization (preserves temporal continuity)")
        print("- Lags measured in actual time steps, not filtered samples")
    else:
        print("- Speed control: Filtering approach (creates gaps in time series)")
    print("\nNew Outputs:")
    print("- Violin plot showing per-trial optimal lag distributions by condition")
    print("- Per-trial optimal lags saved to CSV for further analysis")
    print("\nResults saved to:", results_dir)
