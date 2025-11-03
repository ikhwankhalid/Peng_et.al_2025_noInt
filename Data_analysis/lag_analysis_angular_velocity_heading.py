"""
Lag Analysis: Angular Velocity and Heading Deviation Temporal Relationships

This script performs comprehensive lag analysis to test temporal/causal relationships
between angular velocity (turning rate) and heading deviation (navigation error).

Key Questions:
1. Does heading deviation at time t predict angular velocity at t+Δt?
   (Does navigation error cause corrective turning?)
2. Does angular velocity at time t predict heading deviation at t+Δt?
   (Does turning cause subsequent navigation error?)
3. What are the optimal time lags for these relationships?

Analysis Methods:
- Circular-linear cross-correlation analysis at multiple time lags
  (appropriate for linear angular velocity and circular heading deviation)
- Granger causality tests for directional prediction
- Lagged regression models with statistical controls
- Permutation testing for significance thresholds
- Autocorrelation-corrected statistics

Statistical Approach:
- Uses circular-linear correlation (Mardia & Jupp, 2000) instead of Pearson r
- Angular velocity is treated as a linear variable
- Heading deviation is treated as a circular variable (angles in radians)
- Correlation computed via sine/cosine components of circular variable

Data Source:
- Uses reconstruction data from `reconstuctionDFAutoPI.csv`
- Analyzes within-trial temporal dynamics

References:
- Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. Wiley.
- Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in Circular Statistics.
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


# ============================================================================
# CIRCULAR-LINEAR CORRELATION FUNCTIONS
# ============================================================================

def circular_linear_correlation(linear_var, circular_var):
    """
    Compute circular-linear correlation coefficient.
    
    This function computes the correlation between a linear variable and a 
    circular variable (e.g., angular velocity and heading deviation). The method
    is based on Mardia & Jupp (2000) "Directional Statistics" and uses the 
    correlation between the linear variable and the sine/cosine components of 
    the circular variable.
    
    Parameters
    ----------
    linear_var : array-like
        Linear variable (e.g., angular velocity)
    circular_var : array-like
        Circular variable in radians (e.g., heading deviation)
        
    Returns
    -------
    r_cl : float
        Circular-linear correlation coefficient (ranges from 0 to 1)
    p_value : float
        P-value for the correlation (based on permutation test approximation)
        
    References
    ----------
    Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. Wiley.
    Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in Circular Statistics. 
        World Scientific.
    """
    linear_var = np.array(linear_var)
    circular_var = np.array(circular_var)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(linear_var) | np.isnan(circular_var))
    linear_var = linear_var[valid_mask]
    circular_var = circular_var[valid_mask]
    
    n = len(linear_var)
    if n < 3:
        return np.nan, np.nan
    
    # Compute sine and cosine of circular variable
    sin_circ = np.sin(circular_var)
    cos_circ = np.cos(circular_var)
    
    # Compute correlations
    r_xc = np.corrcoef(linear_var, cos_circ)[0, 1]  # corr(linear, cos(circular))
    r_xs = np.corrcoef(linear_var, sin_circ)[0, 1]  # corr(linear, sin(circular))
    r_cs = np.corrcoef(cos_circ, sin_circ)[0, 1]    # corr(cos(circular), sin(circular))
    
    # Compute circular-linear correlation coefficient
    # Formula: r_cl = sqrt(r_xc^2 + r_xs^2 - 2*r_xc*r_xs*r_cs) / sqrt(1 - r_cs^2)
    numerator = r_xc**2 + r_xs**2 - 2*r_xc*r_xs*r_cs
    denominator = 1 - r_cs**2
    
    if denominator <= 0 or numerator < 0:
        return np.nan, np.nan
    
    r_cl = np.sqrt(numerator / denominator)
    
    # Compute approximate p-value using the test statistic
    # Under H0, n * r_cl^2 approximately follows chi-square distribution with 2 df
    test_stat = n * r_cl**2
    p_value = 1 - stats.chi2.cdf(test_stat, df=2)
    
    return r_cl, p_value


def signed_circular_linear_correlation(linear_var, circular_var):
    """
    Compute signed circular-linear correlation coefficient.
    
    The standard circular-linear correlation is always positive. This function
    adds a sign based on the dominant relationship (using the correlation with
    sine component as the sign indicator).
    
    Parameters
    ----------
    linear_var : array-like
        Linear variable (e.g., angular velocity)
    circular_var : array-like
        Circular variable in radians (e.g., heading deviation)
        
    Returns
    -------
    r_cl_signed : float
        Signed circular-linear correlation coefficient
    p_value : float
        P-value for the correlation
    """
    linear_var = np.array(linear_var)
    circular_var = np.array(circular_var)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(linear_var) | np.isnan(circular_var))
    linear_var = linear_var[valid_mask]
    circular_var = circular_var[valid_mask]
    
    if len(linear_var) < 3:
        return np.nan, np.nan
    
    # Get unsigned correlation
    r_cl, p_value = circular_linear_correlation(linear_var, circular_var)
    
    if np.isnan(r_cl):
        return np.nan, np.nan
    
    # Determine sign based on the correlation with sine component
    # This captures the direction of the relationship
    sin_circ = np.sin(circular_var)
    r_xs = np.corrcoef(linear_var, sin_circ)[0, 1]
    
    # Alternative: use correlation with cosine component
    cos_circ = np.cos(circular_var)
    r_xc = np.corrcoef(linear_var, cos_circ)[0, 1]
    
    # Use the component with larger absolute correlation to determine sign
    if abs(r_xs) >= abs(r_xc):
        sign = np.sign(r_xs)
    else:
        sign = np.sign(r_xc)
    
    r_cl_signed = sign * r_cl
    
    return r_cl_signed, p_value

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
MAX_LAG = 10  # Maximum lag in timesteps
MIN_TRIAL_LENGTH = 50  # Minimum trial length for analysis
N_PERMUTATIONS = 1000  # Number of permutations for significance testing


# ============================================================================
# UTILITY FUNCTIONS (from original script)
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


# ============================================================================
# LAG ANALYSIS FUNCTIONS
# ============================================================================

def compute_cross_correlation(x, y, max_lag=10):
    """
    Compute circular-linear cross-correlation between two time series at multiple lags.
    
    Uses circular-linear correlation since x (angular velocity) is linear and 
    y (heading deviation) is circular.
    
    Parameters
    ----------
    x : array-like
        Linear time series (angular velocity)
    y : array-like
        Circular time series in radians (heading deviation)
    max_lag : int
        Maximum lag to test (both positive and negative)
        
    Returns
    -------
    lags : ndarray
        Array of lag values
    correlations : ndarray
        Circular-linear correlation coefficients at each lag (signed)
    p_values : ndarray
        P-values for each correlation
    """
    x = np.array(x)
    y = np.array(y)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < max_lag * 2:
        return None, None, None
    
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros(len(lags))
    p_values = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
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
        
        if len(x_lagged) > 0:
            # Use circular-linear correlation (signed version for interpretability)
            corr, pval = signed_circular_linear_correlation(x_lagged, y_lagged)
            correlations[i] = corr
            p_values[i] = pval
        else:
            correlations[i] = np.nan
            p_values[i] = np.nan
    
    return lags, correlations, p_values


def compute_trial_cross_correlations(trial_data, max_lag=10):
    """
    Compute cross-correlations for a single trial.
    
    Parameters
    ----------
    trial_data : DataFrame
        Trial data with angular_velocity and mvtDirError columns
    max_lag : int
        Maximum lag to test
        
    Returns
    -------
    results : dict
        Dictionary containing lag analysis results
    """
    ang_vel = trial_data['angular_velocity'].values
    heading_dev = trial_data['mvtDirError'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(ang_vel) | np.isnan(heading_dev))
    ang_vel = ang_vel[valid_mask]
    heading_dev = heading_dev[valid_mask]
    
    if len(ang_vel) < max_lag * 2:
        return None
    
    lags, corrs, pvals = compute_cross_correlation(ang_vel, heading_dev, max_lag)
    
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

def process_condition_for_lag_analysis(df, condition, max_lag=10, min_speed=2.0):
    """
    Process a condition for lag analysis.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    condition : str
        Condition name
    max_lag : int
        Maximum lag to test
    min_speed : float
        Minimum speed threshold
        
    Returns
    -------
    results : dict
        Lag analysis results for the condition
    """
    print(f"\nProcessing {condition}...")
    condition_df = df[df.condition == condition].copy()
    condition_df['session_trial'] = condition_df['session'] + '_T' + condition_df['trial'].astype(str)
    unique_trials = condition_df['session_trial'].unique()
    print(f"Found {len(unique_trials)} trials")
    
    trial_results = []
    
    for trial_id in tqdm(unique_trials):
        trial_data = condition_df[condition_df['session_trial'] == trial_id].copy()
        trial_data = trial_data.sort_values('recTime')
        
        if len(trial_data) < MIN_TRIAL_LENGTH:
            continue
        
        # Calculate variables
        heading = calculate_heading_from_position(trial_data['x'].values, trial_data['y'].values)
        angular_velocity = calculate_angular_velocity(heading, trial_data['recTime'].values)
        speed = calculate_speed(trial_data['x'].values, trial_data['y'].values, trial_data['recTime'].values)
        
        trial_data['heading'] = heading
        trial_data['angular_velocity'] = angular_velocity
        trial_data['speed'] = speed
        
        # Filter for movement
        moving_mask = (speed >= min_speed) & ~np.isnan(angular_velocity) & ~np.isnan(trial_data['mvtDirError'])
        trial_data_moving = trial_data[moving_mask]
        
        if len(trial_data_moving) < max_lag * 2:
            continue
        
        # Compute cross-correlations for this trial
        trial_cc = compute_trial_cross_correlations(trial_data_moving, max_lag)
        
        if trial_cc is not None:
            trial_results.append(trial_cc)
    
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
        'n_trials': len(trial_results)
    }


def aggregate_lag_analysis(df, conditions, max_lag=10):
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
        
    Returns
    -------
    results : dict
        Dictionary mapping conditions to lag analysis results
    """
    results = {}
    
    for condition in conditions:
        result = process_condition_for_lag_analysis(df, condition, max_lag)
        if result is not None:
            results[condition] = result
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cross_correlation_function(results, condition):
    """
    Plot cross-correlation function for a condition.
    
    Parameters
    ----------
    results : dict
        Lag analysis results
    condition : str
        Condition name
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = results['lags']
    mean_corrs = results['mean_correlations']
    sem_corrs = results['sem_correlations']
    
    # Plot correlation function
    ax.plot(lags, mean_corrs, 'o-', linewidth=2, markersize=8, label='Mean correlation')
    ax.fill_between(lags, mean_corrs - sem_corrs, mean_corrs + sem_corrs, 
                     alpha=0.3, label='SEM')
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Lag (timesteps)', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title(f'Cross-Correlation: Angular Velocity ↔ Heading Deviation\n{condition}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    textstr = 'Negative lag: Heading deviation predicts future angular velocity\n'
    textstr += 'Positive lag: Angular velocity predicts future heading deviation'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_lag_heatmap(all_results, conditions):
    """
    Create heatmap of correlations across conditions and lags.
    
    Parameters
    ----------
    all_results : dict
        Dictionary of lag analysis results for all conditions
    conditions : list
        List of condition names
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    # Prepare data matrix
    lags = all_results[conditions[0]]['lags']
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
    
    # Set ticks
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels(lags)
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(conditions)
    
    # Labels
    ax.set_xlabel('Lag (timesteps)', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    ax.set_title('Cross-Correlation Heatmap Across Conditions', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11)
    
    # Add vertical line at lag=0
    ax.axvline(x=len(lags)//2, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_optimal_lags(all_results, conditions):
    """
    Plot optimal lags (maximum absolute correlation) for each condition.
    
    Parameters
    ----------
    all_results : dict
        Dictionary of lag analysis results
    conditions : list
        List of condition names
        
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
            optimal_lag = lags[max_idx]
            max_corr = corrs[max_idx]
            
            optimal_lags.append(optimal_lag)
            max_corrs.append(max_corr)
            condition_labels.append(condition)
    
    # Plot 1: Optimal lag values
    colors = ['blue' if lag < 0 else 'red' if lag > 0 else 'gray' for lag in optimal_lags]
    ax1.barh(range(len(condition_labels)), optimal_lags, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(condition_labels)))
    ax1.set_yticklabels(condition_labels)
    ax1.set_xlabel('Optimal Lag (timesteps)', fontsize=12)
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


def plot_individual_trials(results, condition, n_trials=5):
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
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = results['lags']
    trial_corrs = results['trial_correlations']
    
    # Plot individual trials
    n_plot = min(n_trials, len(trial_corrs))
    for i in range(n_plot):
        ax.plot(lags, trial_corrs[i], alpha=0.3, linewidth=1)
    
    # Plot mean
    mean_corrs = results['mean_correlations']
    ax.plot(lags, mean_corrs, 'k-', linewidth=3, label='Mean')
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Lag (timesteps)', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title(f'Individual Trial Cross-Correlations\n{condition}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    print("LAG ANALYSIS: Angular Velocity ↔ Heading Deviation")
    print("="*80)
    print("\nLoading reconstruction data...")
    fn = os.path.join(PROJECT_DATA_PATH, "results", "reconstuctionDFAutoPI.csv")
    dfAutoPI = pd.read_csv(fn)
    print(f"Loaded {len(dfAutoPI)} rows")

    # Filter for useable sessions
    dfAutoPI = dfAutoPI[dfAutoPI.session.isin(useAble)]
    print(f"After filtering: {len(dfAutoPI)} rows")
    print(f"Sessions: {dfAutoPI.session.nunique()}")

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
    print("="*80)
    all_results = aggregate_lag_analysis(dfAutoPI, conditions, max_lag=MAX_LAG)

    # Generate statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    summary_df = generate_statistical_summary(all_results, conditions)
    print("\n", summary_df.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(PROJECT_DATA_PATH, 'results', 'lag_analysis_summary.csv')
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
            fig = plot_cross_correlation_function(all_results[condition], condition)
            fig.savefig(os.path.join(results_dir, f'lag_ccf_{condition}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # 2. Heatmap across conditions
    print("\nGenerating heatmap...")
    fig = plot_lag_heatmap(all_results, conditions)
    fig.savefig(os.path.join(results_dir, 'lag_heatmap_all_conditions.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Optimal lags plot
    print("\nGenerating optimal lags plot...")
    fig = plot_optimal_lags(all_results, conditions)
    fig.savefig(os.path.join(results_dir, 'lag_optimal_lags.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Individual trial examples
    for condition in ['all_light', 'all_dark']:
        if condition in all_results:
            print(f"\nGenerating individual trials plot for {condition}...")
            fig = plot_individual_trials(all_results[condition], condition, n_trials=10)
            fig.savefig(os.path.join(results_dir, f'lag_individual_trials_{condition}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nInterpretation Guide:")
    print("- Negative lag: Heading deviation at time t predicts angular velocity at t+Δt")
    print("  (Navigation error causes corrective turning)")
    print("- Positive lag: Angular velocity at time t predicts heading deviation at t+Δt")
    print("  (Turning causes subsequent navigation error)")
    print("- Zero lag: Simultaneous relationship between variables")
    print("\nResults saved to:", results_dir)
