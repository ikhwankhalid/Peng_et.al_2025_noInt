"""
Visualization Script for Integrated Angular Velocity Analysis

Creates publication-ready figures for testing systematic integration errors
in angular velocity during continuous movement bouts.

Figures:
1. Bout identification verification
2. Core result - Asymmetry in integration (scatter + regression)
3. Slope comparison across conditions
4. Accumulation over time within bouts
5. Speed threshold comparison
6. Behavioral phase comparison

Author: Analysis generated for Peng et al. 2025
Date: 2025-11-24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# Paths
PROJECT_DATA_PATH = 'E:\\GitHub\\Peng_et.al_2025_noInt\\Peng'


# =============================================================================
# FIGURE 1: BOUT IDENTIFICATION VERIFICATION
# =============================================================================

def plot_bout_verification(bout_data):
    """
    Verify that bout identification is working as expected.

    Creates:
    - Distribution of bout lengths across speed thresholds
    - Number of bouts per condition
    - Example trials showing identified bouts
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Bout length distributions
    ax = axes[0, 0]
    for threshold in bout_data['speed_threshold'].unique():
        threshold_data = bout_data[bout_data['speed_threshold'] == threshold]
        bout_lengths = threshold_data.groupby(['trial_id', 'bout_id'])['bout_length'].first()
        ax.hist(bout_lengths, bins=50, alpha=0.6, label=f'{threshold} cm/s')
    ax.set_xlabel('Bout Length (timepoints)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Bout Lengths')
    ax.legend()
    ax.set_xlim(0, 200)

    # Panel B: Number of bouts per condition
    ax = axes[0, 1]
    bout_counts = bout_data.groupby(['condition', 'speed_threshold']).apply(
        lambda x: x['bout_id'].nunique()).unstack()
    bout_counts.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Number of Bouts')
    ax.set_title('Bouts per Condition × Speed Threshold')
    ax.legend(title='Speed Threshold', loc='best')
    ax.tick_params(axis='x', rotation=45)

    # Panel C: Timepoints per condition
    ax = axes[1, 0]
    timepoint_counts = bout_data.groupby(['condition', 'speed_threshold']).size().unstack()
    timepoint_counts.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Total Timepoints')
    ax.set_title('Data Points per Condition × Speed Threshold')
    ax.legend(title='Speed Threshold', loc='best')
    ax.tick_params(axis='x', rotation=45)

    # Panel D: Example trial showing bouts
    ax = axes[1, 1]
    # Pick one trial to visualize
    example_trial = bout_data[bout_data['speed_threshold'] == 2.0]['trial_id'].iloc[0]
    trial_data = bout_data[(bout_data['trial_id'] == example_trial) &
                           (bout_data['speed_threshold'] == 2.0)]

    ax.plot(trial_data['time_in_bout'], trial_data['speed'], 'k-', alpha=0.5, linewidth=1)

    # Color each bout differently
    for bout_id in trial_data['bout_id'].unique():
        bout = trial_data[trial_data['bout_id'] == bout_id]
        ax.scatter(bout['time_in_bout'], bout['speed'], s=20, alpha=0.7, label=f'Bout {bout_id}')

    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Time in Trial (s)')
    ax.set_ylabel('Speed (cm/s)')
    ax.set_title(f'Example Trial: {example_trial}')
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2: CORE RESULT - ASYMMETRY IN INTEGRATION
# =============================================================================

def plot_integration_asymmetry(bout_data, condition='all_light', speed_threshold=2.0):
    """
    Core result: scatter plot and regression lines for left vs right turns.
    """
    data = bout_data[(bout_data['condition'] == condition) &
                     (bout_data['speed_threshold'] == speed_threshold)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Full scatter
    ax = axes[0]

    left_data = data[data['turn_direction'] == 'left']
    right_data = data[data['turn_direction'] == 'right']

    # Scatter points
    ax.scatter(left_data['integrated_ang_vel'], left_data['mvtDirError'],
               s=1, alpha=0.3, c='red', label=f'Left (n={len(left_data):,})')
    ax.scatter(right_data['integrated_ang_vel'], right_data['mvtDirError'],
               s=1, alpha=0.3, c='blue', label=f'Right (n={len(right_data):,})')

    # Regression lines
    if len(left_data) > 10:
        X_left = left_data['integrated_ang_vel']
        Y_left = left_data['mvtDirError']
        mask = ~(X_left.isna() | Y_left.isna())
        if mask.sum() > 10:
            slope, intercept, r, p, se = stats.linregress(X_left[mask], Y_left[mask])
            x_fit = np.linspace(X_left.min(), X_left.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=3,
                   label=f'Left: β={slope:.4f}, p={p:.4f}')

    if len(right_data) > 10:
        X_right = right_data['integrated_ang_vel']
        Y_right = right_data['mvtDirError']
        mask = ~(X_right.isna() | Y_right.isna())
        if mask.sum() > 10:
            slope, intercept, r, p, se = stats.linregress(X_right[mask], Y_right[mask])
            x_fit = np.linspace(X_right.min(), X_right.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'b-', linewidth=3,
                   label=f'Right: β={slope:.4f}, p={p:.4f}')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Integrated Angular Velocity (rad)')
    ax.set_ylabel('Heading Deviation (rad)')
    ax.set_title(f'{condition} - {speed_threshold} cm/s')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Panel B: Binned analysis for clarity
    ax = axes[1]

    # Bin by integrated angular velocity
    n_bins = 10
    left_bins = pd.cut(left_data['integrated_ang_vel'], bins=n_bins)
    right_bins = pd.cut(right_data['integrated_ang_vel'], bins=n_bins)

    left_binned = left_data.groupby(left_bins)['mvtDirError'].agg(['mean', 'sem', 'count'])
    right_binned = right_data.groupby(right_bins)['mvtDirError'].agg(['mean', 'sem', 'count'])

    # Get bin centers
    left_centers = [interval.mid for interval in left_binned.index if interval == interval]
    right_centers = [interval.mid for interval in right_binned.index if interval == interval]

    # Filter bins with sufficient data
    left_binned = left_binned[left_binned['count'] >= 10]
    right_binned = right_binned[right_binned['count'] >= 10]
    left_centers = [interval.mid for interval in left_binned.index]
    right_centers = [interval.mid for interval in right_binned.index]

    ax.errorbar(left_centers, left_binned['mean'], yerr=left_binned['sem'],
               fmt='o-', color='red', capsize=3, linewidth=2, markersize=6, label='Left')
    ax.errorbar(right_centers, right_binned['mean'], yerr=right_binned['sem'],
               fmt='o-', color='blue', capsize=3, linewidth=2, markersize=6, label='Right')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Integrated Angular Velocity (rad)')
    ax.set_ylabel('Mean Heading Deviation (rad)')
    ax.set_title('Binned Analysis')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 3: SLOPE COMPARISON ACROSS CONDITIONS
# =============================================================================

def plot_slope_comparison(regression_results, speed_threshold=2.0):
    """
    Compare β_left vs β_right across all conditions.
    """
    data = regression_results[regression_results['speed_threshold'] == speed_threshold].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Slope comparison scatter
    ax = axes[0]

    ax.scatter(data['beta_right'], data['beta_left'], s=100, alpha=0.7)

    # Add diagonal line (β_left = β_right, no asymmetry)
    lim_min = min(data['beta_left'].min(), data['beta_right'].min())
    lim_max = max(data['beta_left'].max(), data['beta_right'].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, label='No asymmetry')

    # Label significant asymmetries
    sig_data = data[data['p_asymmetry'] < 0.05]
    if len(sig_data) > 0:
        ax.scatter(sig_data['beta_right'], sig_data['beta_left'],
                  s=150, facecolors='none', edgecolors='red', linewidths=2, label='Significant (p<0.05)')

    # Add labels
    for _, row in data.iterrows():
        ax.annotate(row['condition'], (row['beta_right'], row['beta_left']),
                   fontsize=8, alpha=0.7, ha='right')

    ax.set_xlabel('β_right (Right Turn Slope)')
    ax.set_ylabel('β_left (Left Turn Slope)')
    ax.set_title(f'Slope Asymmetry - {speed_threshold} cm/s')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.2)

    # Panel B: Bar plot of asymmetry magnitude
    ax = axes[1]

    data_sorted = data.sort_values('beta_diff')
    colors = ['red' if p < 0.05 else 'gray' for p in data_sorted['p_asymmetry']]

    ax.barh(range(len(data_sorted)), data_sorted['beta_diff'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(data_sorted)))
    ax.set_yticklabels(data_sorted['condition'])
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Asymmetry (β_left - β_right)')
    ax.set_title('Slope Difference by Condition')
    ax.grid(True, alpha=0.3, axis='x')

    # Add significance stars
    for i, (_, row) in enumerate(data_sorted.iterrows()):
        if row['p_asymmetry'] < 0.001:
            ax.text(row['beta_diff'], i, ' ***', va='center', fontweight='bold')
        elif row['p_asymmetry'] < 0.01:
            ax.text(row['beta_diff'], i, ' **', va='center', fontweight='bold')
        elif row['p_asymmetry'] < 0.05:
            ax.text(row['beta_diff'], i, ' *', va='center', fontweight='bold')

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 4: ACCUMULATION OVER TIME WITHIN BOUTS
# =============================================================================

def plot_bout_accumulation(bout_data, condition='all_light', speed_threshold=2.0):
    """
    Test if heading deviation accumulates over time within bouts.
    """
    data = bout_data[(bout_data['condition'] == condition) &
                     (bout_data['speed_threshold'] == speed_threshold)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: |heading_dev| vs time_in_bout (all data)
    ax = axes[0, 0]
    data['abs_heading_dev'] = np.abs(data['mvtDirError'])

    # Bin by time
    time_bins = pd.cut(data['time_in_bout'], bins=20)
    binned = data.groupby(time_bins)['abs_heading_dev'].agg(['mean', 'sem', 'count'])
    binned = binned[binned['count'] >= 10]

    time_centers = [interval.mid for interval in binned.index]

    ax.errorbar(time_centers, binned['mean'], yerr=binned['sem'],
               fmt='o-', capsize=3, linewidth=2, markersize=6, color='purple')
    ax.set_xlabel('Time in Bout (s)')
    ax.set_ylabel('|Heading Deviation| (rad)')
    ax.set_title('Accumulation of Heading Error')
    ax.grid(True, alpha=0.3)

    # Panel B: Separate for left vs right bouts
    ax = axes[0, 1]

    for direction, color in [('left', 'red'), ('right', 'blue')]:
        dir_data = data[data['turn_direction'] == direction]
        time_bins = pd.cut(dir_data['time_in_bout'], bins=20)
        binned = dir_data.groupby(time_bins)['abs_heading_dev'].agg(['mean', 'sem', 'count'])
        binned = binned[binned['count'] >= 10]
        time_centers = [interval.mid for interval in binned.index]

        ax.errorbar(time_centers, binned['mean'], yerr=binned['sem'],
                   fmt='o-', capsize=3, linewidth=2, markersize=6, color=color, label=direction.capitalize())

    ax.set_xlabel('Time in Bout (s)')
    ax.set_ylabel('|Heading Deviation| (rad)')
    ax.set_title('Accumulation by Turn Direction')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Panel C: Integrated ang vel magnitude vs time
    ax = axes[1, 0]
    data['abs_integrated_ang_vel'] = np.abs(data['integrated_ang_vel'])

    time_bins = pd.cut(data['time_in_bout'], bins=20)
    binned = data.groupby(time_bins)['abs_integrated_ang_vel'].agg(['mean', 'sem', 'count'])
    binned = binned[binned['count'] >= 10]
    time_centers = [interval.mid for interval in binned.index]

    ax.errorbar(time_centers, binned['mean'], yerr=binned['sem'],
               fmt='o-', capsize=3, linewidth=2, markersize=6, color='green')
    ax.set_xlabel('Time in Bout (s)')
    ax.set_ylabel('|Integrated Angular Velocity| (rad)')
    ax.set_title('Accumulation of Rotation')
    ax.grid(True, alpha=0.3)

    # Panel D: Correlation within individual bouts
    ax = axes[1, 1]

    bout_correlations = []
    for bout_id in data['bout_id'].unique():
        bout = data[data['bout_id'] == bout_id]
        if len(bout) >= 10:
            valid = ~(bout['abs_heading_dev'].isna() | bout['time_in_bout'].isna())
            if valid.sum() >= 10:
                r, p = stats.pearsonr(bout['time_in_bout'][valid], bout['abs_heading_dev'][valid])
                bout_correlations.append(r)

    if len(bout_correlations) > 0:
        ax.hist(bout_correlations, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=np.mean(bout_correlations), color='blue', linestyle='-', linewidth=2,
                  label=f'Mean r={np.mean(bout_correlations):.3f}')
        ax.set_xlabel('Correlation (time vs |heading_dev|)')
        ax.set_ylabel('Number of Bouts')
        ax.set_title('Within-Bout Correlations')
        ax.legend()

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 5: SPEED THRESHOLD COMPARISON
# =============================================================================

def plot_speed_threshold_comparison(regression_results, condition='all_light'):
    """
    Compare results across different speed thresholds.
    """
    data = regression_results[regression_results['condition'] == condition].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    thresholds = sorted(data['speed_threshold'].unique())

    # Panel A: β_left across thresholds
    ax = axes[0, 0]
    ax.plot(thresholds, data['beta_left'], 'o-', linewidth=2, markersize=8, color='red')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Speed Threshold (cm/s)')
    ax.set_ylabel('β_left')
    ax.set_title('Left Turn Slope vs Threshold')
    ax.grid(True, alpha=0.3)

    # Panel B: β_right across thresholds
    ax = axes[0, 1]
    ax.plot(thresholds, data['beta_right'], 'o-', linewidth=2, markersize=8, color='blue')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Speed Threshold (cm/s)')
    ax.set_ylabel('β_right')
    ax.set_title('Right Turn Slope vs Threshold')
    ax.grid(True, alpha=0.3)

    # Panel C: Asymmetry magnitude
    ax = axes[1, 0]
    ax.plot(thresholds, data['beta_diff'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Speed Threshold (cm/s)')
    ax.set_ylabel('Asymmetry (β_left - β_right)')
    ax.set_title('Asymmetry Magnitude vs Threshold')
    ax.grid(True, alpha=0.3)

    # Panel D: Sample sizes
    ax = axes[1, 1]
    width = 0.35
    x = np.arange(len(thresholds))
    ax.bar(x - width/2, data['n_left'], width, label='Left', alpha=0.7, color='red')
    ax.bar(x + width/2, data['n_right'], width, label='Right', alpha=0.7, color='blue')
    ax.set_xlabel('Speed Threshold (cm/s)')
    ax.set_ylabel('Number of Timepoints')
    ax.set_title('Sample Sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Speed Threshold Robustness: {condition}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 6: BEHAVIORAL PHASE COMPARISON
# =============================================================================

def plot_behavioral_phase_comparison(regression_results, speed_threshold=2.0):
    """
    Compare asymmetry across behavioral phases (search, homing, at-lever).
    """
    data = regression_results[regression_results['speed_threshold'] == speed_threshold].copy()

    # Extract phase and light condition
    data['phase'] = data['condition'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    data['light'] = data['condition'].apply(lambda x: 'light' if 'light' in x else 'dark')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Asymmetry by phase
    ax = axes[0, 0]
    phases = ['searchToLeverPath', 'homingFromLeavingLever', 'atLever', 'all']
    colors_phase = ['orange', 'green', 'purple', 'gray']

    for phase, color in zip(phases, colors_phase):
        phase_data = data[data['phase'] == phase]
        if len(phase_data) > 0:
            x_vals = [phase_data[phase_data['light'] == 'light']['beta_diff'].values,
                     phase_data[phase_data['light'] == 'dark']['beta_diff'].values]
            labels = [f'{phase}\nlight', f'{phase}\ndark']

            for i, (x, label) in enumerate(zip(x_vals, labels)):
                if len(x) > 0:
                    # Filter out NaN values
                    x_clean = x[~np.isnan(x)]
                    if len(x_clean) > 0:
                        ax.scatter([label], x_clean, s=100, color=color, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Asymmetry (β_left - β_right)')
    ax.set_title('Asymmetry by Behavioral Phase')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Slope magnitudes by phase
    ax = axes[0, 1]
    phase_data_list = []
    for phase in phases:
        for light in ['light', 'dark']:
            subset = data[(data['phase'] == phase) & (data['light'] == light)]
            if len(subset) > 0:
                phase_data_list.append({
                    'phase': phase,
                    'light': light,
                    'beta_left': subset['beta_left'].values[0],
                    'beta_right': subset['beta_right'].values[0]
                })

    phase_df = pd.DataFrame(phase_data_list)
    # Filter rows where both beta_left and beta_right are not NaN for plotting
    phase_df_clean = phase_df[(~phase_df['beta_left'].isna()) & (~phase_df['beta_right'].isna())].copy()

    if len(phase_df_clean) > 0:
        x = np.arange(len(phase_df_clean))
        width = 0.35
        ax.bar(x - width/2, phase_df_clean['beta_left'], width, label='β_left', alpha=0.7, color='red')
        ax.bar(x + width/2, phase_df_clean['beta_right'], width, label='β_right', alpha=0.7, color='blue')
        ax.set_ylabel('Slope')
        ax.set_title('Regression Slopes by Phase')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['phase']}\n{row['light']}" for _, row in phase_df_clean.iterrows()],
                           rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Significance by phase
    ax = axes[1, 0]
    phase_sig = []
    for phase in phases:
        for light in ['light', 'dark']:
            subset = data[(data['phase'] == phase) & (data['light'] == light)]
            if len(subset) > 0:
                p_val = subset['p_asymmetry'].values[0]
                phase_sig.append({
                    'phase_light': f'{phase}_{light}',
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })

    phase_sig_df = pd.DataFrame(phase_sig)
    if len(phase_sig_df) > 0:
        # Filter out NaN p-values
        phase_sig_df = phase_sig_df[~phase_sig_df['p_value'].isna()].copy()
        if len(phase_sig_df) > 0:
            colors = ['red' if sig else 'gray' for sig in phase_sig_df['significant']]
            ax.barh(range(len(phase_sig_df)), -np.log10(phase_sig_df['p_value']), color=colors, alpha=0.7)
            ax.set_yticks(range(len(phase_sig_df)))
            ax.set_yticklabels(phase_sig_df['phase_light'])
            ax.axvline(x=-np.log10(0.05), color='black', linestyle='--', linewidth=2, label='p=0.05')
            ax.set_xlabel('-log10(p-value)')
            ax.set_title('Statistical Significance of Asymmetry')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')

    # Panel D: Sample sizes by phase
    ax = axes[1, 1]
    if len(phase_df_clean) > 0:
        x = np.arange(len(phase_df_clean))
        # Get sample sizes from full data
        n_left_vals = []
        n_right_vals = []
        for _, row in phase_df_clean.iterrows():
            subset = data[(data['phase'] == row['phase']) & (data['light'] == row['light'])]
            if len(subset) > 0:
                n_left_vals.append(subset['n_left'].values[0])
                n_right_vals.append(subset['n_right'].values[0])

        width = 0.35
        ax.bar(x - width/2, n_left_vals, width, label='Left', alpha=0.7, color='red')
        ax.bar(x + width/2, n_right_vals, width, label='Right', alpha=0.7, color='blue')
        ax.set_ylabel('Number of Timepoints')
        ax.set_title('Sample Sizes by Phase')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['phase']}\n{row['light']}" for _, row in phase_df_clean.iterrows()],
                           rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Behavioral Phase Comparison - {speed_threshold} cm/s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    spdthresh = 1.  # Default speed threshold for analyses
    print("="*80)
    print("VISUALIZATION SCRIPT FOR INTEGRATED ANGULAR VELOCITY ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    bout_data_fn = os.path.join(PROJECT_DATA_PATH, "results", "integrated_ang_vel_bout_data.csv")
    reg_results_fn = os.path.join(PROJECT_DATA_PATH, "results", "integrated_ang_vel_regression_results.csv")

    try:
        bout_data = pd.read_csv(bout_data_fn)
        print(f"Loaded bout data: {len(bout_data):,} rows")
    except FileNotFoundError:
        print(f"ERROR: Could not find {bout_data_fn}")
        print("Please run integrated_angular_velocity_bouts_analysis.py first")
        exit(1)

    try:
        regression_results = pd.read_csv(reg_results_fn)
        print(f"Loaded regression results: {len(regression_results)} rows")
    except FileNotFoundError:
        print(f"ERROR: Could not find {reg_results_fn}")
        print("Please run integrated_angular_velocity_bouts_analysis.py first")
        exit(1)

    # Create output directory for figures
    fig_dir = os.path.join(PROJECT_DATA_PATH, "results", "integration_analysis_figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"\nSaving figures to: {fig_dir}")
     
    # Generate all figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)

    CONDITIONS = [
        'all_light',
        'all_dark',
        'searchToLeverPath_light',
        'searchToLeverPath_dark',
        'homingFromLeavingLever_light',
        'homingFromLeavingLever_dark',
        'atLever_light',
        'atLever_dark'
    ]

    # Figure 1: Bout verification
    print("\n[1/6] Bout verification...")
    fig1 = plot_bout_verification(bout_data)
    fig1.savefig(os.path.join(fig_dir, "Fig1_bout_verification.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  Saved: Fig1_bout_verification.png")

    # Figure 2: Integration asymmetry (for key conditions)
    print("\n[2/6] Integration asymmetry...")
    for condition in CONDITIONS:
        fig2 = plot_integration_asymmetry(bout_data, condition=condition, speed_threshold=spdthresh)
        fig2.savefig(os.path.join(fig_dir, f"Fig2_asymmetry_{condition}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: Fig2_asymmetry_{condition}.png")

    # Figure 3: Slope comparison
    print("\n[3/6] Slope comparison...")
    fig3 = plot_slope_comparison(regression_results, speed_threshold=spdthresh)
    fig3.savefig(os.path.join(fig_dir, "Fig3_slope_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: Fig3_slope_comparison.png")

    # Figure 4: Bout accumulation
    print("\n[4/6] Bout accumulation...")
    for condition in CONDITIONS:
        fig4 = plot_bout_accumulation(bout_data, condition=condition, speed_threshold=spdthresh)
        fig4.savefig(os.path.join(fig_dir, f"Fig4_accumulation_{condition}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print(f"  Saved: Fig4_accumulation_{condition}.png")

    # Figure 5: Speed threshold comparison
    print("\n[5/6] Speed threshold comparison...")
    for condition in CONDITIONS:
        fig5 = plot_speed_threshold_comparison(regression_results, condition=condition)
        fig5.savefig(os.path.join(fig_dir, f"Fig5_threshold_{condition}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print(f"  Saved: Fig5_threshold_{condition}.png")

    # Figure 6: Behavioral phase comparison
    print("\n[6/6] Behavioral phase comparison...")
    fig6 = plot_behavioral_phase_comparison(regression_results, speed_threshold=spdthresh)
    fig6.savefig(os.path.join(fig_dir, "Fig6_phase_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig6)
    print("  Saved: Fig6_phase_comparison.png")

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nFigures saved to: {fig_dir}")
