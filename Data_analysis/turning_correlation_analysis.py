"""
Analysis of cumulative turning correlation between actual and reconstructed trajectories.

This script calculates the total cumulative signed turning for each trial in the AutoPI task
and correlates it with the reconstructed trajectory turning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
import os

# Add parent directory to path to import project functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup_project import setup_project_session_lists, PROJECT_DATA_PATH


def calculate_heading_change(movement_vectors):
    """
    Calculate heading change between consecutive movement vectors.
    
    Parameters:
    -----------
    movement_vectors : numpy array
        Array of shape (n_samples, 2) with x, y movement components
        
    Returns:
    --------
    heading_changes : numpy array
        Array of heading changes in radians, properly handling circular wraparound
    """
    # Calculate heading at each time point
    headings = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0])
    
    # Calculate heading change
    heading_changes = np.diff(headings, prepend=np.nan)
    
    # Handle circular wraparound
    heading_changes = np.where(heading_changes > np.pi, heading_changes - 2*np.pi, heading_changes)
    heading_changes = np.where(heading_changes < -np.pi, heading_changes + 2*np.pi, heading_changes)
    
    return heading_changes


def calculate_cumulative_turning(movement_vectors):
    """
    Calculate cumulative signed turning from movement vectors.
    
    Parameters:
    -----------
    movement_vectors : numpy array
        Array of shape (n_samples, 2) with x, y movement components
        
    Returns:
    --------
    cumulative_turning : float
        Total cumulative signed turning wrapped to [-π, π]
    """
    heading_changes = calculate_heading_change(movement_vectors)
    
    # Sum all heading changes (ignoring first NaN)
    total_turning = np.nansum(heading_changes)
    
    # Wrap to [-π, π] range
    wrapped_turning = np.arctan2(np.sin(total_turning), np.cos(total_turning))
    
    return wrapped_turning


def analyze_turning_per_trial(dfAutoPI, condition='all_light'):
    """
    Analyze cumulative turning for each trial in a specific condition.
    
    Parameters:
    -----------
    dfAutoPI : DataFrame
        DataFrame with AutoPI reconstruction data
    condition : str
        Condition to analyze ('all_light' or 'all_dark')
        
    Returns:
    --------
    results_df : DataFrame
        DataFrame with actual and reconstructed turning per trial
    """
    results = []
    
    # Filter for the specified condition
    df_cond = dfAutoPI[dfAutoPI.condition == condition]
    
    # Get unique session-trial combinations
    session_trial_groups = df_cond.groupby(['session', 'trial'])
    
    for (session, trial), trial_data in session_trial_groups:
        # Get movement vectors
        actual_mvt = trial_data[['x', 'y']].values
        pred_mvt = trial_data[['px', 'py']].values
        
        # Calculate cumulative turning
        actual_turning = calculate_cumulative_turning(actual_mvt)
        pred_turning = calculate_cumulative_turning(pred_mvt)
        
        # Get mouse ID
        mouse = trial_data['mouse'].iloc[0] if 'mouse' in trial_data.columns else 'unknown'
        
        results.append({
            'session': session,
            'trial': trial,
            'mouse': mouse,
            'condition': condition,
            'actual_turning': actual_turning,
            'reconstructed_turning': pred_turning
        })
    
    return pd.DataFrame(results)


def plot_turning_correlation(results_df, condition_label, output_path=None, 
                             aggregated=False, figsize=(8, 8)):
    """
    Create scatter plot of actual vs reconstructed cumulative turning.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from analyze_turning_per_trial
    condition_label : str
        Label for the condition (e.g., 'Light Trials', 'Dark Trials')
    output_path : str, optional
        Path to save the figure
    aggregated : bool
        If True, aggregate by session (mean across trials)
    figsize : tuple
        Figure size
    """
    if aggregated:
        # Aggregate by session
        plot_df = results_df.groupby('session').agg({
            'actual_turning': 'mean',
            'reconstructed_turning': 'mean',
            'mouse': 'first'
        }).reset_index()
    else:
        plot_df = results_df.copy()
    
    # Calculate correlation
    r, p = pearsonr(plot_df['actual_turning'], plot_df['reconstructed_turning'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(plot_df['actual_turning'], 
               plot_df['reconstructed_turning'],
               alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Identity line
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', alpha=0.3, label='Identity')
    
    # Set limits
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    # Set ticks
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'-$\pi$', r'-$\pi$/2', '0', r'$\pi$/2', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'-$\pi$', r'-$\pi$/2', '0', r'$\pi$/2', r'$\pi$'])
    
    # Labels
    ax.set_xlabel('Actual Cumulative Turning (rad)', fontsize=12)
    ax.set_ylabel('Reconstructed Cumulative Turning (rad)', fontsize=12)
    
    # Title with correlation info
    aggregation_type = "Session-averaged" if aggregated else "Per-trial"
    title = f'{condition_label} - {aggregation_type}\n'
    title += f'r = {r:.3f}, p = {p:.4f}, n = {len(plot_df)}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    return fig, ax


def main():
    """Main analysis function."""
    
    # Setup project
    print("Loading data...")
    projectName, dataPath, dlcModelPath, myProject, sSessions = setup_project_session_lists(
        projectName="autopi_mec",
        dataPath=PROJECT_DATA_PATH,
        dlcModelPath=""
    )
    
    # Load AutoPI reconstruction data
    fn = myProject.dataPath + "/results/reconstuctionDFAutoPI.csv"
    print(f"Loading {fn}")
    dfAutoPI = pd.read_csv(fn)
    
    print(f"Data loaded: {len(dfAutoPI)} time points, {dfAutoPI.session.nunique()} sessions")
    print(f"Conditions: {dfAutoPI.condition.unique()}")
    
    # Analyze light trials
    print("\nAnalyzing light trials...")
    results_light = analyze_turning_per_trial(dfAutoPI, condition='all_light')
    print(f"Light trials: {len(results_light)} trials from {results_light.session.nunique()} sessions")
    
    # Analyze dark trials
    print("\nAnalyzing dark trials...")
    results_dark = analyze_turning_per_trial(dfAutoPI, condition='all_dark')
    print(f"Dark trials: {len(results_dark)} trials from {results_dark.session.nunique()} sessions")
    
    # Create output directory
    output_dir = os.path.join(myProject.dataPath, "results", "turning_correlation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate figures - Per trial
    print("\nGenerating per-trial figures...")
    
    fig_light_trial, _ = plot_turning_correlation(
        results_light, 
        'Light Trials',
        output_path=os.path.join(output_dir, 'turning_correlation_light_per_trial.pdf'),
        aggregated=False
    )
    plt.show()
    
    fig_dark_trial, _ = plot_turning_correlation(
        results_dark,
        'Dark Trials', 
        output_path=os.path.join(output_dir, 'turning_correlation_dark_per_trial.pdf'),
        aggregated=False
    )
    plt.show()
    
    # Generate figures - Aggregated by session
    print("\nGenerating session-aggregated figures...")
    
    fig_light_agg, _ = plot_turning_correlation(
        results_light,
        'Light Trials',
        output_path=os.path.join(output_dir, 'turning_correlation_light_aggregated.pdf'),
        aggregated=True
    )
    plt.show()
    
    fig_dark_agg, _ = plot_turning_correlation(
        results_dark,
        'Dark Trials',
        output_path=os.path.join(output_dir, 'turning_correlation_dark_aggregated.pdf'),
        aggregated=True
    )
    plt.show()
    
    # Save results to CSV
    print("\nSaving results...")
    results_all = pd.concat([results_light, results_dark], ignore_index=True)
    results_path = os.path.join(output_dir, 'turning_correlation_results.csv')
    results_all.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for condition_name, results_df in [('Light', results_light), ('Dark', results_dark)]:
        print(f"\n{condition_name} Trials:")
        print(f"  Number of trials: {len(results_df)}")
        print(f"  Number of sessions: {results_df.session.nunique()}")
        
        r_trial, p_trial = pearsonr(results_df['actual_turning'], 
                                     results_df['reconstructed_turning'])
        print(f"  Per-trial correlation: r = {r_trial:.3f}, p = {p_trial:.4f}")
        
        # Aggregated by session
        agg_df = results_df.groupby('session').agg({
            'actual_turning': 'mean',
            'reconstructed_turning': 'mean'
        }).reset_index()
        r_agg, p_agg = pearsonr(agg_df['actual_turning'], 
                                agg_df['reconstructed_turning'])
        print(f"  Session-aggregated correlation: r = {r_agg:.3f}, p = {p_agg:.4f}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
