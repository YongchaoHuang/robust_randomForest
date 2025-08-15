# -*- coding: utf-8 -*-
"""Robust random Forest.ipynb

yongchao.huang@abdn.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

def analyze_prediction(X_train, y_train, X_new, n_trees=500, max_depth=5, trim_percent=5.0, confidence_level=90.0,
                       methods=['mean', 'trimmed_mean', 'mode', 'prediction_interval']):
    """
    Trains a robust random forest ensemble and analyzes the prediction distribution
    using a user-specified list of methods.

    Args:
        X_train (np.array): Training feature data.
        y_train (np.array): Training target data.
        X_new (np.array): The new data point(s) to predict.
        n_trees (int): The number of decision trees to train in the ensemble.
        max_depth (int): The maximum depth of each individual decision tree.
        trim_percent (float): Percentage to trim for the trimmed mean.
        confidence_level (float): Confidence level for the prediction interval.
        methods (list): A list of strings for the methods to run.
                        Available methods: 'mean', 'trimmed_mean', 'mode', 'som',
                        'lom', 'cog', 'boa', 'mwm', 'kwa', 'prediction_interval'.

    Returns:
        dict: A dictionary containing raw predictions and calculated metrics.
    """
    # --- 1. Build the Ensemble ---
    trained_trees = []
    print(f"Training {n_trees} individual decision trees...")
    for _ in range(n_trees):
        X_sample, y_sample = resample(X_train, y_train)
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X_sample, y_sample)
        trained_trees.append(tree)
    print("✅ Training complete.")

    # --- 2. Collect Predictions ---
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    predictions = np.array([tree.predict(X_new)[0] for tree in trained_trees])

    # --- 3. Calculate Requested Metrics ---
    results_metrics = {}

    # Pre-calculate histogram data as it's used by multiple methods
    counts, bin_edges = np.histogram(predictions, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if 'mean' in methods:
        results_metrics['mean'] = np.mean(predictions)

    if 'trimmed_mean' in methods:
        low_p = np.percentile(predictions, trim_percent)
        high_p = np.percentile(predictions, 100 - trim_percent)
        trimmed_preds = predictions[(predictions >= low_p) & (predictions <= high_p)]
        results_metrics['trimmed_mean'] = np.mean(trimmed_preds)

    # Maxima and mode-based methods all rely on the same initial calculation
    if any(m in methods for m in ['mode', 'som', 'lom', 'mwm', 'kwa']):
        max_count = np.max(counts)
        # Find all modes (centers of bins with the highest frequency)
        modes = bin_centers[np.where(counts == max_count)]
        if 'mode' in methods: # Mean of Maxima
            results_metrics['mode'] = np.mean(modes)
        if 'som' in methods: # Smallest of Maxima
            results_metrics['som'] = np.min(modes)
        if 'lom' in methods: # Largest of Maxima
            results_metrics['lom'] = np.max(modes)
        if 'mwm' in methods: # Mode-Weighted Mean
            weights = np.zeros_like(predictions)
            for i, pred in enumerate(predictions):
                min_dist_to_mode = np.min(np.abs(pred - modes))
                weights[i] = 1.0 / (min_dist_to_mode + 1e-9)
            results_metrics['mwm'] = np.sum(predictions * weights) / np.sum(weights)
        if 'kwa' in methods: # Kernel Weighted Average
            # Gamma is a hyperparameter for the RBF kernel. A common heuristic is 1 / (2 * sigma^2)
            gamma = 1.0 / (2 * np.var(predictions))
            weights = np.zeros_like(predictions)
            for i, pred in enumerate(predictions):
                min_dist_sq = np.min((pred - modes)**2)
                weights[i] = np.exp(-gamma * min_dist_sq)
            results_metrics['kwa'] = np.sum(predictions * weights) / np.sum(weights)


    if 'cog' in methods: # Center of Gravity
        results_metrics['cog'] = np.sum(bin_centers * counts) / np.sum(counts)

    if 'boa' in methods: # Bisector of Area
        total_area = np.sum(counts)
        cumulative_area = np.cumsum(counts)
        bisector_index = np.where(cumulative_area >= total_area / 2)[0][0]
        results_metrics['boa'] = bin_edges[bisector_index + 1]

    if 'prediction_interval' in methods:
        lower_p = (100.0 - confidence_level) / 2.0
        upper_p = 100.0 - lower_p
        results_metrics['prediction_interval'] = [np.percentile(predictions, lower_p), np.percentile(predictions, upper_p)]

    return {
        'predictions': predictions,
        'metrics': results_metrics,
        'parameters': {'n_trees': n_trees, 'trim_percent': trim_percent, 'confidence_level': confidence_level, 'X_new': X_new[0,0]}
    }

def plot_results(results, ground_truth=None):
    """
    Dynamically visualizes the results from the robust random forest analysis.
    """
    predictions = results['predictions']
    metrics = results['metrics']
    params = results['parameters']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.hist(predictions, bins='auto', color='cornflowerblue', edgecolor='black', alpha=0.7)

    # --- Plot Metrics Dynamically ---
    plot_config = {
        'mean': ('Mean', 'red', '--'),
        'trimmed_mean': (f'Trimmed Mean ({params["trim_percent"]*2}%)', 'black', '-'),
        'mode': ('Mode (MOM)', 'cyan', '--'),
        'som': ('Smallest of Max', 'magenta', ':'),
        'lom': ('Largest of Max', 'brown', ':'),
        'cog': ('Center of Gravity', 'purple', '-.'),
        'boa': ('Bisector of Area', 'orange', '-.'),
        'mwm': ('Mode-Weighted Mean', 'lime', '-'),
        'kwa': ('Kernel Weighted Avg', 'gold', '-')
    }

    if 'prediction_interval' in metrics:
        lower_bound, upper_bound = metrics['prediction_interval']
        ax.axvspan(lower_bound, upper_bound, color='yellow', alpha=0.3, label=f'{params["confidence_level"]}% Prediction Interval')

    if ground_truth is not None:
        ax.axvline(ground_truth, color='darkgreen', linestyle=':', linewidth=3.5, label=f'Ground Truth: {ground_truth:.3f}')

    for method, value in metrics.items():
        if method in plot_config:
            label, color, style = plot_config[method]
            ax.axvline(value, color=color, linestyle=style, linewidth=2.5, label=f'{label}: {value:.3f}')

    ax.set_title(f'Robust Analysis of {params["n_trees"]} Tree Predictions for X = {params["X_new"]}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Value', fontsize=12)
    ax.set_ylabel('Frequency (Number of Trees)', fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Generate sample data
    np.random.seed(42) # not Yong's favorite key but works best :)
    X_train = np.sort(5 * np.random.rand(100, 1), axis=0)
    y_train = np.sin(X_train).ravel()
    y_train += 0.2 * (np.random.rand(len(y_train)) - 0.5)

    X_new = np.array([2.5])
    ground_truth = np.sin(X_new[0])

    # 2. Run analysis with a custom list of methods, including the new 'kwa'
    all_methods = ['mean', 'trimmed_mean', 'mode', 'som', 'lom', 'cog', 'boa', 'mwm', 'kwa', 'prediction_interval']

    analysis_results = analyze_prediction(
        X_train, y_train, X_new,
        n_trees=500,
        confidence_level=95.0,
        methods=all_methods
    )

    # 3. Print numerical results
    print("\n--- Robust Predictions Comparison ---")
    print(f"Ground Truth Value:              {ground_truth:.4f}")
    print("------------------------------------")
    for key, value in analysis_results['metrics'].items():
        if isinstance(value, list):
             print(f"{key.replace('_', ' ').title():<20}: [{value[0]:.4f}, {value[1]:.4f}]")
        else:
            print(f"{key.replace('_', ' ').title():<20}: {value:.4f}")

    # 4. Plot the results
    plot_results(analysis_results, ground_truth)
