
# Robust Random Forest (robust_rf)

**robust_rf** is a Python package that extends the concept of Random Forest regression. Instead of returning a single point prediction (the mean), it analyzes the entire distribution of predictions from an ensemble of decision trees. This allows for a more robust and insightful understanding of a model's prediction, including its certainty and potential alternative outcomes.

This approach treats the prediction distribution as a "fuzzy" result and provides a suite of tools, including methods from fuzzy logic, to derive a "crisp" and well-understood final prediction.

## Features

* **Ensemble Training**: Easily train an ensemble of hundreds of decision trees.
* **Rich Prediction Analysis**: Go beyond the mean with a variety of robust statistical and defuzzification methods.
* **Quantify Uncertainty**: Automatically calculate and visualize prediction intervals to understand model confidence.
* **Robust to Outliers**: Use advanced, mode-aware averaging methods that are robust to outlier predictions.
* **Flexible and Customizable**: Control the number of trees, analysis methods, confidence levels, and more.
* **Comprehensive Visualization**: Generate detailed histograms that plot all calculated metrics against the prediction distribution for easy comparison.

## Installation & Usage

To use this package, simply save the provided `robust_rf.py` file in your project directory. You can then import the functions directly into your script:

```python
from robust_rf import analyze_prediction, plot_results
````

## Quickstart Guide

Here's a simple example of how to use `robust_rf` from start to finish.

### 1\. Import and Generate Data

First, import the necessary libraries and the package functions. We'll create some sample non-linear data for this demonstration.

```python
import numpy as np
from robust_rf import analyze_prediction, plot_results

# Generate sample data
np.random.seed(42)
X_train = np.sort(5 * np.random.rand(100, 1), axis=0)
y_train = np.sin(X_train).ravel()
y_train += 0.2 * (np.random.rand(len(y_train)) - 0.5)

# Define the point we want to predict
X_new = np.array([2.5])
ground_truth = np.sin(X_new[0])
```

### 2\. Run the Analysis

Use the `analyze_prediction()` function to train the ensemble and calculate all available robust metrics.

```python
# Define which analysis methods you want to run
all_methods = ['mean', 'trimmed_mean', 'mode', 'som', 'lom', 'cog', 'boa', 'mwm', 'kwa', 'prediction_interval']

# Run the analysis
analysis_results = analyze_prediction(
    X_train, y_train, X_new,
    n_trees=500,
    confidence_level=95.0,
    methods=all_methods 
)
```

### 3\. Print and Plot the Results

You can print the numerical results and then use the `plot_results()` function to visualize the analysis.

```python
# Print the numerical results
print("\n--- Robust Predictions Comparison ---")
print(f"Ground Truth Value: {ground_truth:.4f}")
print("------------------------------------")
for key, value in analysis_results['metrics'].items():
    if isinstance(value, list):
         print(f"{key.replace('_', ' ').title():<20}: [{value[0]:.4f}, {value[1]:.4f}]")
    else:
        print(f"{key.replace('_', ' ').title():<20}: {value:.4f}")

# Plot the visual results
plot_results(analysis_results, ground_truth)
```

This will output a detailed plot showing the histogram of predictions along with lines indicating the ground truth and the results of all the analysis methods you selected.

## Example Output

Running the quickstart guide will produce a detailed plot like the one below. This visualization is key to understanding the model's behavior. Notice how the standard `Mean` is pulled away from the main cluster of predictions by outliers. In contrast, robust methods such as the `Trimmed Mean`, `Mode-Weighted Mean`, and `Kernel Weighted Avg` provide a more central estimate that aligns with the strongest consensus among the trees. The wide `Prediction Interval` also effectively communicates the model's uncertainty.

<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/1a3eb1ee-161a-45a3-bf25-d07abe4dff10" />

## API Reference

### `analyze_prediction()`

Trains the ensemble and performs the analysis.

**Parameters:**

  * `X_train` (np.array): Training feature data.
  * `y_train` (np.array): Training target data.
  * `X_new` (np.array): The new data point(s) to predict.
  * `n_trees` (int, optional): The number of trees to train. Defaults to `500`.
  * `max_depth` (int, optional): The max depth of each tree. Defaults to `5`.
  * `trim_percent` (float, optional): Percentage to trim from each end for the trimmed mean. Defaults to `5.0`.
  * `confidence_level` (float, optional): Confidence level for the prediction interval. Defaults to `90.0`.
  * `methods` (list, optional): A list of strings specifying which methods to run. Defaults to `['mean', 'trimmed_mean', 'mode', 'prediction_interval']`.

**Available Methods:**
`'mean'`, `'trimmed_mean'`, `'mode'`, `'som'`, `'lom'`, `'cog'`, `'boa'`, `'mwm'`, `'kwa'`, `'prediction_interval'`

**Returns:**

  * `(dict)`: A dictionary containing the raw `predictions`, calculated `metrics`, and input `parameters`.

### `plot_results()`

Visualizes the results from `analyze_prediction()`.

**Parameters:**

  * `results` (dict): The output dictionary from the `analyze_prediction` function.
  * `ground_truth` (float, optional): The true value, if known, to plot for comparison.

## Understanding the Analysis Methods

| Method              | Name                  | Description                                                                                                                   |
| :------------------ | :-------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| `mean`              | Standard Mean         | The simple average of all tree predictions. The standard Random Forest output.                                                |
| `trimmed_mean`      | Trimmed Mean          | The mean calculated after removing a percentage of the smallest and largest predictions. More robust to outliers.               |
| `prediction_interval` | Prediction Interval   | The range within which a certain percentage (the confidence level) of predictions fall. Quantifies model uncertainty.         |
| `mode`              | Mode (Mean of Maxima) | The most frequent prediction value(s). Useful for finding the strongest point of consensus.                                   |
| `som`               | Smallest of Maxima    | The smallest value within the most frequent prediction bin. A conservative estimate.                                          |
| `lom`               | Largest of Maxima     | The largest value within the most frequent prediction bin. An optimistic estimate.                                            |
| `cog`               | Center of Gravity     | The "balance point" of the histogram. A robust measure of central tendency that considers the entire distribution's shape.    |
| `boa`               | Bisector of Area      | The value that splits the area of the histogram into two equal halves.                                                        |
| `mwm`               | Mode-Weighted Mean    | A weighted average where each prediction's weight is inversely proportional to its distance to the nearest mode.                |
| `kwa`               | Kernel Weighted Avg   | A weighted average where weights are generated by an RBF kernel based on distance to the nearest mode, providing a smooth falloff. |

## Citation

If you use this package in your research, you can use the following BibTeX entry to cite it:

```bibtex
@misc{huang2025robustrf,
  author       = {Yongchao Huang, Hassan Raza},
  title        = {robust_rf: A Python package for robust random forest regression analysis},
  year         = {2025},
  publisher    = {University of Aberdeen},
  note         = {yongchao.huang@abdn.ac.uk},
  howpublished = {\url{https://github.com/YongchaoHuang/robust_randomForest}}
}

