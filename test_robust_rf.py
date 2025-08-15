import numpy as np
from robust_rf import analyze_prediction

def test_analyze_prediction_runs():
    """
    Tests if the main analysis function runs without errors and returns the correct structure.
    """
    # 1. Create minimal dummy data for the test
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 1, 1, 0])
    X_new = np.array([1.5])

    # 2. Run the analysis with a small number of trees for speed
    results = analyze_prediction(
        X_train, y_train, X_new,
        n_trees=10,
        methods=['mean', 'mode']
    )

    # 3. Assert that the output is structured correctly
    assert 'predictions' in results
    assert 'metrics' in results
    assert 'mean' in results['metrics']
    assert 'mode' in results['metrics']
    assert len(results['predictions']) == 10
