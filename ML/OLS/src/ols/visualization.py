"""
Visualization helpers for the ML-001 OLS project.

The functions in this module are intentionally simple.

They help learners visualize:
    - the original data points
    - the regression line
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_regression_line(X, y, model):
    """
    Plot data points and the regression line.

    Parameters
    ----------
    X : numpy.ndarray
        Input feature values.

    y : numpy.ndarray
        Actual target values.

    model : OLSRegression
        A fitted OLS regression model.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """

    # Make sure X is a NumPy array.
    X = np.asarray(X, dtype=float)

    # Our current project uses one input feature.
    # X[:, 0] gives us that feature as a one-dimensional array.
    x_values = X[:, 0]

    # Generate predictions using the fitted model.
    predictions = model.predict(X)

    # Create a figure and axes.
    figure, axes = plt.subplots()

    # Plot the original observations.
    axes.scatter(
        x_values,
        y,
        label="Actual data",
    )

    # Plot the regression line.
    axes.plot(
        x_values,
        predictions,
        label="OLS regression line",
    )

    # Add labels so the chart is understandable.
    axes.set_xlabel("Input feature (X)")
    axes.set_ylabel("Target (y)")
    axes.set_title("Ordinary Least Squares Regression")

    # Show a legend explaining the two plotted elements.
    axes.legend()

    # Return the figure instead of displaying it here.
    #
    # This makes the function reusable by:
    #     - tests
    #     - notebooks
    #     - Streamlit
    #     - other future interfaces
    return figure