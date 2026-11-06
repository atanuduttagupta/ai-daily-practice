"""
Small visual demonstration of ML-001.

This script:
1. Creates synthetic regression data.
2. Fits our OLS model.
3. Displays the regression line.
"""

import matplotlib.pyplot as plt

from ols.dataset import create_regression_data
from ols.regression import OLSRegression
from ols.visualization import plot_regression_line


def main():
    # Create our reproducible synthetic dataset.
    X, y = create_regression_data(
        number_of_samples=50,
        random_seed=42,
    )

    # Create and train our OLS model.
    model = OLSRegression()
    model.fit(X, y)

    # Create the regression plot.
    figure = plot_regression_line(X, y, model)

    # Display the figure on the screen.
    #
    # Unlike our reusable visualization function,
    # this executable demo is specifically responsible
    # for displaying the chart.
    plt.show()


if __name__ == "__main__":
    main()