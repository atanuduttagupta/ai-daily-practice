"""
Synthetic dataset generation for the ML-001 OLS project.

The purpose of this module is to create a small dataset that we can
use to demonstrate linear regression.

We will generate data using the idea:

    y = intercept + slope * x + noise

For example:

    y = 10 + 3*x + noise

The noise makes the dataset more realistic because real-world data
usually does not fall perfectly on a straight line.
"""

import numpy as np


def create_regression_data(
    number_of_samples=50,
    intercept=10.0,
    slope=3.0,
    noise_level=5.0,
    random_seed=42,
):
    """
    Create a simple one-feature regression dataset.

    Parameters
    ----------
    number_of_samples : int
        Number of data points to generate.

    intercept : float
        The true intercept used to generate the target values.

    slope : float
        The true slope used to generate the target values.

    noise_level : float
        Controls how much random noise is added to the target.

    random_seed : int
        Seed used to make the generated dataset reproducible.

    Returns
    -------
    X : numpy.ndarray
        Input feature values.

    y : numpy.ndarray
        Target values.
    """

    # Create a random number generator.
    #
    # Using a fixed seed means that we get the same dataset every
    # time we run the function. This is very useful for learning,
    # debugging, and testing.
    random_generator = np.random.default_rng(random_seed)

    # Create evenly spaced feature values.
    #
    # For example, if number_of_samples is 5, this produces:
    #
    # [1, 2, 3, 4, 5]
    #
    # We reshape the result into a two-dimensional array because
    # machine-learning datasets normally represent:
    #
    # rows    = samples
    # columns = features
    X = np.linspace(1, 10, number_of_samples).reshape(-1, 1)

    # Generate random noise.
    #
    # Real-world observations usually contain some variation that
    # cannot be explained perfectly by the model.
    noise = random_generator.normal(
        loc=0.0,
        scale=noise_level,
        size=number_of_samples,
    )

    # Generate the target values using the regression equation:
    #
    # y = intercept + slope*x + noise
    #
    # X[:, 0] selects the first (and currently only) feature column.
    y = intercept + slope * X[:, 0] + noise

    return X, y