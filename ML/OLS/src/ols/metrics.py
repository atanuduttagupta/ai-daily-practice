"""
Regression Evaluation Metrics
-----------------------------

This file contains simple implementations of two common
regression metrics:

1. Mean Squared Error (MSE)
2. R-squared (R²)

We implement them ourselves so that learners can see the
mathematics behind the metrics instead of treating them as
black-box functions.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).

    MSE tells us how far the model's predictions are from
    the actual values, on average.

    Formula:

        MSE = average((actual - predicted)²)

    Example:

        actual    = [10, 20, 30]
        predicted = [11, 18, 31]

    First calculate the errors:

        [-1, 2, -1]

    Square the errors:

        [1, 4, 1]

    Average them:

        (1 + 4 + 1) / 3 = 2

    A smaller MSE generally means the predictions are closer
    to the actual values.
    """

    # Convert both inputs to NumPy arrays.
    #
    # This allows us to perform the mathematical operations
    # element by element.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Calculate the difference between actual and predicted
    # values.
    errors = y_true - y_pred

    # Square every error.
    #
    # We square the errors so that positive and negative errors
    # don't cancel each other out.
    squared_errors = errors ** 2

    # Calculate the average of all squared errors.
    mse = np.mean(squared_errors)

    return float(mse)


def r_squared(y_true, y_pred):
    """
    Calculate R-squared (R²).

    R² tells us how much of the variation in the target values
    is explained by our regression model.

    A simplified interpretation is:

        R² = 1.0
            Perfect predictions.

        R² = 0.0
            The model is no better than simply predicting
            the average of the actual values.

        R² < 0.0
            The model can be worse than that simple average
            prediction.

    Formula:

              Sum of squared prediction errors
        R² = 1 - ------------------------------
              Total sum of squared variation

    In mathematical notation:

              Σ(y - ŷ)²
        R² = 1 - ---------
              Σ(y - ȳ)²

    where:

        y  = actual value
        ŷ  = predicted value
        ȳ  = average of actual values
    """

    # Convert the inputs into NumPy arrays.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Calculate the average of the actual target values.
    #
    # This average gives us a simple baseline:
    #
    # "What if we ignored X completely and always predicted
    # the average y?"
    y_mean = np.mean(y_true)

    # ---------------------------------------------------------
    # STEP 1: Calculate the total variation in the actual data
    # ---------------------------------------------------------
    #
    # For every actual value, calculate how far it is from
    # the average.
    #
    # Then square those differences and add them together.
    #
    # This is called the Total Sum of Squares (SST).
    total_sum_of_squares = np.sum((y_true - y_mean) ** 2)

    # ---------------------------------------------------------
    # STEP 2: Calculate the model's prediction error
    # ---------------------------------------------------------
    #
    # Calculate the difference between the actual values and
    # the model's predictions.
    #
    # Then square those errors and add them together.
    #
    # This is called the Residual Sum of Squares (SSR).
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)

    # ---------------------------------------------------------
    # STEP 3: Calculate R²
    # ---------------------------------------------------------
    #
    # The formula is:
    #
    #             residual_sum_of_squares
    #     R² = 1 - ------------------------
    #             total_sum_of_squares
    #
    # If the model predicts perfectly:
    #
    #     residual error = 0
    #
    # so:
    #
    #     R² = 1
    #
    r2 = 1 - (
        residual_sum_of_squares
        / total_sum_of_squares
    )

    return float(r2)