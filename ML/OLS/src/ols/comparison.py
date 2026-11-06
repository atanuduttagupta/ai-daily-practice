"""
Comparison between our OLS implementation and scikit-learn.

This module helps learners verify that our mathematical
implementation produces results similar to a standard
machine-learning library.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from ols.regression import OLSRegression
from ols.metrics import mean_squared_error, r_squared


def compare_ols_models(X, y):
    """
    Fit our OLS model and a scikit-learn model on the same data.

    Parameters
    ----------
    X : numpy.ndarray
        Input features.

    y : numpy.ndarray
        Target values.

    Returns
    -------
    dict
        Results from both models.
    """

    # ---------------------------------------------------------
    # 1. Fit our from-scratch implementation
    # ---------------------------------------------------------

    our_model = OLSRegression()
    our_model.fit(X, y)

    our_predictions = our_model.predict(X)

    # ---------------------------------------------------------
    # 2. Fit scikit-learn's LinearRegression
    # ---------------------------------------------------------

    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)

    sklearn_predictions = sklearn_model.predict(X)

    # ---------------------------------------------------------
    # 3. Calculate metrics for both models
    # ---------------------------------------------------------

    our_mse = mean_squared_error(y, our_predictions)
    our_r2 = r_squared(y, our_predictions)

    sklearn_mse = mean_squared_error(y, sklearn_predictions)
    sklearn_r2 = r_squared(y, sklearn_predictions)

    # ---------------------------------------------------------
    # 4. Return the results in a simple dictionary
    # ---------------------------------------------------------

    return {
        "our_model": {
            "intercept": our_model.intercept_,
            "coefficient": our_model.coef_[0],
            "mse": our_mse,
            "r_squared": our_r2,
        },
        "scikit_learn": {
            "intercept": float(sklearn_model.intercept_),
            "coefficient": float(sklearn_model.coef_[0]),
            "mse": sklearn_mse,
            "r_squared": sklearn_r2,
        },
    }