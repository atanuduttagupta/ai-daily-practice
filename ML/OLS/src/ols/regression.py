"""
Ordinary Least Squares (OLS) Linear Regression
------------------------------------------------

This file implements linear regression from scratch using NumPy.

The purpose is educational: the code is intentionally simple and
heavily commented so that someone with basic Python knowledge can
understand how OLS works.

For a simple linear regression, we are trying to find:

    y = b0 + b1*x

where:

    y  = predicted value
    b0 = intercept
    b1 = slope / coefficient
    x  = input feature

For multiple features, the same idea becomes:

    y = b0 + b1*x1 + b2*x2 + ... + bn*xn

The mathematical OLS solution is commonly written as:

    beta = (X^T X)^-1 X^T y

In this implementation we use NumPy's pseudo-inverse:

    beta = pinv(X) @ y

This gives us the same OLS solution while avoiding an explicit
matrix inverse.
"""

import numpy as np


class OLSRegression:
    """
    A very simple implementation of Ordinary Least Squares regression.

    Example:

        model = OLSRegression()

        model.fit(X, y)

        predictions = model.predict(X)
    """

    def __init__(self):
        """
        Create an empty regression model.

        We don't know the intercept or coefficients yet because
        the model has not been trained.
        """

        # The intercept is the value of y when all input features
        # are zero.
        #
        # Example:
        #
        #     y = 10 + 2*x
        #
        # The intercept is 10.
        self.intercept_ = None

        # The coefficients are the slopes for our input features.
        #
        # For:
        #
        #     y = 10 + 2*x1 + 5*x2
        #
        # the coefficients are:
        #
        #     [2, 5]
        self.coef_ = None

    def fit(self, X, y):
        """
        Calculate the coefficients from training data.

        Parameters
        ----------
        X : array-like
            Input features.

            Example for one feature:

                [[1],
                 [2],
                 [3],
                 [4]]

        y : array-like
            Actual target values.

                [3, 5, 7, 9]

        Returns
        -------
        self
            The trained regression model.
        """

        # ---------------------------------------------------------
        # STEP 1: Convert the input data into NumPy arrays
        # ---------------------------------------------------------
        #
        # NumPy gives us useful mathematical operations on arrays
        # and matrices.
        #
        # dtype=float makes sure that our calculations use
        # floating-point numbers.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # ---------------------------------------------------------
        # STEP 2: Make sure X has the expected shape
        # ---------------------------------------------------------
        #
        # A single feature might be supplied like this:
        #
        #     [1, 2, 3, 4]
        #
        # But for matrix calculations we want:
        #
        #     [[1],
        #      [2],
        #      [3],
        #      [4]]
        #
        # Therefore, if X is one-dimensional, we turn it into
        # a two-dimensional column.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # ---------------------------------------------------------
        # STEP 3: Make y one-dimensional
        # ---------------------------------------------------------
        #
        # We expect y to contain one target value for each row
        # in X.
        #
        # ravel() converts something like:
        #
        #     [[3],
        #      [5],
        #      [7]]
        #
        # into:
        #
        #     [3, 5, 7]
        y = y.ravel()

        # ---------------------------------------------------------
        # STEP 4: Check that X and y have the same number of rows
        # ---------------------------------------------------------
        #
        # If X contains 10 training examples, y must also contain
        # 10 target values.
        #
        # Otherwise the model would not know which y value belongs
        # to which row of X.
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        # ---------------------------------------------------------
        # STEP 5: Add a column of 1s to X
        # ---------------------------------------------------------
        #
        # This is one of the most important steps to understand.
        #
        # Suppose our original X is:
        #
        #     X =
        #
        #     [[1],
        #      [2],
        #      [3],
        #      [4]]
        #
        # We add a column containing only 1s:
        #
        #     X_design =
        #
        #     [[1, 1],
        #      [1, 2],
        #      [1, 3],
        #      [1, 4]]
        #
        # Why?
        #
        # The first column represents the INTERCEPT.
        #
        # Matrix multiplication will therefore calculate:
        #
        #     1*b0 + x*b1
        #
        # which is:
        #
        #     b0 + b1*x
        #
        # This allows the intercept to be learned as part of
        # the same matrix calculation.
        ones = np.ones((X.shape[0], 1))

        X_design = np.column_stack((ones, X))

        # ---------------------------------------------------------
        # STEP 6: Calculate the OLS coefficients
        # ---------------------------------------------------------
        #
        # The traditional OLS formula is:
        #
        #     beta = (X^T X)^-1 X^T y
        #
        # where:
        #
        #     X^T = transpose of X
        #     ^-1 = matrix inverse
        #
        # We could implement that formula directly using:
        #
        #     np.linalg.inv(X_design.T @ X_design) @ \
        #         X_design.T @ y
        #
        # However, explicitly calculating a matrix inverse can
        # cause numerical problems when the matrix is singular
        # or nearly singular.
        #
        # NumPy provides pinv(), which calculates the
        # Moore-Penrose pseudo-inverse.
        #
        # Therefore we use:
        #
        #     beta = pinv(X_design) @ y
        #
        # This is still an OLS solution, but is generally safer
        # numerically.
        beta = np.linalg.pinv(X_design) @ y

        # ---------------------------------------------------------
        # STEP 7: Extract the intercept
        # ---------------------------------------------------------
        #
        # Remember that the first column of X_design contains
        # the 1s we added earlier.
        #
        # Therefore beta[0] is the intercept.
        #
        # Example:
        #
        #     beta = [10, 2]
        #
        # means:
        #
        #     intercept = 10
        #     coefficient = 2
        self.intercept_ = float(beta[0])

        # ---------------------------------------------------------
        # STEP 8: Extract the feature coefficients
        # ---------------------------------------------------------
        #
        # Everything after beta[0] represents coefficients for
        # the actual input features.
        #
        # For example:
        #
        #     beta = [10, 2, 5]
        #
        # gives:
        #
        #     intercept = 10
        #     coefficients = [2, 5]
        self.coef_ = beta[1:]

        # ---------------------------------------------------------
        # STEP 9: Return the trained model
        # ---------------------------------------------------------
        #
        # Returning self allows us to write:
        #
        #     model = OLSRegression().fit(X, y)
        #
        # instead of:
        #
        #     model = OLSRegression()
        #     model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict target values for new input data.

        Once fit() has calculated the intercept and coefficients,
        prediction is simply:

            y = intercept + X * coefficients

        Parameters
        ----------
        X : array-like
            Input features for which we want predictions.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """

        # ---------------------------------------------------------
        # STEP 1: Make sure the model has been trained
        # ---------------------------------------------------------
        #
        # We cannot make predictions until fit() has calculated
        # the intercept and coefficients.
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError(
                "Model must be fitted before making predictions."
            )

        # ---------------------------------------------------------
        # STEP 2: Convert X into a NumPy array
        # ---------------------------------------------------------
        X = np.asarray(X, dtype=float)

        # Just like in fit(), make a single feature into a
        # two-dimensional column if necessary.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # ---------------------------------------------------------
        # STEP 3: Calculate predictions
        # ---------------------------------------------------------
        #
        # The prediction equation is:
        #
        #     prediction = intercept + X @ coefficients
        #
        # The @ symbol means matrix multiplication in Python.
        #
        # For one feature this is equivalent to:
        #
        #     prediction = b0 + b1*x
        #
        # For multiple features it becomes:
        #
        #     prediction = b0 + b1*x1 + b2*x2 + ...
        predictions = self.intercept_ + X @ self.coef_

        return predictions