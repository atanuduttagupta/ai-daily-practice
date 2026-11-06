"""
Interactive Streamlit experience for ML-001.

This app lets a learner:
    1. Generate synthetic regression data.
    2. Train our OLS implementation.
    3. View the learned coefficients.
    4. View MSE and R-squared.
    5. See the regression line visually.
"""

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

from ols.comparison import compare_ols_models
from ols.dataset import create_regression_data
from ols.regression import OLSRegression
from ols.visualization import plot_regression_line


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="ML-001 — OLS Regression",
    page_icon="📈",
    layout="wide",
)


# ------------------------------------------------------------
# Title and introduction
# ------------------------------------------------------------

st.title("📈 ML-001 — Ordinary Least Squares Regression")

st.write(
    """
    This interactive experience demonstrates Ordinary Least Squares
    regression from scratch using NumPy.

    You can generate synthetic data, train the model, inspect the
    learned parameters, and compare our implementation with
    scikit-learn.
    """
)


# ------------------------------------------------------------
# Sidebar — dataset controls
# ------------------------------------------------------------

st.sidebar.header("Dataset Settings")

number_of_samples = st.sidebar.slider(
    "Number of samples",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
)

noise_level = st.sidebar.slider(
    "Noise level",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=1.0,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10000,
    value=42,
)


# ------------------------------------------------------------
# Generate the dataset
# ------------------------------------------------------------

X, y = create_regression_data(
    number_of_samples=number_of_samples,
    noise_level=noise_level,
    random_seed=random_seed,
)


# ------------------------------------------------------------
# Train our OLS model
# ------------------------------------------------------------

model = OLSRegression()
model.fit(X, y)

predictions = model.predict(X)


# ------------------------------------------------------------
# Display learned model parameters
# ------------------------------------------------------------

st.header("Learned Model")

column1, column2 = st.columns(2)

with column1:
    st.metric(
        "Intercept",
        f"{model.intercept_:.3f}",
    )

with column2:
    st.metric(
        "Coefficient",
        f"{model.coef_[0]:.3f}",
    )


st.write(
    f"""
    **Learned equation:**

    `y = {model.intercept_:.3f} + {model.coef_[0]:.3f} × x`
    """
)


# ------------------------------------------------------------
# Display evaluation metrics
# ------------------------------------------------------------

st.header("Model Evaluation")

comparison = compare_ols_models(X, y)

our_results = comparison["our_model"]

metric1, metric2 = st.columns(2)

with metric1:
    st.metric(
        "Mean Squared Error",
        f"{our_results['mse']:.3f}",
    )

with metric2:
    st.metric(
        "R-squared",
        f"{our_results['r_squared']:.3f}",
    )


# ------------------------------------------------------------
# Display regression visualization
# ------------------------------------------------------------

st.header("Regression Visualization")

figure = plot_regression_line(X, y, model)

st.pyplot(figure)


# ------------------------------------------------------------
# Compare with scikit-learn
# ------------------------------------------------------------

st.header("Our OLS vs scikit-learn")

comparison_column1, comparison_column2 = st.columns(2)

with comparison_column1:
    st.subheader("Our implementation")

    st.write(
        f"Intercept: `{our_results['intercept']:.6f}`"
    )

    st.write(
        f"Coefficient: `{our_results['coefficient']:.6f}`"
    )

with comparison_column2:
    st.subheader("scikit-learn")

    sklearn_results = comparison["scikit_learn"]

    st.write(
        f"Intercept: `{sklearn_results['intercept']:.6f}`"
    )

    st.write(
        f"Coefficient: `{sklearn_results['coefficient']:.6f}`"
    )


# ------------------------------------------------------------
# Learning note
# ------------------------------------------------------------

st.info(
    """
    **Learning note:** Try increasing the noise level.

    You should see the data points become more scattered around
    the regression line. The estimated coefficients may also move
    farther away from the original values used to generate the data.
    """
)