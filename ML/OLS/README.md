# ML-001 — Ordinary Least Squares Regression from Scratch

ML-001 is a hands-on machine learning project that implements
**Ordinary Least Squares (OLS) Linear Regression from scratch using NumPy**.

The goal is not just to use a machine learning library.

The goal is to understand what is happening underneath a
linear regression model.

We will:

1. Understand the mathematics behind OLS.
2. Implement OLS using NumPy.
3. Make predictions with our model.
4. Implement MSE and R² ourselves.
5. Compare our results with scikit-learn.
6. Explore the model through an interactive Streamlit application.

---

## What Will You Learn?

After completing this project, you should be able to:

- Explain what linear regression is.
- Understand the difference between features and targets.
- Understand the role of the intercept and coefficients.
- Understand the OLS equation.
- Explain what a design matrix is.
- Implement linear regression using NumPy.
- Understand why we use the pseudo-inverse.
- Generate predictions from a trained model.
- Calculate Mean Squared Error (MSE).
- Calculate R².
- Compare a from-scratch implementation with scikit-learn.

---

## Prerequisites

You do **not** need advanced Python or advanced mathematics.

You should be comfortable with:

### Python

Basic knowledge of:

- Variables
- Lists
- Functions
- Classes
- Imports
- Basic `if` statements
- Basic loops

### Mathematics

You should be comfortable with:

- Basic algebra
- Addition and multiplication
- Squares
- A basic idea of rows and columns in a matrix

You do **not** need to know the OLS formula before starting.

We will explain the important mathematics as we build the implementation.

---

## Software Requirements

You need:

- Python 3.13 or later
- NumPy
- scikit-learn
- Streamlit
- pytest

You do not need to install these packages individually.

The project contains its own `pyproject.toml` file that defines
the dependencies required by ML-001.

---

## Installation

Open a terminal and navigate to the ML-001 directory:

```powershell
cd projects\ml-001
```

Install the project and its development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

This installs the packages required by the project.

The important idea is that **ML-001 manages its own dependencies**.

You do not need to understand the AI Project Atlas dependency setup
to run this project.

---

## Project Structure

```text
ml-001/
│
├── project.yaml
│       Atlas metadata describing this project
│
├── README.md
│       This learning guide
│
├── ai-context.md
│       Additional context for AI-assisted discovery
│
├── pyproject.toml
│       Python package and dependency configuration
│
├── src/
│   └── ols/
│       ├── __init__.py
│       ├── regression.py
│       │       Our OLS implementation
│       │
│       └── metrics.py
│               Our MSE and R² implementations
│
├── tests/
│   └── test_regression.py
│           Automated tests for our implementation
│
├── notebooks/
│       Exploratory notebooks will go here
│
├── results/
│       Generated results will go here
│
└── assets/
        Supporting images and other assets
```

---

# Understanding the OLS Model

For a simple linear regression with one feature, we are trying to
find a line that best describes our data:

```text
y = b0 + b1*x
```

Where:

- `y` is the predicted value.
- `x` is the input feature.
- `b0` is the intercept.
- `b1` is the coefficient (slope).

For example:

```text
y = 2 + 3*x
```

means:

- intercept = `2`
- coefficient = `3`

If:

```text
x = 5
```

then:

```text
y = 2 + (3 * 5)
  = 17
```

The implementation in this project teaches how those values are
learned from data rather than manually providing them.

---

## The OLS Equation

For multiple features, the OLS solution is commonly written as:

```text
β = (XᵀX)⁻¹Xᵀy
```

This may look intimidating at first.

Don't worry.

The project breaks the calculation into understandable steps.

We also use NumPy's pseudo-inverse:

```python
beta = np.linalg.pinv(X_design) @ y
```

instead of explicitly calculating:

```text
(XᵀX)⁻¹
```

The pseudo-inverse gives us a numerically safer way to calculate
the solution, particularly when a matrix cannot be safely inverted.

---

# Understanding the Implementation

The main implementation is in:

```text
src/ols/regression.py
```

The code contains extensive comments explaining the connection
between the mathematics and the Python implementation.

The important flow is:

```text
Training data
     ↓
Create design matrix
     ↓
Calculate OLS coefficients
     ↓
Separate intercept and coefficients
     ↓
Trained model
     ↓
Make predictions
```

---

# Model Evaluation

A model can make predictions, but we also need to know how good
those predictions are.

ML-001 implements two common regression metrics.

## Mean Squared Error (MSE)

MSE measures the average squared difference between actual and
predicted values.

Conceptually:

```text
Actual values
      ↓
Prediction errors
      ↓
Square the errors
      ↓
Average them
      ↓
MSE
```

Generally, a smaller MSE means the predictions are closer to the
actual values.

The implementation is in:

```text
src/ols/metrics.py
```

---

## R²

R² helps us understand how much of the variation in the target
values is explained by the regression model.

A useful simplified interpretation is:

```text
R² = 1
    Perfect predictions

R² = 0
    No improvement over predicting the average

R² < 0
    Can be worse than that simple average prediction
```

Again, the project implements the calculation ourselves so that
you can understand what the metric is doing.

---

# Running the Tests

From the **AI Project Atlas repository root**, run:

```powershell
python -m pytest projects\ml-001\tests\test_regression.py -v
```

You should see four tests:

```text
test_ols_finds_simple_line
test_ols_predicts_correct_values
test_mean_squared_error
test_r_squared_for_perfect_predictions
```

All four should pass.

---

# Why Do We Test With a Simple Equation?

One of our first datasets follows this exact relationship:

```text
y = 2 + 3*x
```

For example:

```text
x    y
---------
1    5
2    8
3    11
4    14
5    17
```

Because we already know the correct answer, we can check whether
our implementation learns:

```text
intercept = 2
coefficient = 3
```

This is much easier to understand than immediately testing the
model on a large, complicated dataset.

Once the basic implementation works, we can move toward more
realistic synthetic data.

---

# Comparing With scikit-learn

ML-001 intentionally does not stop at the from-scratch
implementation.

We will also use scikit-learn as an independent reference.

The comparison will help answer:

> "Does our implementation produce the same result as a
> well-established machine learning library?"

This gives us both:

```text
Understanding
     +
Verification
```

rather than simply trusting that our implementation is correct.

---

# Interactive Experience

ML-001 will eventually include an interactive Streamlit application.

The application will allow you to experiment with regression
using synthetic data.

You will be able to see things such as:

- Generated data
- Regression line
- Predictions
- Learned coefficients
- Intercept
- MSE
- R²
- Comparison with scikit-learn

The interactive application is intended to make the mathematics
more intuitive by allowing you to change the data and observe
what happens.

---

# Learning Path

A recommended order for studying this project is:

```text
1. Read the OLS explanation
          ↓
2. Look at regression.py
          ↓
3. Understand the design matrix
          ↓
4. Understand the coefficient calculation
          ↓
5. Run the tests
          ↓
6. Study MSE and R²
          ↓
7. Compare with scikit-learn
          ↓
8. Experiment with synthetic data
          ↓
9. Use the Streamlit application
```

Do not worry if the matrix notation looks unfamiliar at first.

The purpose of ML-001 is to make that notation understandable
by connecting each mathematical step to simple Python code.

---

# Project Status

ML-001 is currently under development.

Implemented:

- [x] Basic OLS regression
- [x] Prediction
- [x] MSE
- [x] R²
- [x] Automated tests
- [x] Project-specific dependency configuration

Planned:

- [ ] Synthetic dataset generation
- [ ] scikit-learn comparison
- [ ] Regression visualization
- [ ] Streamlit interactive experience
- [ ] Final project documentation
- [ ] Atlas Experience metadata
