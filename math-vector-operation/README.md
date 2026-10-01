# Vectors & Vector Operations

A small beginner-friendly mathematics project demonstrating fundamental vector operations from first principles using Python and NumPy.

The project is designed as part of the mathematical foundation for machine learning.

## Project Details

- **Project ID:** `daily-math-project`
- **Domain:** Mathematics & Foundations
- **Project Type:** `mathematics`
- **Difficulty:** Beginner
- **Technology:** Python + NumPy
- **Interactive UI:** Streamlit

## What This Project Demonstrates

```text
Vector 1 + Vector 2
        |
        +--> Vector Addition

Vector × Scalar
        |
        +--> Scalar Multiplication

Vector 1, Vector 2
        |
        +--> Dot Product
        |
        +--> Norm
        |
        +--> Angle Between Vectors
```

The project deliberately implements some operations manually and then checks them against NumPy.

## Features

- Vector addition
- Scalar multiplication
- Manual dot product
- NumPy dot-product comparison
- Manual Euclidean norm
- NumPy norm comparison
- Angle calculation between two vectors
- Floating-point protection for cosine calculation
- Command-line execution
- Interactive Streamlit interface

## Project Structure

```text
daily-math-project/
├── app.py
├── main.py
├── project.yaml
├── ai-context.md
├── README.md
├── requirements.txt
├── .gitignore
└── .env                  # Not required for this project
```

## Requirements

- Python 3.12+ recommended
- NumPy
- Streamlit

## Local Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the Python Example

```powershell
python main.py
```

The example uses:

```text
v1 = [3, 4]
v2 = [1, 2]
```

It prints vector addition, scalar multiplication, the manual and NumPy dot products, the manual and NumPy norms, and the angle between the vectors.

## Run the Streamlit Application

```powershell
streamlit run app.py
```

The Streamlit interface allows you to enter vectors and a scalar and interactively calculate the operations.

## Manual vs NumPy Validation

The project intentionally shows what common NumPy operations represent mathematically.

For the dot product:

```text
manual implementation
        ↓
sum of element-wise products
        ↓
compare with np.dot()
```

For the norm:

```text
manual implementation
        ↓
sqrt(sum of squared components)
        ↓
compare with np.linalg.norm()
```

This makes the relationship between mathematical definitions and library functions explicit.

## Angle Calculation

The angle uses the dot-product identity:

```text
v1 · v2 = |v1| |v2| cos(theta)
```

The cosine value is clipped to `[-1, 1]` before applying `arccos` to reduce the effect of tiny floating-point errors.

## Learning Outcomes

After completing this project, the learner should understand:

- What a vector represents.
- How vectors can be added.
- How scalar multiplication changes a vector.
- How the dot product is calculated.
- What the dot product tells us about vector direction.
- How Euclidean norm represents vector magnitude.
- How the angle between two vectors can be derived from the dot product.
- How NumPy implements common numerical operations.
- Why first-principles implementations are useful when learning mathematical foundations.

## What Is Not Included

This project does not currently implement:

- Matrix operations
- Eigenvalues or eigenvectors
- Calculus
- Gradient descent
- Probability and statistics
- Machine-learning algorithms
- Deep learning
- Generative AI
- RAG
- Agents

These belong to later learning stages.

## AI Project Atlas

This project is intended to be catalogued in AI Project Atlas as a mathematical foundation project.

It provides concepts that are useful for later work involving:

```text
Vectors
   ↓
Linear Algebra
   ↓
Machine Learning
   ↓
Neural Networks
   ↓
Embeddings
   ↓
Generative AI
```

## Security

This project does not require API keys or external credentials.

No secrets should be added to the source code.

## License

Add the repository's standard project license here when one is established.
