"""
P0 - Day 1 - Track A (Linear Algebra & Calculus): Subtopic 1 - Vectors & vector operations

Goal: build intuition for what a vector actually IS and DOES before touching
any ML library shortcuts - addition, scalar multiplication, dot product,
norm (magnitude), and the angle between two vectors, all from first
principles with NumPy, plus a check against NumPy's built-in functions.

Setup:
    Create venv: python -m venv .venv , .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python main.py
"""

import numpy as np


def add_vector ( v1: np.ndarray, v2: np.ndarray ) -> np.ndarray:
    """Element-wise addition - moving by v1 then by v2 lands you at v1+v2."""
    return v1 + v2

def scalar_multiply( v: np.ndarray, scalar: float) -> np.ndarray:
    """Stretches (scalar > 1) or shrinks (0 < scalar < 1) a vector; flips
    direction if scalar is negative."""    
    return scalar * v


def dot_product_manual(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Sum of element-wise products - measures how much two vectors point
    in the same direction. Implemented by hand to show what np.dot does
    under the hood."""
    total = 0.0
    for a,b in zip(v1, v2):
        total += (a * b)
    return total    

def norm_manual(v: np.ndarray) -> float:
    """Euclidean length of a vector: sqrt of the sum of squared components."""
    return sum(x**2 for x in v) ** .5

def angle_between_degrees(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Uses the dot product identity: v1 . v2 = |v1| |v2| cos(theta)."""
    cos_theta = np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2)  ) 
    # clip to handle tiny floating-point errors pushing cos slightly past [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def main():
    v1 = np.array([3, 4])
    v2 = np.array([1, 2])

    print(f"v1 = {v1}")
    print(f"v2 = {v2}\n")

    print(f"v1 + v2            = {add_vector(v1, v2)}")
    print(f"3 * v1             = {scalar_multiply(v1, 3)}\n")

    manual_dot = dot_product_manual(v1, v2)
    numpy_dot = np.dot(v1, v2)
    print(f"dot product (manual) = {manual_dot}")
    print(f"dot product (NumPy)  = {numpy_dot}")
    assert manual_dot == numpy_dot, "Manual and NumPy dot products disagree!"    

    manual_norm = norm_manual(v1)
    numpy_norm = np.linalg.norm(v1)
    print(f"Norm (manual) = {manual_norm}")
    print(f"Norm (NumPy)  = {numpy_norm}")
    assert manual_norm == numpy_norm, "Manual and NumPy Norms disagree!"  

    angle = angle_between_degrees(v1, v2)
    print(f"\nAngle between v1 and v2 = {angle:.2f} degrees")


if __name__ == "__main__":
    main()

