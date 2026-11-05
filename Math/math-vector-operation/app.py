import streamlit as st
import numpy as np

from main import (
    add_vector,
    scalar_multiply,
    dot_product_manual,
    norm_manual,
    angle_between_degrees,
)

st.title("Vectors & Vector Operations")
st.write("Explore basic vector operations from first principles with NumPy.")

col1, col2 = st.columns(2)

with col1:
    v1_values = st.text_input("Vector 1", "3,4")

with col2:
    v2_values = st.text_input("Vector 2", "1,2")

scalar = st.number_input(
    "Scalar multiplier",
    value=3.0,
)

if st.button("Calculate"):
    try:
        v1 = np.array([float(x.strip()) for x in v1_values.split(",")])
        v2 = np.array([float(x.strip()) for x in v2_values.split(",")])

        if v1.shape != v2.shape:
            st.error("Vector 1 and Vector 2 must have the same number of components.")
        elif v1.size == 0:
            st.error("Please enter at least one component for each vector.")
        elif np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            st.error("The angle between vectors is undefined when either vector is zero.")
        else:
            st.subheader("Results")

            st.write(f"**v1 + v2:** {add_vector(v1, v2)}")
            st.write(f"**scalar × v1:** {scalar_multiply(v1, scalar)}")

            manual_dot = dot_product_manual(v1, v2)
            numpy_dot = np.dot(v1, v2)

            st.write(f"**Dot product (manual):** {manual_dot}")
            st.write(f"**Dot product (NumPy):** {numpy_dot}")

            manual_norm = norm_manual(v1)
            numpy_norm = np.linalg.norm(v1)

            st.write(f"**Norm (manual):** {manual_norm}")
            st.write(f"**Norm (NumPy):** {numpy_norm}")

            angle = angle_between_degrees(v1, v2)
            st.write(f"**Angle between vectors:** {angle:.2f} degrees")

            if manual_dot == numpy_dot:
                st.success("Manual dot product matches NumPy.")
            else:
                st.error("Manual dot product does not match NumPy.")

            if np.isclose(manual_norm, numpy_norm):
                st.success("Manual norm matches NumPy.")
            else:
                st.error("Manual norm does not match NumPy.")

    except ValueError:
        st.error("Enter vector components as comma-separated numbers, for example: 3,4")
