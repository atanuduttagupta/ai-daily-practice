"""
Streamlit learning interface for the Day 9 CNN Basics project.

Place this file beside main.py and run:
    python -m streamlit run app.py --server.runOnSave true
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from main import (
    SEED,
    VERTICAL_EDGE,
    HORIZONTAL_EDGE,
    TinyCNN,
    Adam,
    conv2d_loop,
    conv2d_fast,
    gradient_check,
)

st.set_page_config(
    page_title="CNN Basics | Day 9",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 CNN Basics — From Convolution to Learning")
st.caption("Day 9 · Track I: Computer Vision & Sequence Models · NumPy implementation")

st.markdown(
    """
This app turns the original terminal-based exercise into an interactive walkthrough.
The CNN itself is still implemented with **NumPy and hand-written backpropagation**;
Streamlit is only the learning interface.
"""
)


@st.cache_data
def get_digits():
    digits = load_digits()
    return digits.images / 16.0, digits.target


@st.cache_data(show_spinner=False)
def train_experiment(n_filters: int, epochs: int, learning_rate: float, batch_size: int):
    images, labels = get_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    Y_train = np.eye(10)[y_train]

    model = TinyCNN(n_filters=n_filters)
    grad_error = gradient_check(model, X_train[:8], Y_train[:8])

    optimizer = Adam(model.params, lr=learning_rate)
    rng = np.random.default_rng(SEED)
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            optimizer.step(
                model.gradients(X_train[batch_indices], Y_train[batch_indices])
            )

        train_loss = model.loss(X_train, Y_train)
        test_accuracy = float((model.predict(X_test) == y_test).mean())
        history.append(
            {
                "Epoch": epoch,
                "Training loss": train_loss,
                "Test accuracy": test_accuracy,
            }
        )

    cnn_predictions = model.predict(X_test)
    cnn_accuracy = float((cnn_predictions == y_test).mean())
    cnn_parameter_count = int(sum(parameter.size for parameter in model.params))

    baseline = LogisticRegression(max_iter=2000)
    baseline.fit(X_train.reshape(len(X_train), -1), y_train)
    baseline_accuracy = float(
        baseline.score(X_test.reshape(len(X_test), -1), y_test)
    )
    baseline_parameter_count = int(
        baseline.coef_.size + baseline.intercept_.size
    )

    return {
        "history": history,
        "gradient_error": float(grad_error),
        "cnn_accuracy": cnn_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "cnn_parameters": cnn_parameter_count,
        "baseline_parameters": baseline_parameter_count,
        "learned_filters": model.W.copy(),
    }


images, labels = get_digits()

tab_conv, tab_params, tab_train, tab_theory = st.tabs(
    [
        "1 · Convolution explorer",
        "2 · Parameter sharing",
        "3 · Train the tiny CNN",
        "4 · Concepts & formulas",
    ]
)

with tab_conv:
    st.subheader("What does a convolution filter detect?")
    left, right = st.columns([1, 2])
    with left:
        image_index = st.slider(
            "Choose a handwritten digit",
            min_value=0,
            max_value=len(images) - 1,
            value=0,
        )
        filter_choice = st.radio(
            "Choose an edge filter",
            ["Vertical-edge (Sobel-X)", "Horizontal-edge (Sobel-Y)"],
        )

    image = images[image_index]
    kernel = VERTICAL_EDGE if filter_choice.startswith("Vertical") else HORIZONTAL_EDGE
    feature_map = conv2d_loop(image, kernel)
    fast_map = conv2d_fast(image, kernel)

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual digit", int(labels[image_index]))
        c2.metric("Input shape", "8 × 8")
        c3.metric(
            "Feature-map shape",
            f"{feature_map.shape[0]} × {feature_map.shape[1]}",
        )

        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("**Input image**")
            st.image(image, clamp=True, width=180)
        with v2:
            st.markdown("**3 × 3 filter**")
            st.dataframe(
                pd.DataFrame(kernel),
                hide_index=True,
                use_container_width=True,
            )
        with v3:
            st.markdown("**Feature map**")
            st.dataframe(
                pd.DataFrame(np.round(feature_map, 2)),
                hide_index=True,
                use_container_width=True,
            )

    if np.allclose(feature_map, fast_map):
        st.success("Loop and vectorised implementations agree.")
    else:
        st.error("The two implementations differ — investigate the calculation.")

    st.info(
        "Each feature-map value is the sum of element-wise products between "
        "the 3 × 3 filter and the image patch underneath it. Large positive or "
        "negative values indicate a strong match to the filter pattern."
    )

with tab_params:
    st.subheader("Why parameter sharing matters")
    st.write(
        "The same 3 × 3 filter weights are reused at every image position. "
        "A fully connected layer would need separate weights for each connection."
    )
    n_filters_demo = st.slider("Number of convolution filters", 1, 16, 4)
    kernel_size = 3
    conv_parameters = n_filters_demo * kernel_size * kernel_size + n_filters_demo
    output_side = 8 - kernel_size + 1
    output_features = output_side * output_side * n_filters_demo
    dense_parameters = 8 * 8 * output_features + output_features

    a, b, c = st.columns(3)
    a.metric("Convolution parameters", f"{conv_parameters:,}")
    b.metric("Dense parameters (same output size)", f"{dense_parameters:,}")
    c.metric("Fewer parameters", f"{dense_parameters / conv_parameters:,.1f}×")

    st.markdown("**Parameter formulas**")
    st.latex(r"(K \times K \times C_{in} + 1) \times C_{out}")
    st.latex(r"(N_{in} \times N_{out}) + N_{out}")
    st.caption(
        "For this grayscale example: K = 3, C_in = 1, and C_out is the "
        "number of filters. The dense comparison maps 64 input pixels to "
        "the same number of output values as the convolution layer."
    )

with tab_train:
    st.subheader("Train a tiny CNN with hand-written backpropagation")
    st.write(
        "The model uses conv → ReLU → flatten → dense → softmax. "
        "No PyTorch or TensorFlow is used."
    )

    with st.expander("Training settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_filters = st.slider("CNN filters", 1, 12, 4)
        with col2:
            epochs = st.slider("Epochs", 5, 50, 30, step=5)
        with col3:
            learning_rate = st.select_slider(
                "Learning rate",
                options=[0.001, 0.003, 0.005, 0.01, 0.02],
                value=0.01,
            )
        batch_size = st.select_slider(
            "Batch size",
            options=[16, 32, 64, 128],
            value=64,
        )

    if st.button("Train / run experiment", type="primary"):
        with st.spinner("Training the NumPy CNN and logistic-regression baseline..."):
            result = train_experiment(
                n_filters,
                epochs,
                learning_rate,
                batch_size,
            )
        st.session_state["cnn_experiment_result"] = result
        st.session_state["cnn_experiment_settings"] = (
            n_filters,
            epochs,
            learning_rate,
            batch_size,
        )

    result = st.session_state.get("cnn_experiment_result")
    if result is not None:
        settings = st.session_state.get("cnn_experiment_settings")
        st.caption(
            f"Current result: {settings[0]} filters · {settings[1]} epochs · "
            f"learning rate {settings[2]} · batch size {settings[3]}"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("CNN test accuracy", f"{result['cnn_accuracy']:.1%}")
        m2.metric("Logistic regression", f"{result['baseline_accuracy']:.1%}")
        m3.metric("CNN parameters", f"{result['cnn_parameters']:,}")

        st.markdown("**Gradient check**")
        st.write(
            f"Worst relative error: `{result['gradient_error']:.2e}` "
            "(the original exercise expects this to be below 1e-4)."
        )
        if result["gradient_error"] < 1e-4:
            st.success("Gradient check passed.")
        else:
            st.warning("Gradient check exceeded the expected tolerance.")

        history_df = pd.DataFrame(result["history"]).set_index("Epoch")
        st.markdown("**Training history**")
        st.line_chart(history_df[["Training loss"]])
        st.line_chart(history_df[["Test accuracy"]])

        comparison = pd.DataFrame(
            [
                {
                    "Model": "Tiny CNN",
                    "Parameters": result["cnn_parameters"],
                    "Test accuracy": result["cnn_accuracy"],
                },
                {
                    "Model": "Logistic regression",
                    "Parameters": result["baseline_parameters"],
                    "Test accuracy": result["baseline_accuracy"],
                },
            ]
        )
        display_comparison = comparison.copy()
        display_comparison["Test accuracy"] = display_comparison[
            "Test accuracy"
        ].map(lambda value: f"{value:.1%}")
        st.markdown("**Model comparison**")
        st.dataframe(
            display_comparison,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**Learned convolution filters**")
        filter_columns = st.columns(min(len(result["learned_filters"]), 4))
        for index, (column, learned_filter) in enumerate(
            zip(filter_columns, result["learned_filters"][:4])
        ):
            with column:
                st.caption(f"Filter {index + 1}")
                st.dataframe(
                    pd.DataFrame(np.round(learned_filter, 3)),
                    hide_index=True,
                    use_container_width=True,
                )

        st.caption(
            "Test accuracy is shown during training for learning purposes. "
            "For rigorous model selection, use a separate validation set and "
            "keep the test set untouched until final evaluation."
        )
    else:
        st.info("Choose the settings and click **Train / run experiment**.")

with tab_theory:
    st.subheader("CNN concepts from today's notes")
    with st.expander("1. Local receptive fields", expanded=True):
        st.write(
            "Each output position looks at only a small local patch of the input. "
            "A 3 × 3 filter sees nine pixels at a time."
        )
    with st.expander("2. Parameter sharing"):
        st.write(
            "The same filter weights are applied at every image location. "
            "This lets the network detect a pattern wherever it appears."
        )
    with st.expander("3. ReLU and feature maps"):
        st.latex(r"\operatorname{ReLU}(x) = \max(0, x)")
        st.write(
            "ReLU keeps positive activations and sets negative activations to zero."
        )
    with st.expander("4. Output-size formula"):
        st.latex(r"\left\lfloor \frac{W-K+2P}{S} \right\rfloor + 1")
        st.write(
            "For W = 8, K = 3, P = 0 and S = 1, the output side length is 6."
        )
    with st.expander("5. CNN pipeline"):
        st.code(
            "8×8 image → 4 filters (3×3) → 4 feature maps (6×6) "
            "→ ReLU → flatten (144 values) → dense → softmax (10 digits)",
            language="text",
        )
    st.warning(
        "This is an intentionally small teaching model: it has one convolution "
        "layer and no pooling. Padding and pooling are the next topic in the track."
    )

st.divider()
st.caption(
    "Dataset: scikit-learn digits (1,797 grayscale images, 8 × 8 pixels). "
    "Model implementation: NumPy. UI: Streamlit."
)
