"""
P0 - Day 9 - Track I (CV & Sequence Models): Subtopic 1 - CNN basics

Goal: understand what a convolutional neural network actually computes, using
only NumPy (no deep-learning framework to install). You will:
  1. implement the 2D convolution operation by hand and see what a filter does
  2. compare parameter counts: convolution vs a fully connected layer
  3. train a tiny CNN (conv -> ReLU -> flatten -> dense -> softmax) on the
     8x8 handwritten digits that ship with scikit-learn, with hand-written
     backpropagation, and prove the gradients are correct
  4. compare it with a plain logistic regression baseline

Setup:
    pip install -r requirements.txt
    python main.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
np.set_printoptions(precision=1, suppress=True, linewidth=120)


# ---------------------------------------------------------------- 1) convolution
def conv2d_loop(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide the kernel over the image; at each position multiply element-wise
    and sum. 'Valid' convolution: no padding, stride 1, so the output is
    smaller: (H - kh + 1) x (W - kw + 1).
    (Deep-learning 'convolution' is technically cross-correlation: the kernel
    is not flipped. Frameworks all do it this way.)"""
    H, W = image.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


def conv2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same result without Python loops, using sliding windows + einsum."""
    windows = sliding_window_view(image, kernel.shape)
    return np.einsum("ijab,ab->ij", windows, kernel)


VERTICAL_EDGE = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)   # Sobel-x
HORIZONTAL_EDGE = VERTICAL_EDGE.T                                              # Sobel-y


# ------------------------------------------------------------- 3) a tiny CNN
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TinyCNN:
    """Input (N, 8, 8) -> conv (F filters, 3x3) -> ReLU -> flatten -> dense -> softmax."""

    def __init__(self, n_filters=4, k=3, n_classes=10, img=8, seed=SEED):
        rng = np.random.default_rng(seed)
        self.k = k
        out = img - k + 1
        flat = out * out * n_filters
        # He initialisation (suited to ReLU)
        self.W = rng.normal(0, np.sqrt(2 / (k * k)), (n_filters, k, k))
        self.b = np.zeros(n_filters)
        self.Wd = rng.normal(0, np.sqrt(2 / flat), (flat, n_classes))
        self.bd = np.zeros(n_classes)

    @property
    def params(self):
        return [self.W, self.b, self.Wd, self.bd]

    def forward(self, X):
        windows = sliding_window_view(X, (self.k, self.k), axis=(1, 2))  # (N,o,o,k,k)
        conv = np.einsum("nijab,fab->nijf", windows, self.W) + self.b     # feature maps
        act = np.maximum(conv, 0)                                          # ReLU
        flat = act.reshape(len(X), -1)
        logits = flat @ self.Wd + self.bd
        return logits, (windows, conv, flat)

    def loss(self, X, Y):
        logits, _ = self.forward(X)
        p = softmax(logits)
        return -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))

    def gradients(self, X, Y):
        """Backpropagation by hand (chain rule, layer by layer)."""
        N = len(X)
        logits, (windows, conv, flat) = self.forward(X)
        p = softmax(logits)
        dlogits = (p - Y) / N                      # softmax + cross-entropy
        dWd = flat.T @ dlogits
        dbd = dlogits.sum(axis=0)
        dflat = dlogits @ self.Wd.T
        dact = dflat.reshape(conv.shape)
        dconv = dact * (conv > 0)                  # ReLU passes gradient only where input > 0
        dW = np.einsum("nijab,nijf->fab", windows, dconv)   # each filter is shared across positions
        db = dconv.sum(axis=(0, 1, 2))
        return [dW, db, dWd, dbd]

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)


def gradient_check(model: TinyCNN, X, Y, n_checks=5, eps=1e-5, seed=SEED):
    """Compare backprop gradients with finite differences on random entries."""
    rng = np.random.default_rng(seed)
    # ReLU is not differentiable at exactly 0, and many all-zero image windows
    # give a pre-activation of exactly `bias`. With bias = 0 the finite-difference
    # nudge would straddle that kink and report a false mismatch. So for the check
    # only, use small random biases and restore the originals afterwards.
    saved = (model.b.copy(), model.bd.copy())
    model.b[:] = rng.normal(0, 0.1, model.b.shape)
    model.bd[:] = rng.normal(0, 0.1, model.bd.shape)
    analytic = model.gradients(X, Y)
    worst = 0.0
    for param, grad in zip(model.params, analytic):
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in param.shape)
            original = param[idx]
            param[idx] = original + eps
            plus = model.loss(X, Y)
            param[idx] = original - eps
            minus = model.loss(X, Y)
            param[idx] = original
            numeric = (plus - minus) / (2 * eps)
            denom = max(1e-8, abs(numeric) + abs(grad[idx]))
            worst = max(worst, abs(numeric - grad[idx]) / denom)
    model.b[:], model.bd[:] = saved
    return worst


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def main():
    digits = load_digits()
    images = digits.images / 16.0          # 1797 images of 8x8, pixel values scaled to 0..1
    labels = digits.target

    # ---- 1) what does a convolution do?
    img = images[0]                        # a handwritten "0"
    print("=== 1) Convolution by hand ===")
    print("Image (8x8, a handwritten", labels[0], "):\n", img, "\n")
    v_map = conv2d_loop(img, VERTICAL_EDGE)
    h_map = conv2d_loop(img, HORIZONTAL_EDGE)
    assert np.allclose(v_map, conv2d_fast(img, VERTICAL_EDGE))
    print("Loop and vectorised convolution agree. Output is 6x6 (8 - 3 + 1).")
    print("Vertical-edge feature map:\n", v_map, "\n")
    print("Horizontal-edge feature map:\n", h_map)
    print("Large +/- values mark where the filter's pattern appears in the image.\n")

    # ---- 2) parameter counts
    print("=== 2) Why convolution? Parameter sharing ===")
    n_filters, k = 4, 3
    conv_params = n_filters * k * k + n_filters
    dense_same_output = (8 * 8) * (6 * 6 * n_filters) + (6 * 6 * n_filters)
    print(f"Conv layer ({n_filters} filters of {k}x{k}): {conv_params} parameters")
    print(f"Dense layer producing the same-size output (64 -> 144): {dense_same_output} parameters")
    print("The conv layer reuses the same few weights at every position.\n")

    # ---- 3) train a tiny CNN
    print("=== 3) Train a tiny CNN with hand-written backprop ===")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=SEED, stratify=labels)
    Y_train = np.eye(10)[y_train]

    model = TinyCNN()
    err = gradient_check(model, X_train[:8], Y_train[:8])
    print(f"Gradient check (backprop vs finite differences), worst relative error: {err:.2e}")
    assert err < 1e-4, "Backprop gradients look wrong!"

    opt = Adam(model.params, lr=0.01)
    rng = np.random.default_rng(SEED)
    n_epochs, batch = 30, 64
    for epoch in range(1, n_epochs + 1):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            opt.step(model.gradients(X_train[idx], Y_train[idx]))
        if epoch % 5 == 0 or epoch == 1:
            acc = (model.predict(X_test) == y_test).mean()
            print(f"epoch {epoch:>2}  train loss {model.loss(X_train, Y_train):.3f}  test accuracy {acc:.1%}")

    cnn_acc = (model.predict(X_test) == y_test).mean()
    total = sum(p.size for p in model.params)
    print(f"\nTiny CNN: {total} parameters, test accuracy {cnn_acc:.1%}\n")

    # ---- 4) baseline
    print("=== 4) Baseline: logistic regression on raw pixels ===")
    base = LogisticRegression(max_iter=2000).fit(X_train.reshape(len(X_train), -1), y_train)
    base_acc = base.score(X_test.reshape(len(X_test), -1), y_test)
    base_params = base.coef_.size + base.intercept_.size
    print(f"Logistic regression: {base_params} parameters, test accuracy {base_acc:.1%}")
    gap = cnn_acc - base_acc
    print(f"\nDifference: {gap:+.1%} on {len(y_test)} test images (about {abs(gap) * len(y_test):.0f} images).")
    print("That is small for a single split, so don't read much into it: on tiny 8x8 digits a")
    print("simple model is already strong. CNNs matter more on larger, messier images.")
    print("Try other seeds or cross-validation to see how stable the gap is.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- optional PyTorch comparison
# to run
# pip install torch
# python -c "from main import run_pytorch_cnn; run_pytorch_cnn()"
def run_pytorch_cnn(
    n_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.01,
):
    """
    Train the same basic CNN using PyTorch instead of hand-written NumPy
    backpropagation. Run separately with:
        python -c "from main import run_pytorch_cnn; run_pytorch_cnn()"
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)

    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # PyTorch expects (batch, channels, height, width).
    X_train = torch.tensor(X_train[:, None, :, :], dtype=torch.float32)
    X_test = torch.tensor(X_test[:, None, :, :], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    # Same architecture as TinyCNN: Conv -> ReLU -> Flatten -> Dense.
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 6 * 6, 10),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    parameter_count = sum(p.numel() for p in model.parameters())
    print("\n=== PyTorch CNN ===")
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()       # automatic backpropagation
            optimizer.step()
            total_loss += loss.item() * len(batch_images)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                predictions = model(X_test).argmax(dim=1)
                accuracy = (predictions == y_test).float().mean().item()

            print(
                f"epoch {epoch:>2}  "
                f"train loss {total_loss / len(X_train):.3f}  "
                f"test accuracy {accuracy:.1%}"
            )

    model.eval()
    with torch.no_grad():
        cnn_accuracy = (
            model(X_test).argmax(dim=1) == y_test
        ).float().mean().item()

    baseline = LogisticRegression(max_iter=2000)
    baseline.fit(X_train.numpy().reshape(len(X_train), -1), y_train.numpy())
    baseline_accuracy = baseline.score(
        X_test.numpy().reshape(len(X_test), -1),
        y_test.numpy(),
    )

    print(f"\nPyTorch CNN: {parameter_count:,} parameters, test accuracy {cnn_accuracy:.1%}")
    print(f"Logistic regression: test accuracy {baseline_accuracy:.1%}")
    print(f"Accuracy difference: {cnn_accuracy - baseline_accuracy:+.1%}")
    print("\nPyTorch's loss.backward() replaces the manual gradient calculations.")
    print("The optimizer handles the parameter updates.")

