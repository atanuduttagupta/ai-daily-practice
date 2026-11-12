# P0 - Day 9 — Track I: CV & Sequence Models (Subtopic 1: CNN basics)

## What this project does

Shows what a convolutional neural network (CNN) really computes, using only
NumPy, so there is no deep-learning framework to install. It has four parts:

1. Implements the 2D convolution operation by hand and applies two edge
   filters to a handwritten digit.
2. Compares the number of parameters in a convolution layer and a fully
   connected layer.
3. Trains a tiny CNN (conv → ReLU → flatten → dense → softmax) with
   hand-written backpropagation on the 8x8 digit images that ship with
   scikit-learn, and proves the gradients are correct.
4. Compares the CNN with a logistic regression baseline.

**Input:** none to provide. The data (1,797 digit images, 8x8 pixels) loads
from scikit-learn with no download.
**Output:** printed pixel arrays and feature maps, parameter counts, a
gradient check result, training progress, and a CNN vs baseline comparison.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

It runs in a few seconds on a normal laptop CPU.

## Theory

### Why not just use a normal (fully connected) network on images?
A fully connected layer connects every pixel to every neuron. For a 224x224
colour image that is about 150,000 inputs, so even one layer with 1,000
neurons needs around 150 million weights. It also ignores the fact that
nearby pixels are related, and it has to relearn the same pattern (say, an
edge) separately for every position in the image.

### The convolution idea
A **filter** (or **kernel**) is a small grid of weights, such as 3x3. Slide it
across the image; at each position, multiply the filter by the patch of
pixels underneath, and add everything up. That gives one number per
position, and the grid of those numbers is a **feature map**. A large value
means "the pattern this filter looks for is here".

Two properties make this powerful:
- **Local receptive fields:** each output looks only at a small patch, which
  matches how image structure works (edges, corners, textures are local).
- **Parameter sharing:** the same few weights are reused at every position.
  In this project 4 filters need 40 parameters, while a dense layer producing
  the same-size output would need 9,360.

Because of sharing, a pattern is detected wherever it appears (this property
is called translation equivariance: shift the input and the feature map
shifts with it).

Note: deep-learning "convolution" does not flip the kernel, so it is
technically cross-correlation. Every framework does it this way, and since
the filters are learned it makes no practical difference.

### Useful formulas
- **Output size:** `floor((W - K + 2P) / S) + 1`, where W is input size, K is
  kernel size, P is padding and S is stride. Here: `(8 - 3 + 0) / 1 + 1 = 6`.
- **Parameters in a conv layer:** `(K × K × C_in + 1) × C_out`. Here:
  `(3 × 3 × 1 + 1) × 4 = 40`. (`C_in` is input channels, 1 for grayscale and
  3 for RGB; `C_out` is the number of filters.)
- **Stride** is how many pixels the filter moves each step. **Padding** adds a
  border of zeros so the output doesn't shrink. This project uses "valid"
  convolution (no padding, stride 1). Padding and pooling are the next topic
  in this track.

### Where the filters come from
Early computer vision hand-designed filters, like the Sobel edge detectors in
part 1. A CNN **learns** its filters: they start random, and backpropagation
adjusts them to reduce the loss. Early layers tend to learn simple things
like edges, and deeper layers combine them into shapes and objects.

### The pipeline in this project
`image (8x8)` → `conv: 4 filters 3x3` → `4 feature maps (6x6)` → `ReLU`
(keep positives, zero out negatives, adds non-linearity) → `flatten` (144
numbers) → `dense layer` → `softmax` → probability for each digit 0–9.

Real CNNs repeat conv → ReLU → pooling several times before the dense layers.

### The same network in PyTorch (for reference, not run here)
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),  # (N,1,8,8) -> (N,4,6,6)
    nn.ReLU(),
    nn.Flatten(),                                             # -> (N,144)
    nn.Linear(144, 10),
)
# train with nn.CrossEntropyLoss() and torch.optim.Adam(model.parameters())
```
PyTorch computes the backward pass automatically (autograd); here we wrote it
by hand to see how it works.

## What the output shows (one run, seed 42)

- The two Sobel filters give large positive/negative values at vertical and
  horizontal edges of the "0".
- **Gradient check:** backprop matched finite-difference gradients to about
  1e-9, so the hand-written backward pass is correct.
- **Tiny CNN:** 1,490 parameters, 97.2% test accuracy.
- **Logistic regression baseline:** 650 parameters, 95.6% test accuracy.

The gap is about 1.7 points, roughly 6 of 360 test images, on a single
train/test split. That is too small to draw conclusions from: on tiny 8x8
digits a simple model is already strong, and CNNs earn their keep on larger,
messier images. Try other seeds or cross-validation to see how stable it is.

## A debugging lesson from building this

My first gradient check failed (relative error about 0.14), but only for the
bias of the conv layer. The cause was the test, not the backprop: many image
windows are all zeros, so their value before ReLU is exactly the bias, which
was 0, right on ReLU's non-differentiable point. The finite-difference nudge
straddled that point and gave a meaningless answer. The check now uses small
random biases, and the error is about 1e-9. Moral: gradient checks can
mislead at kinks, so understand a mismatch before "fixing" the code.

## Try these

1. Change `n_filters` to 1, 8, 16. How do accuracy and parameter count change?
2. Replace the Sobel kernels with your own (a blur: all ones divided by 9).
3. Print the learned filters (`model.W`) after training. Do they look like edges?
4. Add a second filter size, or stride 2, and work out the output size by hand.
5. Run with several seeds and report mean ± std for CNN vs logistic regression.

## One thing learned

A CNN is a fully connected network with two constraints, local connections and
shared weights, which cut the parameter count dramatically and build in the
assumption that the same pattern can appear anywhere in the image.

## Next in this track (Day 19, Track I subtopic 2)

Pooling and padding.
