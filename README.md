# Neural Networks from Scratch: micrograd to a NumPy CNN

A learning project that builds deep learning up from first principles, following Andrej Karpathy's micrograd lecture, then scales it to a convolutional neural network (CNN) written in pure NumPy and validated against an equivalent PyTorch model on MNIST.

## Results

| Model | Test accuracy (MNIST, 10,000 images) |
|---|---|
| CNN in pure NumPy (hand-written forward and backward passes) | **97.57%** |
| Equivalent PyTorch CNN (same architecture, split and hyperparameters) | **98.00%** |

Validation accuracy per epoch (5,000 held-out images):

| Epoch | NumPy loss | PyTorch loss | NumPy val acc | PyTorch val acc |
|---|---|---|---|---|
| 1 | 0.3626 | 0.2316 | 0.9398 | 0.9616 |
| 2 | 0.1054 | 0.0997 | 0.9590 | 0.9648 |
| 3 | 0.0831 | 0.0787 | 0.9718 | 0.9682 |
| 4 | 0.0700 | 0.0687 | 0.9746 | 0.9736 |
| 5 | 0.0616 | 0.0598 | 0.9768 | 0.9662 |
| 6 | 0.0554 | 0.0557 | 0.9724 | 0.9708 |
| 7 | 0.0517 | 0.0513 | 0.9670 | 0.9802 |
| 8 | 0.0478 | 0.0472 | 0.9780 | 0.9798 |

The training losses of the two implementations track each other closely from epoch 3 onward, which is the main evidence that the hand-written backpropagation is correct.

## What is in the notebook (`neuralnet_to_cnn.ipynb`)

**Part 1: micrograd (original, unchanged).** A scalar autograd engine (`Value`) with `+`, `*`, `**`, `tanh`, `exp` and reverse-mode backpropagation, a small MLP (`Neuron`, `Layer`, `MLP`), numerical-vs-analytical gradient checks, comparison with PyTorch autograd, and a tiny training loop.

**Part 2: CNN in NumPy.** The same ideas (forward pass, local gradients, chain rule, gradient descent) scaled from scalars to arrays.

- Data: MNIST downloaded automatically, split into 55,000 train, 5,000 validation and 10,000 test images, normalised with mean 0.1307 and std 0.3081.
- Layers, each with `forward` and `backward`: `Conv` (3x3, 1 to 8 channels), `ReLU`, `MaxPool` (2x2), `Dense`.
- Architecture: Conv(1, 8, 3x3) -> ReLU -> MaxPool(2) -> Dense(1352, 10). About 13.6k parameters.
- Loss: softmax cross-entropy, with the combined gradient `(probs - onehot) / N`.
- Optimiser: plain mini-batch gradient descent, the same update rule as in micrograd (`p -= lr * grad`).
- Hyperparameters: learning rate 0.02, batch size 64, 8 epochs.
- Evaluation: per-epoch validation, a single final test run, per-digit accuracy, a confusion matrix (pandas) and a view of misclassified digits.
- Optional numerical gradient check on the conv and dense weights.

**Part 3: PyTorch comparison.** The identical architecture and settings in `torch.nn`, trained on the same splits, with a side-by-side table and plot. An optional cell copies the trained NumPy weights into the PyTorch model and compares the outputs on 100 test images, as a direct check that the forward passes match.

## What went wrong along the way

1. **First runs were stuck at about 86 to 90% accuracy.** Validation accuracy swung by up to 10 points between epochs. The causes were a learning rate that was too high, only 10k training images for 5 epochs, and raw 0 to 1 pixel values. Fix: normalise the inputs, train on the full 55k images, lower the learning rate to 0.02 and train for 8 epochs. Accuracy rose to about 97.5%.
2. **PyTorch loss of 1712 in epoch 1.** The PyTorch optimiser had `lr=0.05` hardcoded while the NumPy run had moved to `LR = 0.02`, so the two were not comparable. One batch diverged, then training recovered but stayed noisy and worse. Fix: `lr=LR`, so both models read the same setting. Lesson: when comparing implementations, share one set of hyperparameters rather than duplicating values.
3. **Early version was too complex.** The first NumPy CNN used `im2col`, dropout, Adam and two conv layers. It worked but was hard to follow, so it was replaced with a one-conv-layer network and a loop over kernel positions that is easy to read.
4. **Testing caveat.** The code was developed in a sandbox without internet, so it was checked on scikit-learn's small digits dataset (gradient check passed, about 94% accuracy) and the real MNIST runs were done locally.

## Why the NumPy version is slower than PyTorch

These are explanations from how the two implementations work, not measurements. To confirm them, time each epoch for both models (the PyTorch cell already records `torch_time`) and profile the NumPy loop.

- **Optimised kernels.** PyTorch dispatches convolution and matrix multiplication to compiled C++ libraries (such as oneDNN or MKL on CPU) that are tuned for cache use, vectorised instructions and multiple threads. The NumPy version expresses convolution as a Python loop over the 9 kernel positions, each calling `einsum`, so the arithmetic is done in many smaller, less optimised steps.
- **`einsum` without optimisation.** By default `np.einsum` can evaluate these contractions with straightforward loops instead of routing them to BLAS matrix multiplication, which is much faster. This is likely a major cost in both the forward and backward passes.
- **Data type.** The NumPy path runs in float64, while PyTorch uses float32. That doubles memory traffic and halves how many values fit in each vector instruction.
- **Temporary arrays.** The NumPy layers allocate new arrays at nearly every step (the padded outputs, the pooling mask, the gradient buffers), which costs time and memory bandwidth. PyTorch reuses buffers and fuses operations where it can.
- **Python overhead.** Each layer call, loop iteration and batch goes through the Python interpreter. With a batch size of 64 and a small network, this overhead is a meaningful share of the total.
- **Hand-written backward pass.** PyTorch's autograd runs optimised backward kernels. The NumPy backward pass repeats a loop of `einsum` calls per kernel position.

What this suggests trying next, if speed matters: convert the data and weights to float32, use `im2col` plus a single matrix multiplication for the convolution, pass `optimize=True` to `einsum`, or use `np.matmul`/`tensordot` instead.

The slowdown does not affect correctness. Both models reach nearly the same loss curves and accuracy, which shows the hand-written implementation does the same computation, only less efficiently.

## Reading the results honestly

- The 0.43 point gap in test accuracy (97.57% vs 98.00%) comes from a single run per model with different random initialisations. Validation accuracy moves by several tenths of a point from epoch to epoch, so this gap is not strong evidence that PyTorch is better. A fair comparison would repeat each run with several seeds and report the mean and spread.
- The network is deliberately small. A deeper CNN with two conv layers, dropout and a better optimiser would likely reach about 99%.

## Roadmap

1. **Done:** NumPy CNN on MNIST with train, validation and test splits, validated against PyTorch.
2. **Next:** Bangla digit recognition (retrain the same CNN on a Bangla digit dataset), then Bangla and English handwriting.
3. **Later:** live handwriting-to-text from a graphics tablet, feeding the trained model from a drawing canvas.

## Run it

```
pip install numpy pandas matplotlib torch graphviz
```

Open `neuralnet_to_cnn.ipynb` and run the cells in order. MNIST downloads once into `mnist_data/`. Set `N_TRAIN = 55000` and `EPOCHS = 8` in the training cell for the full run reported above.

## CV summary

> Implemented a convolutional neural network from scratch in NumPy (convolution, max-pooling, softmax cross-entropy, hand-derived backpropagation), verified with numerical gradient checking and against an equivalent PyTorch model; 97.6% test accuracy on MNIST (PyTorch reference: 98.0%).

## Credits

Part 1 follows Andrej Karpathy's micrograd lecture (neural networks: zero to hero).
