# MNIST Demo

Adam can train a real neural network from scratch. The `examples/mnist.adam` program implements a 2-layer neural network that classifies handwritten digits, achieving **97.2% test accuracy** on the MNIST dataset.

## Results

```
Epoch 1 — avg loss: 0.2927
Epoch 2 — avg loss: 0.1410
Epoch 3 — avg loss: 0.1008
Epoch 4 — avg loss: 0.0792
Epoch 5 — avg loss: 0.0652
Test accuracy: 9704 / 9984 = 97.2%
```

Training completes in ~2 minutes on a single CPU core using Adam's naive C virtual machine — no BLAS, no GPU, no external libraries.

## Running It

```bash
just prepare-mnist   # download MNIST via scikit-learn (~15 seconds)
just mnist           # compile and train (~2 minutes)
```

The `prepare-mnist` step downloads the 70,000 MNIST images via scikit-learn and saves them as binary tensor files in `data/`. The binary format is simple: `[ndim:i32][shape[0]:i32]...[shape[n]:i32][data:f64*]`.

## Network Architecture

| Layer | Shape | Activation |
|-------|-------|------------|
| Input | [batch, 784] | — |
| Hidden | [batch, 128] | ReLU |
| Output | [batch, 10] | Softmax |

- **Loss**: Cross-entropy
- **Optimizer**: Mini-batch SGD (batch size 32, learning rate 0.1)
- **Initialization**: Xavier (sqrt(2 / (fan_in + fan_out)))
- **Epochs**: 5 over 60,000 training images

## How It Works

The entire program is written in Adam — no Python, no C extensions. It uses Adam's built-in tensor operations for all computation:

### Forward Pass

```adam
let z1 = x @@ w1 + b1          // linear layer 1: [32,784] @@ [784,128] + [1,128]
let h = tensor_relu(z1)         // ReLU activation
let logits = h @@ w2 + b2       // linear layer 2: [32,128] @@ [128,10] + [1,10]

// Numerically stable softmax
let max_l = tensor_max(logits)
let shifted = logits - max_l
let exps = tensor_exp(shifted)
let sum_exps = tensor_sum_axis(exps, 1)   // sum per row -> [32,1]
let probs = exps / sum_exps               // broadcast [32,10] / [32,1]
```

### Backward Pass (Manual)

```adam
let d_logits = (probs - targets) / 32.0     // softmax + cross-entropy gradient
let d_w2 = tensor_transpose(h) @@ d_logits  // weight gradient
let d_b2 = tensor_sum_axis(d_logits, 0)     // bias gradient
let d_h = d_logits @@ tensor_transpose(w2)  // backprop through weights
let d_z1 = tensor_relu_backward(z1, d_h)    // ReLU backward
let d_w1 = tensor_transpose(x) @@ d_z1
let d_b1 = tensor_sum_axis(d_z1, 0)

// SGD update
w1 = w1 - lr * d_w1
b1 = b1 - lr * d_b1
```

### Evaluation

The test loop processes 10,000 test images in batches, computing argmax of the logits to find the predicted class, then comparing against the true label.

## Native Functions Added

These tensor operations were implemented as C native functions in `vm/src/native.c`:

| Function | Purpose |
|----------|---------|
| `tensor_exp(t)` | Element-wise exp (for softmax) |
| `tensor_log(t)` | Element-wise log (for cross-entropy) |
| `tensor_relu(t)` | Element-wise max(0, x) |
| `tensor_relu_backward(z, grad)` | ReLU gradient: grad where z > 0, else 0 |
| `tensor_max(t)` | Maximum element (scalar) |
| `tensor_sum_axis(t, axis)` | Sum along axis (for row/column reduction) |
| `tensor_slice(t, start, count)` | Extract rows from a tensor |
| `tensor_one_hot(labels, n)` | Convert labels to one-hot encoding |
| `tensor_load(path)` | Load tensor from binary file |

## VM Enhancements

### Tensor-Scalar Broadcast

Arithmetic operators (`+`, `-`, `*`, `/`) now support mixed tensor-scalar operands:

```adam
let w = tensor_randn([784, 128]) * 0.047  // scale each element
let shifted = logits - max_val             // subtract scalar from tensor
```

### 2D Tensor Broadcasting

The VM supports NumPy-style broadcasting for 2D tensors:

- **Row broadcast**: `[M,N] op [1,N]` — repeats the single row across M rows
- **Column broadcast**: `[M,N] op [M,1]` — repeats the single column across N columns

This is critical for operations like `exps / sum_exps` where `exps` is `[32,10]` and `sum_exps` is `[32,1]`.

### GC Safety for Large Tensors

Loading MNIST requires allocating ~440MB of tensor data. Two GC bugs were fixed:

1. **Uninitialized fields**: `adam_new_tensor` now zeroes `shape`, `data`, and `count` before any sub-allocation, preventing the GC sweep from freeing garbage pointers.

2. **Missing GC root**: The tensor is pushed onto the VM stack during `adam_new_tensor` so the garbage collector sees it as a live root during subsequent allocations that may trigger collection.

## Why Not Autograd?

Adam has a compile-time autograd system (`grad()` intrinsic), but the MNIST demo uses manual backpropagation because:

- Autograd only differentiates w.r.t. the first parameter (MNIST has 4 weight matrices)
- Autograd doesn't yet handle ReLU, exp, or log
- Manual backprop demonstrates the full tensor operation set

The autograd system works well for simpler cases (see `examples/neural_net.adam`).

## Files

| File | Purpose |
|------|---------|
| `examples/mnist.adam` | Full MNIST training program |
| `examples/mnist_smoke.adam` | Minimal forward pass test (used in E2E tests) |
| `stdlib/src/adam_tools/prepare_mnist.py` | Downloads and converts MNIST data |
| `vm/src/native.c` | Native tensor functions |
| `vm/src/vm.c` | Tensor-scalar and 2D broadcasting |
| `vm/src/object.c` | GC-safe tensor allocation |
