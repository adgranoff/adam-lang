# Transformer Demo

Adam can train a character-level transformer language model from scratch. The `examples/transformer.adam` program implements a single-layer, single-head transformer that learns to generate plausible-sounding names after training on 32,000 real names.

## Results

```
Loading data...
Train: 28829 | d_model: 64
Initializing weights...
Training...
Epoch 0 — Loss: 1.28412
Epoch 1 — Loss: 1.06402
Epoch 2 — Loss: 1.00706
Time: 328s

Generated names:
  eic
  genn
  ndle
  symeeyi
  mabxo
  ibpan
  tsgamnnv
  faaicno
  ikcyrni
  eaowya
```

Training completes in ~5 minutes on a single CPU core using Adam's naive C virtual machine — no BLAS, no GPU, no external libraries.

## Running It

```bash
just prepare-names   # download names dataset (~1 second)
just transformer     # compile and train (~5 minutes)
```

The `prepare-names` step downloads 32,033 names from a public dataset, encodes them as character sequences (vocabulary: `.` + `a-z` + `'` = 28 tokens, max sequence length 19), and saves them as binary tensor files in `data/`.

## Architecture

| Component | Shape | Description |
|-----------|-------|-------------|
| Token embedding | [28, 64] | Character → vector |
| Positional embedding | [19, 64] | Position → vector |
| Q/K/V projections | [64, 64] each | Self-attention projections |
| Output projection | [64, 64] | Attention output |
| FFN layer 1 | [64, 256] | Feedforward expansion |
| FFN layer 2 | [256, 64] | Feedforward compression |
| Output head | [64, 28] | Vector → character logits |

- **Attention**: Single-head scaled dot-product with causal mask
- **Activation**: ReLU in feedforward block
- **Residual connections**: After attention and feedforward blocks
- **Loss**: Cross-entropy with numerically stable softmax
- **Optimizer**: Mini-batch SGD (batch size 64, learning rate 0.01)
- **Initialization**: Scaled random (0.1 for embeddings, 0.125 for projections)

## How It Works

The entire program is written in Adam — no Python, no C extensions. It uses Adam's built-in tensor operations for all computation.

### Numerically Stable Softmax

```adam
fn softmax(x) {
    let mx = tensor_max_axis(x, -1)    // max per row for stability
    let e = tensor_exp(x - mx)         // shift before exp
    e / tensor_sum_axis(e, -1)         // normalize
}
```

### Forward Pass

The forward pass computes self-attention with a causal mask, followed by a feedforward network with residual connections:

```adam
fn forward(x_batch, tok_emb, pos_emb, wq, wk, wv, wo, w1, b1, w2, b2, w_out, b_out, mask) {
    let x_emb = tensor_embedding_lookup(tok_emb, x_batch) + pos_emb

    // Self-attention: Q, K, V projections → scaled dot-product → causal mask
    let q = x_emb @@ wq
    let k = x_emb @@ wk
    let v = x_emb @@ wv
    let scores = (q @@ tensor_permute(k, [0, 2, 1])) * scale + mask
    let attn = softmax(scores)
    let h = x_emb + (attn @@ v) @@ wo          // residual connection

    // Feedforward: expand → ReLU → compress → residual
    let h2 = h + tensor_relu(h @@ w1 + b1) @@ w2 + b2

    // Output projection → softmax
    let logits = h2 @@ w_out + b_out
    let probs = softmax(logits)
    ...
}
```

### Backward Pass (Manual)

All gradients are computed manually through the full transformer — attention, softmax, feedforward, and embeddings:

```adam
// Output gradient: softmax + cross-entropy combined
let d_logits = (probs - y_oh) * inv_n

// Attention backward: compute gradients for Q, K, V, and output projection
let d_scores = attn * (d_attn - tensor_sum_axis(attn * d_attn, -1)) * scale
let d_q = d_scores @@ k
let d_k = tensor_permute(tensor_permute(q, [0, 2, 1]) @@ d_scores, [0, 2, 1])

// Embedding backward: scatter gradients back to token embedding table
let dtok = tensor_scatter_add(tensor_zeros([vocab_size, d_model]), x_flat, d_flat)
```

### Text Generation

After training, the model generates new names by sampling from the learned distribution:

```adam
fn generate_name(...) {
    let seq = tensor_zeros([1, seq_len])
    while pos < seq_len - 1 {
        // Run transformer forward on current sequence
        let g_probs = softmax(g_h2 @@ w_out + b_out)
        // Sample next character from probability distribution
        let pred = tensor_sample(g_probs, -1)
        let next_token = to_int(tensor_get(pred, pos - 1))
        ...
    }
}
```

## Native Functions Added

These tensor operations were added to `vm/src/native.c` for the transformer:

| Function | Purpose |
|----------|---------|
| `tensor_permute(t, axes)` | Reorder tensor dimensions (e.g., transpose [0,2,1]) |
| `tensor_max_axis(t, axis)` | Max along axis with keepdim (for stable softmax) |
| `tensor_embedding_lookup(table, indices)` | Look up rows from embedding table |
| `tensor_causal_mask(size)` | Create [1, size, size] lower-triangular mask |
| `tensor_argmax(t, axis)` | Index of max value along axis |
| `tensor_sample(probs, axis)` | Sample index from probability distribution |
| `tensor_scatter_add(target, indices, source)` | Reverse of embedding lookup (for gradients) |
| `tensor_set(t, idx, val)` | Return copy of tensor with one element changed |
| `tensor_get(t, idx)` | Get single element by flat index |
| `tensor_sqrt(t)` | Element-wise square root |
| `tensor_tanh(t)` | Element-wise hyperbolic tangent |
| `chr(code)` | Convert integer to single-character string |

## VM Enhancements

### N-Dimensional Broadcasting

Broadcasting was generalized from 2D to N-dimensional, following NumPy's right-aligned rules. This enables 3D tensor operations like `[B, T, T] * [B, T, 1]` (attention score masking) and `[B, T, D] + [1, T, D]` (positional embedding broadcast).

### Batched Matrix Multiply

The `@@` operator now correctly handles batched operands: `[B, M, K] @@ [B, K, N] → [B, M, N]`, where each batch element is multiplied independently.

### N-Dimensional `tensor_sum_axis`

The `tensor_sum_axis` function was generalized from 2D to N-dimensional with keepdim semantics and negative axis support. This is essential for the softmax numerator/denominator computations on 3D tensors.

### Constant Pool Deduplication

The compiler now deduplicates Int, Float, and String constants in each function's constant pool. This was necessary because the transformer's top-level code references many global variables (12 weight tensors, each multiple times), which previously exceeded the 256-entry limit.

## Comparison: MNIST vs Transformer

| Aspect | MNIST | Transformer |
|--------|-------|-------------|
| Model | 2-layer MLP | 1-layer transformer |
| Task | Image classification | Name generation |
| Tensor rank | 2D (matrices) | 3D (batched sequences) |
| New ops needed | 9 | 12 additional |
| Parameters | ~100K | ~130K |
| Training time | ~2 min | ~5 min |

The transformer demo exercises the full N-dimensional tensor system (3D matmul, broadcasting, permute, attention masking) that was built on top of the 2D foundations established by MNIST.

## Files

| File | Purpose |
|------|---------|
| `examples/transformer.adam` | Full transformer training program |
| `stdlib/src/adam_tools/prepare_names.py` | Downloads and converts names data |
| `vm/src/native.c` | Native tensor functions (permute, embedding, mask, etc.) |
| `vm/src/vm.c` | N-dim broadcasting and batched matmul |
| `compiler/src/types.rs` | Type registrations for new natives |
| `compiler/src/compiler.rs` | Constant pool deduplication |
