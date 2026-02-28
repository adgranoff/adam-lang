"""Prepare names dataset for transformer training in Adam.

Reads names.txt, encodes each name as a sequence of character indices,
pads to fixed length, and saves as binary tensors.

Vocabulary: '.' (padding/end), 'a'-'z', "'" (apostrophe)
  Index 0 = '.' (padding / start / end token)
  Index 1-26 = 'a'-'z'
  Index 27 = "'"

Binary format per file: [ndim:i32][shape[0]:i32]...[shape[n]:i32][data:f64*]

Usage:
    uv run python -m adam_tools.prepare_names
"""

from __future__ import annotations

import random
import struct
from pathlib import Path


VOCAB = "." + "abcdefghijklmnopqrstuvwxyz" + "'"
VOCAB_SIZE = len(VOCAB)  # 28
MAX_LEN = 20  # Max name length (including start/end tokens)
# Format: [START] + name_chars + [END] + padding
# START and END are both '.' (index 0)


def char_to_idx(c: str) -> int:
    """Convert character to vocabulary index."""
    idx = VOCAB.find(c.lower())
    if idx == -1:
        return 0  # Unknown chars → padding
    return idx


def save_tensor(path: Path, data: list[float], shape: list[int]) -> None:
    """Save a tensor in Adam's binary format."""
    ndim = len(shape)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", ndim))
        for dim in shape:
            f.write(struct.pack("<i", dim))
        f.write(struct.pack(f"<{len(data)}d", *data))
    print(f"  Saved {path} — shape {shape}, {len(data)} elements")


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    names_file = data_dir / "names.txt"
    if not names_file.exists():
        print(f"Error: {names_file} not found.")
        print("Download it: curl -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt")
        raise SystemExit(1)

    # Read and filter names
    names = []
    for line in names_file.read_text().strip().splitlines():
        name = line.strip().lower()
        if not name:
            continue
        # Filter: only keep names that fit in MAX_LEN - 2 (room for start + end)
        if len(name) <= MAX_LEN - 2:
            # Check all chars are in vocab
            if all(c in VOCAB for c in name):
                names.append(name)

    print(f"Loaded {len(names)} names (max length {MAX_LEN - 2} chars)")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Sample names: {names[:5]}")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(names)

    # Encode each name as: [START(0)] + chars + [END(0)] + padding(0)
    # Input sequence: positions 0..MAX_LEN-2 (what the model sees)
    # Target sequence: positions 1..MAX_LEN-1 (what the model predicts)
    inputs = []   # [N, MAX_LEN-1] — input tokens
    targets = []  # [N, MAX_LEN-1] — target tokens (shifted by 1)

    seq_len = MAX_LEN - 1  # 19

    for name in names:
        # Full sequence: [0] + [char indices] + [0] + [0 padding...]
        full_seq = [0]  # START
        for c in name:
            full_seq.append(char_to_idx(c))
        full_seq.append(0)  # END
        while len(full_seq) < MAX_LEN:
            full_seq.append(0)  # PAD

        # Input: positions 0..seq_len-1
        # Target: positions 1..seq_len
        inputs.extend(float(x) for x in full_seq[:seq_len])
        targets.extend(float(x) for x in full_seq[1:seq_len + 1])

    n = len(names)

    # Split: 90% train, 10% test
    train_n = int(n * 0.9)
    test_n = n - train_n

    train_inputs = inputs[:train_n * seq_len]
    train_targets = targets[:train_n * seq_len]
    test_inputs = inputs[train_n * seq_len:]
    test_targets = targets[train_n * seq_len:]

    print(f"Train: {train_n} names, Test: {test_n} names")
    print(f"Sequence length: {seq_len}")

    print("\nSaving tensors...")
    save_tensor(data_dir / "names_train_inputs.bin", train_inputs, [train_n, seq_len])
    save_tensor(data_dir / "names_train_targets.bin", train_targets, [train_n, seq_len])
    save_tensor(data_dir / "names_test_inputs.bin", test_inputs, [test_n, seq_len])
    save_tensor(data_dir / "names_test_targets.bin", test_targets, [test_n, seq_len])

    # Save vocab info as a simple text file for reference
    vocab_file = data_dir / "names_vocab.txt"
    with open(vocab_file, "w") as f:
        f.write(f"vocab_size={VOCAB_SIZE}\n")
        f.write(f"seq_len={seq_len}\n")
        f.write(f"max_name_len={MAX_LEN - 2}\n")
        for i, c in enumerate(VOCAB):
            f.write(f"{i}={repr(c)}\n")
    print(f"  Saved {vocab_file}")

    print("\nDone! Run transformer with: just transformer")


if __name__ == "__main__":
    main()
