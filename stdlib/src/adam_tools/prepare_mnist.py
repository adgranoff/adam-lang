"""Download and prepare MNIST data for Adam's tensor_load native.

Binary format per file: [ndim:i32][shape[0]:i32]...[shape[n]:i32][data:f64*]

Usage:
    uv run --with scikit-learn python -m adam_tools.prepare_mnist
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


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
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        print("Error: scikit-learn required. Run:")
        print("  uv run --with scikit-learn python -m adam_tools.prepare_mnist")
        sys.exit(1)

    # Find repo root
    root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print("Downloading MNIST (this may take a minute)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    images = mnist["data"]  # (70000, 784) numpy array
    labels = mnist["target"].astype(int)  # (70000,) numpy array

    # Normalize to 0-1
    images = images / 255.0

    # Split: 60000 train, 10000 test
    train_images = images[:60000]
    train_labels = labels[:60000]
    test_images = images[60000:]
    test_labels = labels[60000:]

    print("Saving tensors...")
    save_tensor(
        data_dir / "mnist_train_images.bin",
        train_images.flatten().tolist(),
        [60000, 784],
    )
    save_tensor(
        data_dir / "mnist_train_labels.bin",
        train_labels.flatten().tolist(),
        [60000],
    )
    save_tensor(
        data_dir / "mnist_test_images.bin",
        test_images.flatten().tolist(),
        [10000, 784],
    )
    save_tensor(
        data_dir / "mnist_test_labels.bin",
        test_labels.flatten().tolist(),
        [10000],
    )

    print("Done! Files saved to data/")


if __name__ == "__main__":
    main()
