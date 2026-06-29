"""
Step 2 – CPU-only Inference on Kria KV260
This script evaluates a trained CIFAR-10 CNN model using
CPU-only execution on the Kria KV260 board.
Purpose of this step:
- Establish a performance baseline before DPU acceleration
- Measure accuracy, latency, and throughput on CPU
- Enable fair comparison with Step 3 (DPU inference)
Key characteristics:
- FP32 inference using TensorFlow / Keras
- Batched execution for realistic performance measurement
- Automatic handling of NHWC / NCHW model input formats
"""
import os
# Reduce TensorFlow verbosity (must be set BEFORE importing TF)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all logs, 3=errors only
import numpy as np
import pickle
from time import time
import argparse
# Load CIFAR-10 test data from local directory
def load_cifar10_from_directory(directory):
"""
    Loads the CIFAR-10 test_batch from the cifar-10-batches-py directory.

    Returns:
        x_test : numpy array of shape (N, 32, 32, 3), float32
        y_test : numpy array of shape (N,), int64
    """
    def load_batch(batch_file):
        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"]   # (N, 3072)
        labels = batch[b"labels"] # (N,)
        return images, labels
    test_path = os.path.join(directory, "test_batch")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find 'test_batch' in: {directory}\n"
        )
# Load raw CIFAR-10 data        
    x_test, y_test = load_batch(test_path)
# Convert from (N, 3072) to (N, 32, 32, 3) RGB format
    x_test = x_test.reshape(x_test.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    return x_test.astype("float32"), np.array(y_test, dtype=np.int64)
# Main inference routine
def main():
# Parse command-line arguments
    parser = argparse.ArgumentParser(description="CPU inference for CIFAR-10 RGB .h5 model (FAST batched, quiet)")
    parser.add_argument("--mode", type=str, required=False, default="1",
                        help="Kept for compatibility with your command (not used).")
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 Keras model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to cifar-10-batches-py directory")
    parser.add_argument("--num_images", type=str, default="100",
                        help="Number of images (e.g. 100) or 'full' for all 10000")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for model.predict (try 128/256/512)")
    args = parser.parse_args()
# Import TensorFlow AFTER environment variables are set
    import tensorflow as tf
    from tensorflow.keras.models import load_model
# Silence TensorFlow Python-level logs
    tf.get_logger().setLevel("ERROR")
# Load CIFAR-10 test dataset
    x_test, y_test = load_cifar10_from_directory(args.dataset_dir)
    print(f"Test samples: {x_test.shape[0]}")
# Normalize images to [0, 1]
    x_test_norm = x_test / 255.0
    print("Normalization complete.")
# Load trained FP32 model
    model = load_model(args.model, compile=False)
    print(f"Model '{args.model}' loaded successfully.")
    print(f"Model input shape: {model.input_shape}")
# Select number of images
    if str(args.num_images).lower() == "full":
        num_images = x_test_norm.shape[0]
    else:
        num_images = min(int(args.num_images), x_test_norm.shape[0])
# Handle NHWC vs NCHW input format
    in_shape = model.input_shape  
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected model input_shape: {in_shape}")
# If model expects channels-first (NCHW), transpose input
    if in_shape[1] == 3 and in_shape[2] == 32 and in_shape[3] == 32:
        x_infer = np.transpose(x_test_norm[:num_images], (0, 3, 1, 2))
    else:
        x_infer = x_test_norm[:num_images]
    print(f"Running inference on {num_images} images (batch_size={args.batch_size})...")
# Timed batched inference
    start = time()
    probs = model.predict(x_infer, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    stop = time()
# Compute performance metrics
    true_classes = y_test[:num_images]
    correct = int(np.sum(y_pred == true_classes))
    accuracy = correct / num_images * 100.0
    total_time = stop - start
    avg_time_per_image = total_time / num_images
    fps = num_images / total_time if total_time > 0 else float("inf")
# Report results
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total execution time: {total_time:.4f} s")
    print(f"Average inference time per image: {avg_time_per_image * 1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
if __name__ == "__main__":
    main()
# Entry point