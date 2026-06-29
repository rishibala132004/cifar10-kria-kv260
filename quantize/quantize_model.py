"""
Step 3 : Model Optimization using Vitis-AI Quantization
This script performs post-training INT8 quantization of a trained CIFAR-10
CNN model using the Vitis-AI TensorFlow 2.x quantization toolkit.
Objectives:
- Load a trained FP32 Keras model (.h5)
- Load the CIFAR-10 dataset from local files (no internet dependency)
- Perform DPU compatibility inspection
- Apply post-training quantization (INT8)
- Save the optimized quantized model for DPU compilation
Target hardware:
- Kria KV260
- DPUCZDX8G (ISA1_B4096)
This step reduces model precision and prepares the network
for efficient execution on the FPGA-based DPU.
"""
# Import required libraries
# TensorFlow: model loading and inference
# NumPy: numerical operations
# Pickle: load CIFAR-10 dataset files
# Vitis-AI: model inspection and quantization
import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow_model_optimization.quantization.keras import vitis_inspect
# File paths
# MODEL_PATH: Trained FP32 Keras model from Step 1
# QUANT_MODEL_PATH: Output path for INT8 quantized model
# CIFAR_DIR: Local CIFAR-10 dataset directory
MODEL_PATH = "model/model.h5"
QUANT_MODEL_PATH = "model/model_quantized.h5"
CIFAR_DIR = "cifar-10-batches-py"
# Load CIFAR-10 dataset from local files
# CIFAR-10 images are stored as flattened vectors:
# 3072 = 1024 R + 1024 G + 1024 B
# This function reshapes them into (32, 32, 3) RGB format.
def load_cifar10_batch(batch_file):
# Load a single CIFAR-10 batch file
    with open(batch_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
# Extract image data
    images = data[b'data']  # shape: (10000, 3072)
# Reshape to (N, 3, 32, 32) then convert to HWC format
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)  # CHW to HWC
    return images
# Load CIFAR-10 training data for quantization calibration
# Calibration data is used to determine scaling factors
# for INT8 quantization (weights and activations).
print("Loading CIFAR-10 from local directory...")
x_train = []
for i in range(1, 6):
    batch_path = os.path.join(CIFAR_DIR, f"data_batch_{i}")
    x_train.append(load_cifar10_batch(batch_path))
x_train = np.concatenate(x_train, axis=0)
# Normalize pixel values to [0, 1], matching training preprocessing
x_train = x_train.astype("float32") / 255.0
print("Calibration dataset shape:", x_train.shape)
# Load FP32 model
print("\nLoading FP32 model...")
# Load trained FP32 model (Step 1)
# compile=False since training is not required for quantization
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.summary()
# DPU compatibility inspection
# This step analyzes which layers can run on the DPU
# and which layers will fall back to CPU execution.
# Results are saved for documentation and debugging.
print("\nInspecting model for DPU compatibility...")
inspector = vitis_inspect.VitisInspector(
    target="DPUCZDX8G_ISA1_B4096"
)
# Inspect model using target DPU architecture
# dump_model=True saves inspected model
# dump_results=True saves compatibility report
inspector.inspect_model(
    model,
    input_shape=(32, 32, 3),
    dump_model=True,
    dump_results=True
)
print("\nStarting INT8 quantization...")
# Post-Training Quantization (INT8)
# Convert FP32 model to INT8 for efficient DPU execution
quantizer = vitis_quantize.VitisQuantizer(model)
# Use a small subset of training data for calibration
# This reduces quantization time while maintaining accuracy
calib_dataset = x_train[:500]
quantized_model = quantizer.quantize_model(
    calib_dataset=calib_dataset
)
# Save INT8 quantized model
# This model will be compiled into an .xmodel in the next step
quantized_model.save(QUANT_MODEL_PATH)
print("\nQuantized model saved to:", QUANT_MODEL_PATH)
# Sanity check
# Verify that the quantized model runs correctly
print("\nRunning sanity inference check...")
dummy_input = x_train[:1]
_ = quantized_model.predict(dummy_input)
print("\n Quantization completed successfully.")
