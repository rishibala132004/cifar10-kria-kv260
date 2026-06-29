# CIFAR-10 DNN Acceleration on Kria KV260

End-to-end embedded AI pipeline: DNN design and training → CPU inference baseline → INT8 quantization → FPGA/DPU-accelerated deployment on the AMD Kria KV260 using Vitis-AI.

---

## Results

| Stage | Accuracy | Latency (ms/img) | Throughput (FPS) |
|-------|----------|-----------------|------------------|
| Step 1 – PC Training (baseline network) | 84.30% | — | — |
| Step 1 – PC Training (high-accuracy network) | **90.69%** | — | — |
| Step 2 – Kria CPU inference (FP32) | 91.66% | 32 ms | ~30 FPS |
| Step 3 – Kria FPGA/DPU inference (INT8) | **90.55%** | **0.9 ms** | **~1,100 FPS** |

**35× latency reduction · 37× throughput gain · <1.1% accuracy loss from quantization**

---

## Pipeline Overview

```
Step 1: Train CNN  (PC / Colab)
   └─ VGG-inspired architecture: 8 Conv layers + 2 FC layers, 429K parameters
   └─ Techniques: data augmentation, dropout, Global Average Pooling, label smoothing
   └─ Output: model/model.h5  (FP32 Keras model)

Step 2: CPU Baseline  (Kria KV260 — ARM Cortex-A)
   └─ TensorFlow FP32 batched inference
   └─ Output: latency / throughput / accuracy baseline numbers

Step 3: FPGA/DPU Acceleration  (Kria KV260 — Vitis-AI)
   ├─ 3a. Quantization  : FP32 → INT8  (Vitis-AI PTQ, 500-image calibration set)
   ├─ 3b. Compilation   : .h5 → .xmodel  (target: DPUCZDX8G_ISA1_B4096)
   ├─ 3c. Deployment    : load bitstream → load xmodel → run DPU inference
   └─ Output: 0.9 ms/img · ~1,100 FPS
```

---

## Repository Structure

```
cifar10-kria-kv260/
├── train/
│   └── dnn_training.ipynb       # Step 1 — model architecture, training, evaluation
├── inference/
│   ├── cpu_inference.py         # Step 2 — FP32 CPU baseline on Kria KV260
│   └── dpu_inference.py         # Step 3 — INT8 DPU-accelerated inference
├── quantize/
│   └── quantize_model.py        # Step 3a — Vitis-AI post-training quantization
├── bitstream/
│   ├── dpu.bit                  # FPGA overlay bitstream (DPUCZDX8G)
│   ├── dpu.hwh                  # Hardware handoff file
│   └── dpu.xclbin               # Xilinx compiled binary
├── model/
│   ├── model.h5                 # Trained FP32 Keras model
│   └── model.xmodel             # Compiled INT8 Vitis-AI model (ready for DPU)
└── README.md
```

---

## Model Architecture

VGG-inspired CNN, sized for CIFAR-10 (32×32 RGB) and DPU resource constraints:

```
Input: (32, 32, 3)
├── Conv Block 1: 2× Conv2D(64, 3×3) + BatchNorm + MaxPool
├── Conv Block 2: 2× Conv2D(96, 3×3) + BatchNorm + MaxPool
├── Conv Block 3: 2× Conv2D(190, 3×3) + BatchNorm + MaxPool
├── Conv Block 4: 2× Conv2D(190, 3×3) + BatchNorm + MaxPool
├── Global Average Pooling          ← replaces large FC layers; reduces overfitting
├── Dense(256) + Dropout(0.5)
└── Dense(10, softmax)
```

**Design decisions:**

| Choice | Reason |
|--------|--------|
| VGG-style repeated 3×3 convolutions | Maps well to DPU hardware primitives |
| Filters: 64 → 96 → 190 | Grows capacity while staying within DPU BRAM limits |
| Global Average Pooling instead of Flatten+FC | Fewer parameters, less overfitting, DPU-friendly |
| Data augmentation (flip, crop, rotation) | CIFAR-10 has only 50K images; augmentation prevents overfitting |
| Label smoothing | Prevents overconfident softmax outputs; improves generalization |
| Adaptive LR + early stopping | Faster convergence; avoids overfitting at the end of training |

---

## INT8 Quantization — How It Works

The Kria KV260 DPU runs exclusively on **INT8**. The FP32 model must be converted before deployment.

**Post-Training Quantization (PTQ) — `quantize/quantize_model.py`:**
1. Load trained FP32 model
2. Pass 500 calibration images through — Vitis-AI records per-layer activation ranges
3. Compute `fix_point` scaling factors: `scale = 2^fix_point`
4. Quantize all weights and activations to INT8
5. Save as `model_quantized.h5` → compile to `model.xmodel`

**Why only 500 calibration images?**  
PTQ does not retrain — it only needs to observe typical activation ranges. 500 images (50 per class) is sufficient to capture representative ranges across all 10 CIFAR-10 categories.

**Input scaling in `dpu_inference.py`:**
```python
scale = 2 ** fix_point
q = np.round(img_f32_01 * scale)
q = np.clip(q, -128, 127).astype(np.int8)
```
`fix_point` is queried directly from the compiled model's tensor metadata at runtime — it is specific to this model and bitstream, not a hardcoded constant.

---

## Running the Code

### Prerequisites
- AMD Kria KV260 running Ubuntu with PYNQ 3.0.1
- Vitis-AI 2.5 installed (for quantization, run in Vitis-AI Docker on host PC)
- CIFAR-10 dataset: download from https://www.cs.toronto.edu/~kriz/cifar.html → extract `cifar-10-batches-py/`

### Step 2 — CPU Inference (run on Kria KV260)
```bash
python3 inference/cpu_inference.py \
  --model    model/model.h5 \
  --dataset_dir cifar-10-batches-py \
  --num_images full \
  --batch_size 128
```

### Step 3 — DPU Inference (run on Kria KV260)
```bash
python3 inference/dpu_inference.py \
  --bit     bitstream/dpu.bit \
  --xmodel  model/model.xmodel \
  --dataset cifar-10-batches-py
```

### Step 3a — Quantization (run in Vitis-AI Docker on host PC)
```bash
# Place model/model.h5 and cifar-10-batches-py/ in the working directory, then:
python3 quantize/quantize_model.py
# Output: model/model_quantized.h5
# Then compile:
vai_c_tensorflow2 \
  --model   model/model_quantized.h5 \
  --arch    /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  --output_dir model/ \
  --net_name model
```

---

## Environment

| Component | Version |
|-----------|---------|
| Board | AMD Kria KV260 |
| DPU | DPUCZDX8G_ISA1_B4096 |
| Vitis-AI | 2.5 |
| Vivado | 2022.x |
| PYNQ | 3.0.1 |
| TensorFlow | 2.x |
| Python | 3.8+ |

---

## Challenges

- **DPU layer compatibility**: Layers not supported by the DPU fall back to CPU (hybrid execution). Architecture was kept DPU-friendly to minimise this.
- **Quantization accuracy loss**: Held below 1.1% through representative calibration data and DPU-compatible layer choices (no exotic activations).
- **Vitis-AI / TensorFlow version pinning**: Vitis-AI 2.5 requires a specific TF 2.x subversion; mismatches cause silent failures in the quantizer.
- **NHWC vs NCHW**: Keras trains in NHWC (channels-last). `cpu_inference.py` detects the format from `model.input_shape` and transposes automatically if the model was saved as NCHW.

---

## Topics
`embedded-ai` `fpga` `kria-kv260` `vitis-ai` `cifar-10` `int8-quantization` `dpu` `deep-learning` `embedded-systems` `xilinx` `python` `tensorflow` `university-of-twente`
