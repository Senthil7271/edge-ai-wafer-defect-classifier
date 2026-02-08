# 🔬🧠 Edge-AI Wafer Defect Classification — MobileNetV2 

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)
![API](https://img.shields.io/badge/API-Keras-red)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green)
![Input](https://img.shields.io/badge/Input-Grayscale-lightgrey)
![Training](https://img.shields.io/badge/Training-GPU%20CUDA-success)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96%25-brightgreen)
![Deployment](https://img.shields.io/badge/Export-ONNX-blueviolet)
![Target](https://img.shields.io/badge/Target-Edge%20AI-brightgreen)

---

# 📌 Project Overview

This project implements an **Edge-AI semiconductor wafer defect classification system** using a **MobileNetV2 classifier trained on grayscale inspection images**.

The model is built with **TensorFlow + Keras**, trained using **CUDA-accelerated GPU**, and exported to **ONNX format** for lightweight edge deployment.

The design goal is to balance:

* ⚡ Low latency
* 💾 Small model size
* 🧠 High classification accuracy
* 🏭 Edge manufacturing constraints

---

# 🎯 Objectives

* Detect and classify wafer/die defects
* Support multiple defect categories
* Train using grayscale inspection images
* Use lightweight MobileNetV2 backbone
* Enable ONNX edge deployment
* Provide full evaluation metrics

---

# 🖼️ Model Pipeline

```
Wafer Inspection Image
        │
        ▼
Grayscale Conversion
        │
        ▼
Resize → 224×224
        │
        ▼
Gray → 3-channel adaptation
        │
        ▼
MobileNetV2 Classifier
        │
        ▼
Defect Class Prediction
```

---

# 🧬 Defect Class Descriptions

The model classifies wafer and die inspection images into the following semiconductor defect categories:

---

### 🔗 Bridge

Unintended conductive connection between adjacent metal lines or features.
May create electrical shorts and cause circuit malfunction.

**Typical cause:** Lithography or metal patterning errors.

---

### 🧩 Cracks

Physical fractures in material layers or device structures.
Can propagate and reduce long-term reliability.

**Typical cause:** Mechanical stress or thermal cycling.

---

### 📏 LER (Line Edge Roughness)

Irregular or rough feature edges instead of smooth boundaries.
Affects critical dimensions and electrical performance.

**Typical cause:** Lithography and etch process variation.

---

### 💧 Stain

Localized discoloration or residue patches on the wafer surface.

**Typical cause:** Chemical residue, poor rinse, or process contamination.

---

### ⚙️ CMP (Chemical Mechanical Planarization defect)

Surface non-uniformity or slurry residue after CMP polishing.

**Typical cause:** Over-polish, slurry residue, pad wear.

---

### 🔓 Open

Broken or disconnected conductive path leading to open circuits.

**Typical cause:** Etch breaks, missing metal, or via fill failure.

---

### 🧪 Particle Contamination

Foreign particles or debris present on the wafer surface that interfere with patterns.

**Typical cause:** Environmental or handling contamination.

---

### 🕳 Via Defect

Defects in via holes connecting metal layers, including voids or misalignment.

**Typical cause:** Via etch or metallization issues.

---

### 🧼 Clean

No visible defect present.
Represents nominal defect-free inspection regions and serves as the baseline class.

---

### ❓ Other

Abnormal patterns that do not match predefined defect categories.
Acts as a catch-all class for uncommon or mixed anomalies.

---




---

# ⚙️ Training Setup

| Component         | Used                            |
| ----------------- | ------------------------------- |
| Framework         | TensorFlow + Keras              |
| Model             | MobileNetV2                     |
| Input             | Grayscale                       |
| Image Size        | 224×224                         |
| Training Hardware | NVIDIA GPU                      |
| Acceleration      | CUDA                            |
| Strategy          | Transfer Learning + Fine-tuning |

---

# 📊 Test Performance (Actual Results)

**Test Samples:** 107
**Overall Accuracy:** **0.96**

## Macro Metrics

| Metric    | Score |
| --------- | ----- |
| Precision | 0.97  |
| Recall    | 0.96  |
| F1 Score  | 0.96  |

---

## 📈 Per-Class Performance

| Class                  | Precision | Recall | F1   |
| ---------------------- | --------- | ------ | ---- |
| Bridge                 | 0.91      | 1.00   | 0.95 |
| Clean                  | 0.90      | 1.00   | 0.95 |
| Cracks                 | 1.00      | 0.92   | 0.96 |
| LER                    | 1.00      | 1.00   | 1.00 |
| Stain                  | 1.00      | 1.00   | 1.00 |
| cmp                    | 1.00      | 0.90   | 0.95 |
| open                   | 1.00      | 0.86   | 0.92 |
| other                  | 1.00      | 0.89   | 0.94 |
| particle contamination | 0.92      | 1.00   | 0.96 |
| via                    | 1.00      | 1.00   | 1.00 |

---

# 🔍 Confusion Matrix

Confusion matrix generated on the held-out test set:


![Confusion Matrix](Confusion_Matrix.jpeg)



Highlights:

* Perfect classification for: **LER, Stain, via**
* Minor confusion between: cracks/clean and open/bridge classes
* Strong diagonal dominance → stable classifier

---

# 📦 Edge Deployment — ONNX

Model exported from TensorFlow/Keras → ONNX for edge inference.

## Export
```
[convert_savedmodel_onnx.py](convert_savedmodel_onnx.py)
```


✔ CPU compatible
✔ Edge runtime ready
✔ Quantization ready

---

# 💾 Model Footprint

| Artifact    | Size      |
| ----------- | ----------|
| Keras Model | 26.437 MB |
| ONNX Model  | 9.251  MB |

Optimized for edge compute limits.

---

# 📁 Repository Structure

```
├── train.py
├── evaluate.py
├── export_onnx.py
├── inference_onnx.py
├── requirements.txt
│
├── models/
│   ├── mobilenetv2_gray.h5
│   └── defect_model.onnx
│
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── metrics.json
```

---

# ▶️ How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py
```

Evaluate:

```bash
python test_folder_eval.py
```

Export ONNX:

```bash
python export_savedmodel_onnx.py
```

---

# ⚡ Edge Optimization Techniques

* Lightweight MobileNetV2 backbone
* Grayscale training (reduced redundancy)
* Transfer learning
* Resolution control
* ONNX export
* Edge-ready inference pipeline

---

# ⭐ Submission Deliverables Included

* Dataset structure (Train / Validation / Test)
* Training pipeline code
* Evaluation metrics
* Confusion matrix
* ONNX model
* Edge inference script
* Reproducible workflow

---

If useful, consider giving this repository a ⭐
