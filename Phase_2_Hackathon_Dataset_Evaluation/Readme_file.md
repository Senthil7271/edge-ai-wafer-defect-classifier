🧠 Wafer Defect Classification
Phase 2 – ONNX Inference Evaluation
<p align="center"> <img src="phase2_confusion_matrix.png" width="600"> </p>
🚀 Project Overview

This project performs multi-class wafer defect classification using a deep learning model based on MobileNetV2.

🔹 Trained in TensorFlow / Keras
🔹 Exported to ONNX
🔹 Evaluated using ONNX Runtime

🏗 Model Architecture
Input (224x224x1 Grayscale)
        ↓
Grayscale → RGB (internal conversion)
        ↓
MobileNetV2 (ImageNet pretrained)
        ↓
Global Average Pooling
        ↓
Dense + Dropout
        ↓
Softmax (10 classes)

📂 Dataset Summary
🏷 Training Classes (10)
Class	Description
Bridge	Pattern bridging defect
Clean	No defect
cmp	CMP related defect
Cracks	Structural cracks
LER	Line edge roughness
open	Open circuit
other	Miscellaneous
particle contamination	Particle defects
Stain	Surface stain
via	Via defect
🧪 Test Dataset (9 Classes)

⚠ "Stain" class not present in test dataset.

Evaluation performed on available 9 classes.

📊 Performance Metrics
🎯 Overall Performance
Metric	Score
Accuracy	~25%
Micro F1	~0.25
Macro F1	~0.25
Weighted F1	~0.23
📈 Class-Level Performance (Example)
Bridge       → F1: 0.22
CMP          → F1: 0.38
Clean        → F1: 0.28
Crack        → F1: 0.44
LER          → F1: 0.26
Open         → F1: 0.24
Other        → F1: 0.00
Particle     → F1: 0.26
VIA          → F1: 0.16

🔍 Key Observations

✔ Performance above random baseline (~11%)
✔ Stronger predictions in CMP and Crack classes
✔ Lower recall for “Other” class
✔ Confusion observed among structurally similar defect types

⚙ Preprocessing Pipeline
During Training

Grayscale images

Resize → 224×224

Performance reflects realistic multi-class classification challenges under dataset constraints.
