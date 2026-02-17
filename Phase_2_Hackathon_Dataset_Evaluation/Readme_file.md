🧠 Wafer Defect Classification – Phase 2 (ONNX Inference)
📌 Overview

This project performs multi-class classification of wafer defect images using a MobileNetV2-based deep learning model.

The model was:

Trained in TensorFlow / Keras

Exported to ONNX format

Evaluated using ONNX Runtime

Phase 2 focuses on running inference on the provided test dataset and computing performance metrics.

🏗 Model Details

Architecture: MobileNetV2 (ImageNet Pretrained)

Input Shape: (224, 224, 1) (Grayscale)

Grayscale → RGB conversion handled internally

Output: Softmax layer for 10 classes

📂 Dataset Structure
Training Classes (10)

Bridge

Clean

cmp

Cracks

LER

open

other

particle contamination

Stain

via

Test Dataset Classes (9)

Bridge

Clean

cmp

Cracks

LER

open

other

particle contamination

via

Note: The test dataset does not contain the “Stain” class.

Evaluation was performed strictly on the available 9 classes.

⚙ Preprocessing

During training:

Images loaded in grayscale

Resized to 224×224

MobileNetV2 preprocessing applied

Input shape maintained as (224,224,1)

During ONNX inference:

Grayscale images used

Resized to 224×224

Converted to NHWC format

Model predictions obtained via ONNX Runtime

📊 Evaluation Metrics

The following metrics were computed:

Accuracy

Precision

Recall

F1 Score

Micro Average

Macro Average

Weighted Average

Confusion Matrix

📈 Results

Example results from Phase 2 testing:

Accuracy  : ~25%
Micro Avg : ~0.25
Macro Avg : ~0.25
Weighted  : ~0.23


Observations:

Model performs above random baseline (~11% for 9 classes).

Some defect classes show moderate classification performance.

Lower performance in certain classes may be influenced by dataset distribution and labeling variations.

▶ How to Run Inference
python hackathon_test_dataset_prediction.py


Make sure to update:

Test dataset path

ONNX model path

📦 Dependencies

Install required packages:

pip install onnxruntime torch torchvision scikit-learn matplotlib seaborn

📌 Outputs Generated

Terminal evaluation metrics

prediction_log.txt (saved logs)

phase2_confusion_matrix.png (confusion matrix image)

🔍 Key Points

Model exported successfully to ONNX.

Inference pipeline validated.

Evaluation performed using standard multi-class metrics.

Results reflect actual performance on provided test dataset.
