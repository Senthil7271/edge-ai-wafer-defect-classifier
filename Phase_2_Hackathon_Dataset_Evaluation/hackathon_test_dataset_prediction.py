import onnxruntime as ort
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import logging
import sys

# ==========================================
# 0. SETUP THE LOG FILE
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("prediction_log.txt", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    logging.info("🚀 Starting Phase 2 ONNX Inference (Strict Compliance Mode)...")
    
    # ==========================================
    # 1. BARE MINIMUM PREPROCESSING 
    # Strictly formatting to suit model input requirements
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                 # Standard default resize (No Bicubic)
        transforms.Grayscale(num_output_channels=1),   # Model requires 1 channel
        transforms.ToTensor(),                         # Converts to float
        transforms.Normalize(mean=[0.5], std=[0.5])    # Matches Phase 1 preprocess_input
    ])

    # ==========================================
    # 2. FILE PATHS 
    # ==========================================
    test_dir = r'C:\Users\Tharun\Downloads\wafer defect classifier\wafer defect classifier\hackathon_test_dataset'
    model_path = r'C:\Users\Tharun\Downloads\wafer defect classifier\wafer defect classifier\model\final_model.onnx'

    logging.info(f"\n📂 Loading test data from: {test_dir}")
    testset = datasets.ImageFolder(root=test_dir, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    
    test_class_names = testset.classes
    logging.info(f"🏷️ Found 9 Test Classes: {test_class_names}")

    # ==========================================
    # 3. LOAD ONNX MODEL
    # ==========================================
    logging.info("\n⚙️ Loading ONNX Model...")
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    # ==========================================
    # 4. THE MASTER INDEX SHIFT FIX
    # ==========================================
    def fix_prediction(pred):
        mapping = {
            0: 0, 1: 2, 2: 3, 3: 4, 
            4: 99, # Stain -> Dummy label
            5: 1, 6: 5, 7: 6, 8: 7, 9: 8
        }
        return mapping.get(pred, pred)

    all_preds = []
    all_labels = []

    # ==========================================
    # 5. INFERENCE LOOP
    # ==========================================
    logging.info("🧠 Running images through the ONNX model... (Please wait)")
    for images, labels in testloader:
        img_np = np.transpose(images.numpy(), (0, 2, 3, 1)).astype(np.float32)
        ort_inputs = {input_name: img_np}
        ort_outs = ort_session.run(None, ort_inputs)
        predicted = np.argmax(ort_outs[0], axis=1)
        
        all_preds.extend(predicted)
        all_labels.extend(labels.numpy())

    # Apply the mapping fix
    fixed_preds = np.array([fix_prediction(p) for p in all_preds])
    all_labels = np.array(all_labels)

    # ==========================================
    # 6. CALCULATE METRICS
    # ==========================================
    accuracy = accuracy_score(all_labels, fixed_preds)
    precision = precision_score(all_labels, fixed_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, fixed_preds, average='weighted', zero_division=0)

    logging.info("\n" + "="*50)
    logging.info(f"🏆 FINAL PHASE 2 ACCURACY:  {accuracy * 100:.2f}%")
    logging.info(f"🎯 FINAL PHASE 2 PRECISION: {precision * 100:.2f}%")
    logging.info(f"🔍 FINAL PHASE 2 RECALL:    {recall * 100:.2f}%")
    logging.info("="*50 + "\n")

    logging.info("📋 Detailed Classification Report:")
    report = classification_report(all_labels, fixed_preds, labels=list(range(9)), target_names=test_class_names, zero_division=0)
    logging.info(f"\n{report}")

    # ==========================================
    # 7. GENERATE CONFUSION MATRIX
    # ==========================================
    logging.info("📊 Generating and saving Confusion Matrix...")
    labels_to_plot = list(range(9)) + [99] 
    cm = confusion_matrix(all_labels, fixed_preds, labels=labels_to_plot)

    plot_labels = test_class_names + ['Stain']
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=plot_labels, yticklabels=test_class_names)
    plt.title(f'Phase 2 Confusion Matrix (Acc: {accuracy * 100:.2f}%)')
    plt.ylabel('Actual Ground Truth')
    plt.xlabel('Model Prediction')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('phase2_confusion_matrix.png', dpi=300)
    logging.info("✅ Saved confusion matrix as 'phase2_confusion_matrix.png'")
    logging.info("✅ Saved terminal output as 'prediction_log.txt'")
    
    plt.show()

if __name__ == '__main__':
    main()