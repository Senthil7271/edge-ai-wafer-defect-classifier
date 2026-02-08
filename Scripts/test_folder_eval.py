import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# CONFIG
# =====================
TEST_DIR = "dataset/test"
IMG_SIZE = 224
BATCH_SIZE = 32

# =====================
# LOAD MODEL
# =====================
model = tf.keras.models.load_model("final_model.keras")
print("Model input:", model.input_shape)
print("Model classes:", model.output_shape[-1])

# =====================
# GENERATOR (match training)
# =====================
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

CLASS_NAMES = list(test_gen.class_indices.keys())
print("Class order:", CLASS_NAMES)

# =====================
# PREDICT
# =====================
preds = model.predict(test_gen)

y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# =====================
# SAFE LABEL HANDLING
# =====================
labels_present = sorted(np.unique(y_true))
names_present = [CLASS_NAMES[i] for i in labels_present]

# =====================
# REPORT
# =====================
print("\n=== Classification Report ===\n")

print(classification_report(
    y_true,
    y_pred,
    labels=labels_present,
    target_names=names_present
))

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_true, y_pred, labels=labels_present)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=names_present,
    yticklabels=names_present
)

plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================
# SAVE CSV RESULTS
# =====================
results = []

for i, fname in enumerate(test_gen.filenames):
    results.append({
        "file": fname,
        "true": CLASS_NAMES[y_true[i]],
        "pred": CLASS_NAMES[y_pred[i]],
        "confidence": float(np.max(preds[i]))
    })

df = pd.DataFrame(results)
df.to_csv("test_predictions.csv", index=False)

print("\nSaved: test_predictions.csv")

# =====================
# SHOW SAMPLE OUTPUT
# =====================
print("\nSample predictions:\n")
print(df.head(20))
