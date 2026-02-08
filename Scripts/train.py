import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report


# ========================
# CONFIG
# ========================

DATASET_DIR = "dataset"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

INIT_LR = 1e-3
FINE_LR = 1e-4

MODEL_PATH = "best_model.keras"

SEED = 42


# ========================
# GPU SETUP
# ========================

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU Found:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Running on CPU")


# ========================
# DATA GENERATORS
# ========================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)


NUM_CLASSES = train_gen.num_classes
CLASS_NAMES = list(train_gen.class_indices.keys())

print("Classes:", CLASS_NAMES)


# ========================
# MODEL
# ========================

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Convert grayscale → RGB
x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=x
)

# Freeze base model
base_model.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)


# ========================
# COMPILE (PHASE 1)
# ========================

model.compile(
    optimizer=Adam(INIT_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ========================
# CALLBACKS
# ========================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ========================
# TRAIN (TRANSFER LEARNING)
# ========================

print("\n--- Transfer Learning Phase ---\n")

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ========================
# FINE TUNING
# ========================

print("\n--- Fine Tuning Phase ---\n")

# Unfreeze top layers
for layer in base_model.layers[-40:]:
    layer.trainable = True


model.compile(
    optimizer=Adam(FINE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ========================
# SAVE FINAL MODEL
# ========================

model.save("final_model.keras")

print("Final model saved as final_model.keras")


# ========================
# PLOT CURVES
# ========================

def plot(h1, h2):

    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]

    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="Train")
    plt.plot(epochs, val_acc, label="Val")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="Train")
    plt.plot(epochs, val_loss, label="Val")
    plt.legend()
    plt.title("Loss")

    plt.show()


plot(history1, history2)


# ========================
# CONFUSION MATRIX
# ========================

print("\n--- Evaluation ---\n")

val_gen.reset()

preds = model.predict(val_gen)

y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9,7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
))
