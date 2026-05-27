"""
Neural Network Classifier - Kakhovka Dam Collapse
Land Cover Classification using Sentinel-2 Spectral Bands
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------
# VARIOUS UTILS
# ---------------------------------------------
C_RST = "\033[0m"  # reset color
C_R   = "\033[91m" # color Red
C_G   = "\033[92m" # color Green
C_Y   = "\033[93m" # color Yellow
C_B   = "\033[94m" # color Blue

dataset           = "training_dataset.tsv"
datasetPerTest    = 0.3  # percentage (%) of dataset used for testing the prediction
pathSaveFigs      = "docs/figures"  # relative path to directory where figures/images are stored
nameFigConfMatrix = "nn_confusion_matrix.png"   # name of confusion matrix figure
nameFigHistory    = "nn_training_history.png"   # name of training history figure

# ---------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------
df = pd.read_csv(dataset, sep="\t")

print("Dataset shape:", df.shape)
print(f"\n{C_Y}Class distribution{C_RST}:")
print(df["label"].value_counts())

# ---------------------------------------------
# 2. FEATURES AND LABELS
# ---------------------------------------------
BAND_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

X = df[BAND_COLS].values
y = df["label"].values

# Encode string labels => integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(le.classes_)
print("\nClasses:", le.classes_)

# ---------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=datasetPerTest,
    random_state=42,
    stratify=y_encoded   # keeps class proportion in both splits
)

print(f"\n{C_B}Train samples{C_RST} - {len(X_train)} samples: {Counter(le.inverse_transform(y_train))}")
print(f"{C_G}Test  samples{C_RST} - {len(X_test)} samples: {Counter(le.inverse_transform(y_test))}")

# ---------------------------------------------
# 4. FEATURE SCALING
# ---------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)  # only transform, never fit on test

# ---------------------------------------------
# 5. ONE-HOT ENCODE LABELS
#    Keras needs labels as [1,0,0,0] not 0
# ---------------------------------------------
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# ---------------------------------------------
# 6. BUILD AND TRAIN NEURAL NETWORK
# ---------------------------------------------
model = Sequential([
    Dense(64,  activation="relu", input_shape=(len(BAND_COLS),)),  # input layer
    Dropout(0.3),                                                   # avoid overfitting
    Dense(128, activation="relu"),                                  # hidden layer
    Dropout(0.3),
    Dense(64,  activation="relu"),                                  # hidden layer
    Dense(num_classes, activation="softmax")                        # output layer
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print(f"\n{C_Y}Training Neural Network{C_RST}...")
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.1,  # 10% of train used for validation during training
    verbose=1
)
print(f"{C_G}Done{C_RST}.")

# ---------------------------------------------
# 7. PREDICT
# ---------------------------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)  # pick class with highest probability

# ---------------------------------------------
# 8. METRICS
# ---------------------------------------------

# - Overall Accuracy -
oa = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {C_G}{oa * 100:.2f}{C_RST}%")

# - Per-class report -
print(f"\n{C_B}Classification Report{C_RST}:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

# - Confusion Matrix -
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Neural Network Confusion Matrix\nOverall Accuracy: {oa * 100:.2f}%")
plt.tight_layout()
plt.savefig(f"{pathSaveFigs}/{nameFigConfMatrix}", dpi=150)
plt.show()
print(f"\nConfusion matrix {C_Y}figure/image saved into{C_RST} => {pathSaveFigs}/{nameFigConfMatrix}")

# ---------------------------------------------
# 9. TRAINING HISTORY
# ---------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Validation")
ax1.set_title("Accuracy over epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(history.history["loss"],     label="Train")
ax2.plot(history.history["val_loss"], label="Validation")
ax2.set_title("Loss over epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{pathSaveFigs}/{nameFigHistory}", dpi=150)
plt.show()
print(f"\nTraining history {C_Y}figure/image saved into{C_RST} => {pathSaveFigs}/{nameFigHistory}")
