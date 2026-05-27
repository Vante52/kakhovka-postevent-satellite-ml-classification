"""
SVM Classifier - Kakhovka Dam Collapse
Land Cover Classification using Sentinel-2 Spectral Bands
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ---------------------------------------------
# VARIOUS UTILS
# ---------------------------------------------


C_RST = "\033[0m" # reset color
C_R = "\033[91m"  # color Red
C_G = "\033[92m"  # color Green
C_Y = "\033[93m"  # color Yellow
C_B = "\033[94m"  # color Blue
dataset = "training_dataset.tsv"
datasetPerTest = 0.3 # percentage (%) of dataset used for testing the prediction
pathSaveFigs = "docs/figures" # relative path to directory where figures/images are stored
nameFigConfMatrix = "svm_confusion_matrix.png" # name of image/figure of the confusion matrix for SVM (support vector machine)

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
# 4. FEATURE SCALING  <= CRUCIAL FOR SVM
# ---------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)       # only transform, never fit on test

# ---------------------------------------------
# 5. TRAIN SVM
# ---------------------------------------------

svm = SVC(
    kernel="rbf",       # Radial Basis Function - best for spectral data
    C=10,               # regularization: higher = less tolerant to errors
    gamma="scale",      # kernel coefficient, "scale" is a safe default
    random_state=42
)

print(f"\n{C_Y}Training SVM{C_RST}...")
svm.fit(X_train, y_train)
print(f"{C_G}Done{C_RST}.")

# ---------------------------------------------
# 6. PREDICT
# ---------------------------------------------

y_pred = svm.predict(X_test)

# ---------------------------------------------
# 7. METRICS
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
ax.set_title(f"SVM Confusion Matrix\nOverall Accuracy: {oa * 100:.2f}%")
plt.tight_layout()
plt.savefig(f"{pathSaveFigs}/{nameFigConfMatrix}", dpi=150)
plt.show()
print(f"\nConfusion matrix {C_Y}figure/image saved into{C_RST} => {pathSaveFigs}/{nameFigConfMatrix}")
