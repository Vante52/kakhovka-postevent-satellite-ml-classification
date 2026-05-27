"""
knn.py
------------------
Implementa K-Nearest Neighbors (KNN) para clasificación de cobertura terrestre
a partir del dataset TSV de bandas Sentinel-2.

Salidas:
  - Matriz de confusión
  - Overall Accuracy, Kappa, F1-Score, Precision, Recall por clase
  - Mejor valor de K encontrado por HPO (GridSearchCV)

USO:
  python knn.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# CONFIGURACIÓN
TSV_FILE   = "training_dataset.tsv"
OUTPUT_CM  = "knn_confusion_matrix.png"
BANDS      = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
TEST_SIZE  = 0.2
RANDOM_STATE = 42

#CARGAR DATOS
df = pd.read_csv(TSV_FILE, sep="\t")
print(f"\nDataset cargado: {len(df)} píxeles")
print(f"Clases:\n{df['label'].value_counts().to_string()}")

X = df[BANDS].values
y = df["label"].values

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train)} píxeles | Test: {len(X_test)} píxeles")


#NORMALIZACIÓN 
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

#HYPERPARAMETER OPTIMIZATION (HPO)
print("\nBuscando mejor K (GridSearchCV)...")

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean"],
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_sc, y_train)

best_params = grid_search.best_params_
print(f"\nMejores parámetros: {best_params}")
print(f"Mejor F1-macro (CV): {grid_search.best_score_:.4f}")

#MODELO FINAL
knn = grid_search.best_estimator_
y_pred = knn.predict(X_test_sc)

#MÉTRICAS
classes = sorted(np.unique(y))

oa    = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=classes, digits=4)

print("\n")
print(f"Overall Accuracy : {oa:.4f}  ({oa*100:.2f}%)")
print(f"Kappa Coefficient: {kappa:.4f}")
print("\nReporte por clase:")
print(report)

#MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_test, y_pred, labels=classes)

print("Matriz de confusión (filas=predicho, columnas=referencia):")
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df.to_string())

# Guardar figura
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(
    f"KNN — Confusion Matrix\n"
    f"OA: {oa*100:.2f}%  |  Kappa: {kappa:.4f}  |  "
    f"K={best_params['n_neighbors']}, w={best_params['weights']}",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=150)
print(f"\nMatriz de confusión guardada en: {OUTPUT_CM}")
plt.show()