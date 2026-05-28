"""
Decision Tree Classifier - Kakhovka Dam Collapse
Land Cover Classification using Sentinel-2 Spectral Bands

Outputs:
  - Confusion matrix figure
  - Overall Accuracy
  - Kappa coefficient
  - F1-score (macro and weighted)

Run:
  python decision_tree.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
DATASET_PATH = Path("training_dataset.tsv")
OUTPUT_DIR = Path("docs/figures")
CM_FILENAME = "dt_confusion_matrix.png"

BAND_COLS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
LABEL_COL = "label"


def load_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontro {dataset_path}. Ejecuta primero `python dataset.py`."
        )

    df = pd.read_csv(dataset_path, sep="\t")
    required_cols = BAND_COLS + [LABEL_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {dataset_path}: {missing}")

    df = df[required_cols].copy()
    df[BAND_COLS] = df[BAND_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required_cols)

    X = df[BAND_COLS]
    y = df[LABEL_COL].astype(str)
    return X, y


def main() -> None:
    X, y_text = load_data(DATASET_PATH)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    class_names = encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("Decision Tree training finished.")
    print(f"Best params: {search.best_params_}")
    print(f"Overall Accuracy: {oa:.4f} ({oa * 100:.2f}%)")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print(f"F1-score Macro: {f1_macro:.4f}")
    print(f"F1-score Weighted: {f1_weighted:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / CM_FILENAME

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(
        "Decision Tree Confusion Matrix\n"
        f"OA: {oa * 100:.2f}% | Kappa: {kappa:.4f} | F1-macro: {f1_macro:.4f}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\nConfusion matrix guardada en: {output_path}")


if __name__ == "__main__":
    main()