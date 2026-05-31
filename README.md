# Kakhovka Post-Event Satellite ML Classification

Machine learning pipeline for post-event Sentinel-2 imagery classification of the Kakhovka Dam area (Ukraine), comparing Decision Trees, SVM, ANN, KNN and
Naive Bayes for land-cover prediction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Statement](#project-statement)
- [Project Phases](#project-phases)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Build and Run](#build-and-run)
- [Outputs](#outputs)

---

## Project Overview

This project implements a complete GeoAI workflow for supervised land-cover classification using Sentinel-2 multispectral imagery. The study area
corresponds to the post-collapse Kakhovka Dam region, where machine learning models are trained and compared using spectral signatures extracted from
satellite imagery.

The objective is to identify the best-performing classifier and use it to generate a thematic land-cover map for the entire scene.

---

## Project Statement

### Data Extraction and Dataset Construction

Training data must be extracted from Sentinel imagery and converted into a TSV dataset.

Example classes used in the project:

- Vegetation
- Water
- Bare Soil
- Built-up

### Machine Learning Models

The project requires implementation and comparison of:

1. Decision Trees (DT)
2. Support Vector Machines (SVM)
3. Artificial Neural Networks (ANN)
4. K-Nearest Neighbors (KNN)
5. Naive Bayes (NB)

### Evaluation Metrics

All models must be evaluated using:

- Confusion Matrix
- Overall Accuracy
- Kappa Coefficient
- Precision
- Recall
- F1-Score

### Hyperparameter Optimization

A hyperparameter optimization process must be performed to identify the best configuration for each model.

### Final Prediction (Inference)

The best-performing model must be applied to the complete raster stack to generate:

- A classified GeoTIFF
- A thematic land-cover map
- Area statistics per class

Requirements:

- Preserve CRS and georeferencing
- Visualize results in QGIS
- Quantify area per class

---

## Project Phases

### Phase 1 — Problem Definition

Deliverables:

- Event selection
- Study area definition
- Satellite metadata
- Dataset creation

### Phase 2 — Pre-processing and Training

Deliverables:

- TSV training dataset
- Initial model evaluation
- Confusion matrices
- Accuracy metrics

### Phase 3 — Synthesis and Prediction

Deliverables:

- Classified GeoTIFF
- Technical report
- Source code

---

## Repository Structure

```text
.
├── docs/
├── LICENSE
├── README.md
├── requirements.txt
├── decision_tree.py
├── svm.py
├── ann.py
├── knn.py
├── train_naive_bayes.py
├── dataset.py
├── training_dataset.tsv
├── entrenamiento.dbf
├── entrenamiento.prj
├── entrenamiento.shp
├── entrenamiento.shx
├── B2.tif
├── B3.tif
├── B4.tif
├── B5.tif
├── B6.tif
├── B7.tif
├── B8A.tif
├── B8.tif
├── B11.tif
└── B12.tif
```

---

## Environment Setup

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt # install packages and dependencies from the requirements.txt file
```

---

## Build and Run

Activate the environment:

```bash
source .venv/bin/activate # load the virtual environment for python
```

Run each model independently:

### Decision Tree

```bash
python decision_tree.py
```

### Support Vector Machine

```bash
python svm.py
```

### Artificial Neural Network

```bash
python ann.py
```

### K-Nearest Neighbors

```bash
python knn.py
```

### Naive Bayes

```bash
python train_naive_bayes.py
```

---

## Outputs

Generated outputs include:

- Confusion matrices (.png)
- Classification reports
- Performance metrics
- Trained models (when applicable)
- Final classified raster (GeoTIFF)
- Land-cover statistics

---

## License

Academic project developed for the EDT course.
