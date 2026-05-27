# Kakhovka-postevent-satellite-ml-classification

Machine learning pipeline for post-event Sentinel satellite imagery classification of the Kakhovka Dam area in Ukraine, comparing DT, SVM, ANN, KNN, and Naive Bayes models for land-cover prediction.

## Preparación del entorno virtual

Este proyecto usa Python 3. En algunas distribuciones de Ubuntu el comando `python` no existe por defecto, por lo que primero se debe crear y activar un entorno virtual con `python3`.

Desde la raíz del repositorio, ejecuta:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt