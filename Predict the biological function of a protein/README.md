# CAFA 6 Protein Function Prediction

This project implements a baseline model for the **CAFA 6 Protein Function Prediction** competition. It uses the **ESM-2** protein language model (specifically `esm2_t33_650M_UR50D`) to predict Gene Ontology (GO) terms for protein sequences.

## Project Structure

- `main.ipynb`: The main Jupyter Notebook containing the training pipeline, EDA, and prediction logic.
- `train.py`: A Python script version of the training pipeline.
- `requirements.txt`: List of Python dependencies.
- `pyproject.toml`: Project configuration and dependencies (for `uv` or modern pip).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Or if you use `uv`:
    ```bash
    uv sync
    ```

2.  **Data**:
    Place the competition data in a folder named `cafa-6-protein-function-prediction/data/`.
    Structure:
    ```
    cafa-6-protein-function-prediction/data/
    ├── Train/
    │   ├── go-basic.obo
    │   ├── train_sequences.fasta
    │   └── train_terms.tsv
    └── Test/
        └── testsuperset.fasta
    ```

## Usage

Run the notebook `main.ipynb` to train the model and generate predictions.

## Model Details

- **Backbone**: ESM-2 (650M parameters)
- **Head**: Linear layer + Sigmoid
- **Loss**: Binary Cross Entropy
- **Post-processing**: Graph-based score propagation using the GO ontology structure.
