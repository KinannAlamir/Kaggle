# Loan Payback Prediction

A machine learning project for predicting loan payback using Kaggle competition data.

## Setup

This project uses `uv` for dependency management and virtual environment handling.

### Prerequisites

- Python 3.13+
- uv package manager

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
   ```bash
   uv sync
   ```

### Usage

#### Running Jupyter Notebook
```bash
uv run jupyter notebook
```

#### Running Jupyter Lab
```bash
uv run jupyter lab
```

#### Running Python scripts
```bash
uv run python main.py
```

### Development

Install development dependencies:
```bash
uv sync --extra dev
```

## Project Structure

- `main.ipynb` - Main Jupyter notebook for analysis
- `data/` - Dataset files
  - `train.csv` - Training data
  - `test.csv` - Test data  
  - `sample_submission.csv` - Sample submission format
- `pyproject.toml` - Project configuration and dependencies

### Jupyter kernel (optional)

If you want Jupyter notebooks to use the `uv` environment directly, install the kernel and register it:

```bash
uv add ipykernel
uv run python -m ipykernel install --user --name loan-payback-uv --display-name "Python (loan-payback-uv)"
```

Then from the Jupyter UI choose the kernel named "Python (loan-payback-uv)".
