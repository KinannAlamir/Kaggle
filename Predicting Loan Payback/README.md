# Loan Payback Prediction - Kaggle Playground Series 2025

A comprehensive machine learning solution for predicting loan payback probability using advanced feature engineering and state-of-the-art models.

## 🎯 Competition Overview

Predict the probability that a borrower will pay back their loan based on 13 original features including:
- Demographics (gender, marital status, education)
- Financial metrics (annual income, debt-to-income ratio, credit score)
- Loan details (amount, interest rate, purpose, grade)
- Employment status

## 📊 Dataset

- **Training Data**: 593,994 samples
- **Test Data**: 254,569 samples
- **Target Distribution**: 79.9% loans paid back (imbalanced)
- **Missing Values**: None

## 🔧 Approach

### Feature Engineering (33 Features Created)

1. **Financial Risk Metrics**
   - Risk score, payment capacity, debt burden, affordability index

2. **Interactions & Transformations**
   - Credit-income interactions, log transformations, square root credit score

3. **Categorical Engineering**
   - Grade/subgrade extraction, credit score binning, frequency encoding

### Models

1. **CatBoost with Optuna Optimization** (Champion)
   - 20 optimization trials
   - Validation AUC: **0.9214**
   - Top features: Employment Status (54.9%), Debt-to-Income (14.1%)

2. **TabNet (Attention-Based Deep Learning)**
   - Sequential attention mechanism
   - Validation AUC: 0.9130

3. **Hybrid Ensemble**
   - CatBoost + TabNet combination
   - Validation AUC: 0.9213

## 🏆 Results

| Model | Validation AUC | Description |
|-------|---------------|-------------|
| **CatBoost (Optimized)** | **0.9214** | Best single model |
| Hybrid Ensemble | 0.9213 | CatBoost + TabNet |
| TabNet | 0.9130 | Attention-based NN |

## 🛠️ Setup

This project uses `uv` for dependency management.

### Prerequisites

- Python 3.13+
- uv package manager

### Installation

```bash
# Install dependencies
uv sync

# Run Jupyter
uv run jupyter notebook
```

### Key Dependencies

- polars, pandas, numpy - Data processing
- scikit-learn - ML utilities
- catboost, optuna - Gradient boosting & optimization
- pytorch-tabnet - Attention-based deep learning
- matplotlib, seaborn - Visualization

## 💡 Key Insights

- **Well-tuned traditional models** (CatBoost) can outperform complex transformers on tabular data
- **Hyperparameter optimization** with Optuna is crucial for performance
- **Feature quality** matters more than quantity (33 engineered features optimal)
- **Employment status** is by far the most important predictor (54.9% importance)
- **TabNet provides different insights** through attention mechanism but didn't improve AUC

## 📝 License

MIT License - Feel free to use and adapt for your own Kaggle competitions!
