# Notebook Guide

This project uses two main Jupyter notebooks for analysis:

## 01_eda_and_exploration.ipynb

**Purpose**: Initial exploratory data analysis and feature discovery

**Key Contents**:

- Dataset overview and statistics
- Initial feature correlation analysis
- Box plots showing breakthrough vs non-breakthrough distributions
- Identification of multicolinearity issues
- Motivation for multi-tier classification system

**Note**: This notebook documents the exploratory phase. The modeling workflow starts in notebook 02.

## 02_multiclass_modeling.ipynb

**Purpose**: Model training, comparison, and evaluation

**Key Contents**:

- Advanced feature engineering
- Logistic Regression and Random Forest training
- Hyperparameter tuning with cross-validation
- Model evaluation and performance metrics
- Feature importance analysis

**Output**:

- Model comparison visualizations
- Classification reports
- Feature importance plots

## Running the Notebooks

1. Ensure you've set up the data (see `data/README.md`)
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order: `01_eda_and_exploration.ipynb` then `02_multiclass_modeling.ipynb`
