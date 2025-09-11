# Copilot Instructions

This document provides guidance for AI coding agents to effectively contribute to the `BayesianOptimizationPractice` repository.

## Project Overview

This project demonstrates the application of Bayesian optimization for two main purposes:

1.  **Hyperparameter Tuning:** Finding the optimal hyperparameters for machine learning models (e.g., `RandomForestRegressor`, `XGBoost`).
2.  **Feature Optimization:** Finding the optimal input feature values to maximize a target variable (e.g., wine quality, concrete strength).

The repository contains Jupyter notebooks and Python scripts for different datasets. Each notebook/script follows a consistent workflow.

## Core Workflow

A typical workflow for any dataset in this project is as follows:

1.  **Data Fetching:** Datasets are fetched from the UCI Machine Learning Repository using the `ucimlrepo` library.

    ```python
    from ucimlrepo import fetch_ucirepo
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    ```

2.  **Data Preparation:** The data is split into training and testing sets using `sklearn.model_selection.train_test_split`.

3.  **Exploratory Data Analysis (EDA):** Standard EDA is performed using `pandas`, `matplotlib`, and `seaborn` to understand the data distribution, correlations, and identify any issues like missing values.

4.  **Model Hyperparameter Tuning:**

    - A regression model is defined (e.g., `RandomForestRegressor`).
    - `skopt.BayesSearchCV` is used to perform Bayesian optimization to find the best hyperparameters for the model. The search space is defined as a dictionary.
    - The objective is to minimize a scoring metric like `neg_mean_squared_error`.

5.  **Feature Optimization:**
    - An objective function is defined that takes a set of input features as parameters and returns the negative of the predicted target variable (since the goal is to maximize the target). This function uses the best model from the hyperparameter tuning step.
    - `skopt.gp_minimize` is used to find the combination of input features that maximizes the target variable. The search space for the features is defined based on their ranges in the dataset.

## Key Libraries and Functions

- `ucimlrepo.fetch_ucirepo`: To get datasets.
- `sklearn.model_selection.train_test_split`: For splitting data.
- `sklearn.ensemble.RandomForestRegressor`, `xgboost.XGBRegressor`: Example models used.
- `skopt.BayesSearchCV`: For hyperparameter tuning.
- `skopt.gp_minimize`: For feature optimization.
- `skopt.utils.use_named_args`, `skopt.space`: For defining search spaces.

## How to Contribute

### Adding a New Dataset and Model

To add a new experiment for a different dataset:

1.  Create a new notebook (e.g., `new_dataset.ipynb`) and a corresponding Python script (`new_dataset.py`).
2.  Follow the core workflow described above.
3.  Fetch the new dataset using `ucimlrepo` or by loading a local file.
4.  Perform EDA to understand the new dataset.
5.  Define a model and a hyperparameter search space suitable for the dataset.
6.  Implement the hyperparameter tuning using `BayesSearchCV`.
7.  Define the objective function and search space for feature optimization based on the new dataset's features.
8.  Implement feature optimization using `gp_minimize`.
9.  Add visualizations for the results.
10. Update the `README.md` to include the new notebook.

When adding code, please maintain the structure of using `#%% md` cells in the Python scripts to separate logical sections, similar to the existing scripts.

## Coding Standards

- When using jupytext, use --update to avoid overwriting existing metadata.
- Follow PEP 8 style guidelines for Python code.