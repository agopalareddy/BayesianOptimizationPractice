# Bayesian Optimization for Material and Product Optimization

This repository contains a collection of Jupyter notebooks demonstrating the application of Bayesian optimization techniques to various regression problems. The notebooks showcase how to use Bayesian optimization for both model hyperparameter tuning and for finding optimal input feature values to maximize target variables.

## Project Overview

The project explores three different datasets from the UCI Machine Learning Repository, applying similar machine learning techniques to each:

1. **Superconductivity Dataset**: Predicting and optimizing the critical temperature of superconducting materials
2. **Concrete Dataset**: Predicting and optimizing the compressive strength of concrete mixtures
3. **Wine Quality Dataset**: Predicting and optimizing the quality of wine based on physicochemical properties

Each notebook follows a structured approach:
- Data acquisition and preparation
- Exploratory data analysis (EDA)
- Model training using Bayesian optimization for hyperparameter tuning
- Prediction and evaluation
- Feature optimization using Bayesian optimization to maximize target variables

## Notebooks

### WIP_superconductivity_dataset.ipynb

This notebook analyzes superconductivity data to predict and optimize critical temperature. It demonstrates:
- How to use Bayesian optimization to tune Random Forest and XGBoost models
- How to find the optimal material properties that maximize critical temperature
- Visualization of model performance and feature relationships

### concrete_dataset.ipynb

This notebook analyzes concrete data to predict and optimize compressive strength. It demonstrates:
- How to use Bayesian optimization to tune Random Forest and XGBoost models
- How to find the optimal concrete mix parameters that maximize compressive strength
- Visualization of model performance and feature relationships

### wine_dataset.ipynb

This notebook analyzes wine data to predict and optimize wine quality. It demonstrates:
- How to use Bayesian optimization to tune Random Forest and XGBoost models
- How to find the optimal wine properties that maximize quality ratings
- Visualization of model performance and feature relationships

## Installation

To run these notebooks, you'll need Python 3.6+ and the following packages:

```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn scikit-optimize xgboost ucimlrepo
```

## Usage

1. Clone this repository
2. Install the required dependencies
3. Launch Jupyter Notebook or Jupyter Lab
4. Open and run the notebooks

```bash
jupyter notebook
```

## Dependencies

- **jupyter**: For running the notebooks
- **numpy**: For numerical operations
- **pandas**: For data manipulation and analysis
- **matplotlib** and **seaborn**: For data visualization
- **scikit-learn**: For machine learning models and evaluation metrics
- **scikit-optimize**: For Bayesian optimization
- **xgboost**: For gradient boosting models
- **ucimlrepo**: For fetching datasets from the UCI Machine Learning Repository

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- UCI Machine Learning Repository for providing the datasets
- The scikit-optimize team for their Bayesian optimization implementation