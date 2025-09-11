#%% md
# # Install Dependencies and Fetch Dataset
#%%
random_state = 1
from pprint import pprint
#%%
from ucimlrepo import fetch_ucirepo

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)
#%% md
# ## Data Preparation
#%%
# test and train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#%% md
# # Exploratory Data Analysis (EDA)
#%% md
# ## Data Inspection
#%%
import pandas as pd
# Display the first few rows of the features and target
display(X_train.head())
display(y_train.head())
display(X_test.head())
display(y_test.head())
#%%
# Display the shape of the features and target
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
#%%
# Display the data types of the features and target
print("Data types of features (X):")
print(X_train.dtypes)
print("\nData types of target (y):")
print(y_train.dtypes)
#%% md
# ## Check for Missing Values
#%%
missing_values_X = X.isnull().sum()
missing_values_y = y.isnull().sum()
print("Missing values in features (X):")
print(missing_values_X)
print("\nMissing values in target (y):")
print(missing_values_y)
#%% md
# ## Summary Statistics
#%%
summary_X_train = X_train.describe()
print("Summary statistics for features (X_train):")
print(summary_X_train)
#%%
summary_y_train = y_train.describe()
print("\nSummary statistics for target (y_train):")
print(summary_y_train)
#%% md
# ## Visualizations
#%%
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Histograms for each feature
X_train.hist(figsize=(12, 10))
plt.suptitle("Histograms of Features", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
#%%
# Box plots for each feature
plt.figure(figsize=(12, 10))
for i, column in enumerate(X_train.columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=X_train[column])
    plt.title(f"Box Plot of {column}")
plt.tight_layout()
plt.show()
#%%
# Histogram of quality
plt.figure(figsize=(8, 6))
sns.histplot(y_train["quality"], kde=True)
plt.title("Histogram of quality", fontsize=16)
plt.xlabel("quality")
plt.ylabel("Frequency")
plt.show()
#%%
# Box Plot of quality
plt.figure(figsize=(8, 6))
sns.boxplot(x=y_train["quality"])
plt.title("Box Plot of quality", fontsize=16)
plt.xlabel("quality")
plt.show()
#%% md
# ## Correlation Analysis
#%%
plt.figure(figsize=(12, 8))
correlation_matrix = X_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Features')
plt.show()
#%% md
# # Bayesian Optimization
#%% md
# ## Objective Function
#%% md
# The goal is to optimize the wine quality using Bayesian optimization. The objective function will be defined to minimize the negative of the quality, as we want to maximize it. The parameters to be optimized will include the features of the wine.
#%%
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
#%%
# Define the model
model = RandomForestRegressor(random_state=random_state)
#%%
# Define the search space for hyperparameters
search_space = {
    'n_estimators': (50, 500),  # Number of trees in the forest
    'max_depth': (5, 50),        # Maximum depth of the tree
    'min_samples_split': (2, 20), # Minimum number of samples required to split an internal node
    'min_samples_leaf': (1, 20),  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}
#%%
# Define the Bayesian optimization search
opt = BayesSearchCV(
    model,
    search_space,
    n_iter=50,  # Number of iterations for optimization
    scoring='neg_mean_squared_error',  # Objective function to minimize
    cv=5,  # Cross-validation splitting strategy
    n_jobs=-1,  # Use all available cores
    random_state=random_state
)
#%%
# Fit the model using Bayesian optimization
opt.fit(X_train, y_train.values.ravel())
#%%
# Display the best parameters found by Bayesian optimization
print("Best parameters found by Bayesian optimization:")
pprint(opt.best_params_)
#%%
# Display the best score achieved
print(f"Best score achieved (negative MSE): {opt.best_score_}")
#%%
# Evaluate the optimized model on the test set
from sklearn.metrics import mean_squared_error
y_pred = opt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")
#%% md
# ## Use Optimized Model to Optimize Quality using Bayesian Optimization
#%%
# Use the optimized model to predict quality
optimized_quality = opt.predict(X_test)
print("Predicted quality using Optimized Model:")
print(optimized_quality)
#%%
# Visualize the predicted vs actual quality
plt.figure(figsize=(10, 6))
plt.scatter(y_test, optimized_quality, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual quality', fontsize=16)
plt.xlabel('Actual quality')
plt.ylabel('Predicted quality')
plt.grid()
plt.show()
#%%
# First check the actual column names in the DataFrame
print("Actual column names in X_train:")
print(X_train.columns.tolist())
#%%
# Objective function for Bayesian optimization
def objective_function(params):
    """
    Objective function to minimize the negative of the quality.
    This function takes a the parameters of the wine and returns the negative quality.
    """
    # Unpack the parameters
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol = params
    # Create a DataFrame with the parameters
    wine_feat = pd.DataFrame({
        'fixed_acidity': [fixed_acidity],
        'volatile_acidity': [volatile_acidity],
        'citric_acid': [citric_acid],
        'residual_sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free_sulfur_dioxide': [free_sulfur_dioxide],
        'total_sulfur_dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })
    # Predict the quality using the optimized model
    predicted_quality = opt.predict(wine_feat)
    # Return the negative quality (as we want to maximize it)
    return -predicted_quality[0]
#%%
# Define the search space for the parameters of the wine based on the min and max values in the training set
search_space_wine = [
    (X_train['fixed_acidity'].min(), X_train['fixed_acidity'].max()),  # fixed_acidity
    (X_train['volatile_acidity'].min(), X_train['volatile_acidity'].max()),  # Slag
    (X_train['citric_acid'].min(), X_train['citric_acid'].max()),  # Ash
    (X_train['residual_sugar'].min(), X_train['residual_sugar'].max()),  # residual_sugar
    (X_train['chlorides'].min(), X_train['chlorides'].max()),  # chlorides
    (X_train['free_sulfur_dioxide'].min(), X_train['free_sulfur_dioxide'].max()),  # free_sulfur_dioxide
    (X_train['total_sulfur_dioxide'].min(), X_train['total_sulfur_dioxide'].max()),  # total_sulfur_dioxide
    (X_train['density'].min(), X_train['density'].max()),  # density
    (X_train['pH'].min(), X_train['pH'].max()),  # pH
    (X_train['sulphates'].min(), X_train['sulphates'].max()),  # sulphates
    (X_train['alcohol'].min(), X_train['alcohol'].max())  # alcohol
]
#%%
from skopt import gp_minimize
# Perform Bayesian optimization to find the optimal wine parameters
result = gp_minimize(
    objective_function,
    search_space_wine,
    n_calls=50,  # Number of evaluations
    random_state=random_state,
    verbose=True
)
#%%
# Neatly display the best parameters and the best predicted quality along with column names
best_params = result.x
best_quality = -result.fun  # Negate the result to get the actual quality
print("Best parameters found by Bayesian optimization:")
for i, param in enumerate(best_params):
    print(f"{X_train.columns[i]}: {param}")
print(f"Best predicted quality: {best_quality}")
#%% md
# ## Use XGBoost for the Model instead of Random Forest
#%%
from xgboost import XGBRegressor
#%%
# Define the XGBoost model
xgb_model = XGBRegressor(random_state=random_state, n_jobs=-1)
#%%
# Define the Bayesian optimization search for XGBoost hyperparameters
xgb_search_space = {
    'n_estimators': (50, 500),  # Number of trees in the forest
    'max_depth': (3, 10),        # Maximum depth of the tree
    'learning_rate': (0.01, 0.3, 'uniform'),  # Step size shrinkage used in update to prevent overfitting
    'subsample': (0.5, 1.0, 'uniform'),  # Subsample ratio of the training instances
    'colsample_bytree': (0.5, 1.0, 'uniform'),  # Subsample ratio of columns when constructing each tree
    'gamma': (0, 5),  # Minimum loss reduction required to make a further partition on a leaf
    'reg_alpha': (0, 1),  # L1 regularization term on weights
    'reg_lambda': (0, 1)  # L2 regularization term on weights
}
#%%
# Define the Bayesian optimization search for XGBoost
xgb_opt = BayesSearchCV(
    xgb_model,
    xgb_search_space,
    n_iter=50,  # Number of iterations for optimization
    scoring='neg_mean_squared_error',  # Objective function to minimize
    cv=5,  # Cross-validation splitting strategy
    n_jobs=-1,  # Use all available cores
    random_state=random_state
)
#%%
# Fit the XGBoost model using Bayesian optimization
xgb_opt.fit(X_train, y_train.values.ravel())
#%%
# Display the best parameters found by Bayesian optimization for XGBoost
print("Best parameters found by Bayesian optimization for XGBoost:")
pprint(xgb_opt.best_params_)
#%%
# Display the best score achieved by XGBoost
print(f"Best score achieved (negative MSE) by XGBoost: {xgb_opt.best_score_}")
#%%
# Evaluate the optimized XGBoost model on the test set
y_pred_xgb = xgb_opt.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"Mean Squared Error on test set by XGBoost: {mse_xgb}")
#%%
# Use the optimized XGBoost model to predict quality
optimized_quality_xgb = xgb_opt.predict(X_test)
print("Predicted quality using Optimized XGBoost Model:")
print(optimized_quality_xgb)
#%%
# Objective function for Bayesian optimization with XGBoost
def objective_function_xgb(params):
    """
    Objective function to minimize the negative of the quality using XGBoost.
    This function takes the parameters of the wine and returns the negative quality.
    """
    # Unpack the parameters
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol = params
    # Create a DataFrame with the parameters
    wine_feat = pd.DataFrame({
        'fixed_acidity': [fixed_acidity],
        'volatile_acidity': [volatile_acidity],
        'citric_acid': [citric_acid],
        'residual_sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free_sulfur_dioxide': [free_sulfur_dioxide],
        'total_sulfur_dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })
    # Predict the quality using the optimized XGBoost model
    predicted_quality = xgb_opt.predict(wine_feat)
    # Return the negative quality (as we want to maximize it)
    return -predicted_quality[0]
#%%
# Define the search space for the parameters of the wine based on the min and max values in the training set
search_space_wine_xgb = [
    (X_train['fixed_acidity'].min(), X_train['fixed_acidity'].max()),  # fixed_acidity
    (X_train['volatile_acidity'].min(), X_train['volatile_acidity'].max()),  # Slag
    (X_train['citric_acid'].min(), X_train['citric_acid'].max()),  # Ash
    (X_train['residual_sugar'].min(), X_train['residual_sugar'].max()),  # residual_sugar
    (X_train['chlorides'].min(), X_train['chlorides'].max()),  # chlorides
    (X_train['free_sulfur_dioxide'].min(), X_train['free_sulfur_dioxide'].max()),  # free_sulfur_dioxide
    (X_train['total_sulfur_dioxide'].min(), X_train['total_sulfur_dioxide'].max()),  # total_sulfur_dioxide
    (X_train['density'].min(), X_train['density'].max()),  # density
    (X_train['pH'].min(), X_train['pH'].max()),  # pH
    (X_train['sulphates'].min(), X_train['sulphates'].max()),  # sulphates
    (X_train['alcohol'].min(), X_train['alcohol'].max())  # alcohol
]
#%%
from skopt import gp_minimize
# Perform Bayesian optimization to find the optimal wine parameters using XGBoost
result_xgb = gp_minimize(
    objective_function_xgb,
    search_space_wine_xgb,
    n_calls=50,  # Number of evaluations
    random_state=random_state,
    verbose=True
)
#%%
# Neatly display the best parameters and the best predicted quality along with column names for XGBoost
best_params_xgb = result_xgb.x
best_quality_xgb = -result_xgb.fun  # Negate the result to get the actual quality
print("Best parameters found by Bayesian optimization for XGBoost:")
for i, param in enumerate(best_params_xgb):
    print(f"{X_train.columns[i]}: {param}")
print(f"Best predicted quality using XGBoost: {best_quality_xgb}")
#%% md
# # Conclusion
# 
# In this notebook, I have demonstrated how to perform Bayesian optimization on a wine quality dataset using both Random Forest and XGBoost models. The process involved:
# 1. Fetching the dataset and preparing it for analysis.
# 2. Conducting exploratory data analysis (EDA) to understand the data.
# 3. Implementing Bayesian optimization to find the best hyperparameters for the models.
# 4. Evaluating the optimized models on a test set.
# 5. Using the optimized models to predict wine quality based on the features.
# The results showed that both models could effectively predict wine quality, with XGBoost providing a slightly better performance in terms of mean squared error. The best parameters found through Bayesian optimization were also displayed, allowing for further insights into the optimal conditions for wine quality prediction.