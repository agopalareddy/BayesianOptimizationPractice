#%% md
# # Install Dependencies and Fetch Dataset
#%%
random_state = 1
from pprint import pprint
#%%
from ucimlrepo import fetch_ucirepo

# fetch dataset
concrete_compressive_strength = fetch_ucirepo(id=165)

# data (as pandas dataframes)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# metadata
pprint(concrete_compressive_strength.metadata)

# variable information
pprint(concrete_compressive_strength.variables)
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
# Histogram of Concrete Compressive Strength
plt.figure(figsize=(8, 6))
sns.histplot(y_train["Concrete compressive strength"], kde=True)
plt.title("Histogram of Concrete Compressive Strength", fontsize=16)
plt.xlabel("Concrete compressive strength")
plt.ylabel("Frequency")
plt.show()
#%%
# Box Plot of Concrete Compressive Strength
plt.figure(figsize=(8, 6))
sns.boxplot(x=y_train["Concrete compressive strength"])
plt.title("Box Plot of Concrete Compressive Strength", fontsize=16)
plt.xlabel("Concrete compressive strength")
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
# The goal is to optimize the concrete compressive strength using Bayesian optimization. The objective function will be defined to minimize the negative of the compressive strength, as we want to maximize it. The parameters to be optimized will include the features of the concrete mix.
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
# ## Use Optimized Model to Optimize Concrete Mix using Bayesian Optimization
#%%
# Use the optimized model to predict concrete compressive strength
optimized_strength = opt.predict(X_test)
print("Predicted Concrete Compressive Strength using Optimized Model:")
print(optimized_strength)
#%%
# Visualize the predicted vs actual concrete compressive strength
plt.figure(figsize=(10, 6))
plt.scatter(y_test, optimized_strength, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual Concrete Compressive Strength', fontsize=16)
plt.xlabel('Actual Concrete Compressive Strength')
plt.ylabel('Predicted Concrete Compressive Strength')
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
    Objective function to minimize the negative of the compressive strength.
    This function takes a the parameters of the concrete mix and returns the negative compressive strength.
    """
    # Unpack the parameters
    cement, slag, ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age = params
    # Create a DataFrame with the parameters
    concrete_mix = pd.DataFrame({
        'Cement': [cement],
        'Blast Furnace Slag': [slag],
        'Fly Ash': [ash],
        'Water': [water],
        'Superplasticizer': [superplasticizer],
        'Coarse Aggregate': [coarse_aggregate],
        'Fine Aggregate': [fine_aggregate],
        'Age': [age]
    })
    # Predict the compressive strength using the optimized model
    predicted_strength = opt.predict(concrete_mix)
    # Return the negative compressive strength (as we want to maximize it)
    return -predicted_strength[0]
#%%
# Define the search space for the parameters of the concrete mix based on the min and max values in the training set
search_space_concrete = [
    (X_train['Cement'].min(), X_train['Cement'].max()),  # Cement
    (X_train['Blast Furnace Slag'].min(), X_train['Blast Furnace Slag'].max()),  # Slag
    (X_train['Fly Ash'].min(), X_train['Fly Ash'].max()),  # Ash
    (X_train['Water'].min(), X_train['Water'].max()),  # Water
    (X_train['Superplasticizer'].min(), X_train['Superplasticizer'].max()),  # Superplasticizer
    (X_train['Coarse Aggregate'].min(), X_train['Coarse Aggregate'].max()),  # Coarse Aggregate
    (X_train['Fine Aggregate'].min(), X_train['Fine Aggregate'].max()),  # Fine Aggregate
    (X_train['Age'].min(), X_train['Age'].max())  # Age
]
#%%
from skopt import gp_minimize
# Perform Bayesian optimization to find the optimal concrete mix parameters
result = gp_minimize(
    objective_function,
    search_space_concrete,
    n_calls=50,  # Number of evaluations
    random_state=random_state,
    verbose=True
)
#%%
# Neatly display the best parameters and the best predicted compressive strength along with column names
best_params = result.x
best_strength = -result.fun  # Negate the result to get the actual strength
print("Best parameters found by Bayesian optimization:")
for i, param in enumerate(best_params):
    print(f"{X_train.columns[i]}: {param}")
print(f"Best predicted Concrete Compressive Strength: {best_strength}")
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
# Use the optimized XGBoost model to predict concrete compressive strength
optimized_strength_xgb = xgb_opt.predict(X_test)
print("Predicted Concrete Compressive Strength using Optimized XGBoost Model:")
print(optimized_strength_xgb)
#%%
# Visualize the predicted vs actual concrete compressive strength using XGBoost
plt.figure(figsize=(10, 6))
plt.scatter(y_test, optimized_strength_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual Concrete Compressive Strength using XGBoost', fontsize=16)
plt.xlabel('Actual Concrete Compressive Strength')
plt.ylabel('Predicted Concrete Compressive Strength')
plt.grid()
plt.show()
#%%
# Objective function for Bayesian optimization with XGBoost
def objective_function_xgb(params):
    """
    Objective function to minimize the negative of the compressive strength using XGBoost.
    This function takes the parameters of the concrete mix and returns the negative compressive strength.
    """
    # Unpack the parameters
    cement, slag, ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age = params
    # Create a DataFrame with the parameters
    concrete_mix = pd.DataFrame({
        'Cement': [cement],
        'Blast Furnace Slag': [slag],
        'Fly Ash': [ash],
        'Water': [water],
        'Superplasticizer': [superplasticizer],
        'Coarse Aggregate': [coarse_aggregate],
        'Fine Aggregate': [fine_aggregate],
        'Age': [age]
    })
    # Predict the compressive strength using the optimized XGBoost model
    predicted_strength = xgb_opt.predict(concrete_mix)
    # Return the negative compressive strength (as we want to maximize it)
    return -predicted_strength[0]
#%%
# Define the search space for the parameters of the concrete mix based on the min and max values in the training set
search_space_concrete_xgb = [
    (X_train['Cement'].min(), X_train['Cement'].max()),  # Cement
    (X_train['Blast Furnace Slag'].min(), X_train['Blast Furnace Slag'].max()),  # Slag
    (X_train['Fly Ash'].min(), X_train['Fly Ash'].max()),  # Ash
    (X_train['Water'].min(), X_train['Water'].max()),  # Water
    (X_train['Superplasticizer'].min(), X_train['Superplasticizer'].max()),  # Superplasticizer
    (X_train['Coarse Aggregate'].min(), X_train['Coarse Aggregate'].max()),  # Coarse Aggregate
    (X_train['Fine Aggregate'].min(), X_train['Fine Aggregate'].max()),  # Fine Aggregate
    (X_train['Age'].min(), X_train['Age'].max())  # Age
]
#%%
from skopt import gp_minimize
# Perform Bayesian optimization to find the optimal concrete mix parameters using XGBoost
result_xgb = gp_minimize(
    objective_function_xgb,
    search_space_concrete_xgb,
    n_calls=50,  # Number of evaluations
    random_state=random_state,
    verbose=True
)
#%%
# Neatly display the best parameters and the best predicted compressive strength along with column names for XGBoost
best_params_xgb = result_xgb.x
best_strength_xgb = -result_xgb.fun  # Negate the result to get the actual strength
print("Best parameters found by Bayesian optimization for XGBoost:")
for i, param in enumerate(best_params_xgb):
    print(f"{X_train.columns[i]}: {param}")
print(f"Best predicted Concrete Compressive Strength using XGBoost: {best_strength_xgb}")
#%% md
# # Conclusion
# 
# In this notebook, I successfully fetched the Concrete Compressive Strength dataset, performed exploratory data analysis, and applied Bayesian optimization to find the optimal concrete mix parameters using both Random Forest and XGBoost models. The results showed that we could predict the concrete compressive strength effectively, and the optimized parameters were displayed for both models. The XGBoost model provided a robust alternative to the Random Forest model, demonstrating the flexibility of using different machine learning algorithms for optimization tasks.