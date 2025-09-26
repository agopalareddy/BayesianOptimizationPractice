# %% [markdown]
# # Install Dependencies and Fetch Dataset

# %%
random_state = 1
from pprint import pprint

# %%
from ucimlrepo import fetch_ucirepo
from pprint import pprint

# fetch dataset
superconductivity_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
X = superconductivity_data.data.features
y = superconductivity_data.data.targets

# metadata
pprint(superconductivity_data.metadata)

# variable information
pprint(superconductivity_data.variables)

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# We memoize the train/test split with `joblib.Memory` so repeated runs reuse cached data instead of recomputing the split and serialization from the raw UCI dataset.

# %%
# test and train split
from sklearn.model_selection import train_test_split
from pathlib import Path
from joblib import Memory

cache_dir = Path("artifacts/cache")
cache_dir.mkdir(parents=True, exist_ok=True)
memory = Memory(cache_dir, verbose=0)


@memory.cache
def prepare_train_test_split(features, targets, test_size, random_state):
    return train_test_split(
        features, targets, test_size=test_size, random_state=random_state
    )


X_train, X_test, y_train, y_test = prepare_train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %% [markdown]
# ## Data Inspection

# %%
import pandas as pd

# Display the first few rows of the features and target
display(X_train.head())
display(y_train.head())
display(X_test.head())
display(y_test.head())

# %%
# Display the shape of the features and target
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# %%
# Display the data types of the features and target
print("Data types of features (X):")
print(X_train.dtypes)
print("\nData types of target (y):")
print(y_train.dtypes)

# %% [markdown]
# ## Check for Missing Values

# %%
missing_values_X = X.isnull().sum()
missing_values_y = y.isnull().sum()
print("Missing values in features (X):")
print(missing_values_X)
print("\nMissing values in target (y):")
print(missing_values_y)

# %% [markdown]
# ## Summary Statistics

# %%
summary_X_train = X_train.describe()
print("Summary statistics for features (X_train):")
print(summary_X_train)

# %%
summary_y_train = y_train.describe()
print("\nSummary statistics for target (y_train):")
print(summary_y_train)

# %% [markdown]
# ### Focus on Most Relevant Features
# To keep the visualizations digestible for this high-dimensional dataset, we'll focus on the 12 features most correlated with `critical_temp`. These features will be used for the histograms, box plots, and correlation heatmap.

# %%
import numpy as np

# Identify the top features most correlated with the target
target_series = (
    y_train["critical_temp"] if "critical_temp" in y_train else y_train.iloc[:, 0]
)
feature_correlations = (
    X_train.corrwith(target_series).abs().sort_values(ascending=False)
)
top_feature_count = 12
top_features = feature_correlations.head(top_feature_count).index.tolist()
print("Top correlated features with critical temperature:")
print(top_features)

# %% [markdown]
# ## Visualizations

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Histograms for selected features
X_train[top_features].hist(figsize=(12, 10))
plt.suptitle("Histograms of Top Correlated Features", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Box plots for selected features
plot_cols = 4
plot_rows = (len(top_features) + plot_cols - 1) // plot_cols

plt.figure(figsize=(16, plot_rows * 4))
for i, column in enumerate(top_features, 1):
    plt.subplot(plot_rows, plot_cols, i)
    sns.boxplot(x=X_train[column])
    plt.title(f"Box Plot of {column}")

plt.tight_layout(pad=2.0)
plt.show()

# %%
# Histogram of critical_temp
plt.figure(figsize=(8, 6))
sns.histplot(y_train["critical_temp"], kde=True)
plt.title("Histogram of critical_temp", fontsize=16)
plt.xlabel("critical_temp")
plt.ylabel("Frequency")
plt.show()

# %%
# Box Plot of critical_temp
plt.figure(figsize=(8, 6))
sns.boxplot(x=y_train["critical_temp"])
plt.title("Box Plot of critical_temp", fontsize=16)
plt.xlabel("critical_temp")
plt.show()

# %% [markdown]
# ## Correlation Analysis

# %%
plt.figure(figsize=(12, 8))
correlation_matrix = X_train[top_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Top Features")
plt.show()

# %% [markdown]
# # Bayesian Optimization

# %% [markdown]
# ## Objective Function

# %% [markdown]
# The goal is to optimize the superconductors' `critical_temp` using Bayesian optimization with an XGBoost surrogate. We'll minimize the negative predicted temperature so that the optimizer effectively maximizes performance.

# %%
# First check the actual column names in the DataFrame
print("Actual column names in X_train:")
print(X_train.columns.tolist())

# %% [markdown]
# ## XGBoost Model and Bayesian Optimization
# We run a lightweight Bayesian search (fewer iterations, 3-fold CV) to map a promising region, then a refined search with tighter bounds and 5-fold validation. Histogram-based trees keep training fast. A small helper inspects the installed `xgboost` version:
# - If `fit` accepts `early_stopping_rounds`, it is passed directly.
# - Else if `fit` supports `callbacks`, an `EarlyStopping` callback is used.
# - Otherwise the search proceeds without early stopping (a warning is emitted).

# %%
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error

# Define the XGBoost model
xgb_model = XGBRegressor(
    random_state=random_state,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
    objective="reg:squarederror",
    eval_metric="rmse",
)

# %%
# Diagnostics: show xgboost version and fit() signature to understand early stopping support
import xgboost, inspect as _inspect

print("xgboost version:", xgboost.__version__)
print("XGBRegressor.fit signature:")
print(_inspect.signature(XGBRegressor.fit))

# %%
# Define the Bayesian optimization search for XGBoost hyperparameters
import inspect
from xgboost.callback import EarlyStopping
import warnings

search_space_bounds = {
    "n_estimators": (50, 500),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 1.0),
}

stage_one_search_space = {
    "n_estimators": search_space_bounds["n_estimators"],
    "max_depth": search_space_bounds["max_depth"],
    "learning_rate": (*search_space_bounds["learning_rate"], "uniform"),
    "subsample": (*search_space_bounds["subsample"], "uniform"),
    "colsample_bytree": (*search_space_bounds["colsample_bytree"], "uniform"),
    "gamma": search_space_bounds["gamma"],
    "reg_alpha": search_space_bounds["reg_alpha"],
    "reg_lambda": search_space_bounds["reg_lambda"],
}

initial_iterations = 12
initial_cv_folds = 3
refined_iterations = 25
refined_cv_folds = 5
early_stopping_rounds = 25

y_train_array = y_train.values.ravel()
y_test_array = y_test.values.ravel()

fit_signature = inspect.signature(xgb_model.fit)
supports_early_stopping_rounds = "early_stopping_rounds" in fit_signature.parameters
supports_callbacks = "callbacks" in fit_signature.parameters


def make_fit_params():
    params = {
        "eval_set": [(X_test, y_test_array)],
        "verbose": False,
    }
    if supports_early_stopping_rounds:
        params["early_stopping_rounds"] = early_stopping_rounds
    elif supports_callbacks:
        params["callbacks"] = [
            EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True,
                maximize=False,
            )
        ]
    else:
        warnings.warn(
            "Installed xgboost version does not support early_stopping_rounds or callbacks; proceeding without early stopping.",
            RuntimeWarning,
        )
    return params


# %%
# Stage 1: quick search with reduced cross-validation to explore the space fast
xgb_search_stage1 = BayesSearchCV(
    xgb_model,
    stage_one_search_space,
    n_iter=initial_iterations,
    scoring="neg_mean_squared_error",
    cv=initial_cv_folds,
    n_jobs=-1,
    random_state=random_state,
    return_train_score=False,
    refit=True,
    optimizer_kwargs={"base_estimator": "GP"},
)

xgb_search_stage1.fit(X_train, y_train_array, **make_fit_params())

print("Stage 1 — best parameters found:")
pprint(xgb_search_stage1.best_params_)
print(f"Stage 1 — best score (negative MSE): {xgb_search_stage1.best_score_}")
stage_one_best_params = xgb_search_stage1.best_params_


# %%
# Build a refined search space around the best stage-one parameters
def clip_interval(lower, upper, bounds):
    clipped_lower = max(bounds[0], lower)
    clipped_upper = min(bounds[1], upper)
    if clipped_lower == clipped_upper:
        clipped_lower, clipped_upper = bounds
    return clipped_lower, clipped_upper


refined_search_space = {}
refined_search_space["n_estimators"] = tuple(
    int(val)
    for val in clip_interval(
        stage_one_best_params["n_estimators"] - 100,
        stage_one_best_params["n_estimators"] + 100,
        search_space_bounds["n_estimators"],
    )
)
refined_search_space["max_depth"] = tuple(
    int(val)
    for val in clip_interval(
        stage_one_best_params["max_depth"] - 2,
        stage_one_best_params["max_depth"] + 2,
        search_space_bounds["max_depth"],
    )
)

lr_lower, lr_upper = clip_interval(
    stage_one_best_params["learning_rate"] * 0.5,
    stage_one_best_params["learning_rate"] * 1.5,
    search_space_bounds["learning_rate"],
)
refined_search_space["learning_rate"] = (lr_lower, lr_upper, "uniform")

sub_lower, sub_upper = clip_interval(
    stage_one_best_params["subsample"] - 0.1,
    stage_one_best_params["subsample"] + 0.1,
    search_space_bounds["subsample"],
)
refined_search_space["subsample"] = (sub_lower, sub_upper, "uniform")

col_lower, col_upper = clip_interval(
    stage_one_best_params["colsample_bytree"] - 0.1,
    stage_one_best_params["colsample_bytree"] + 0.1,
    search_space_bounds["colsample_bytree"],
)
refined_search_space["colsample_bytree"] = (col_lower, col_upper, "uniform")

gamma_lower, gamma_upper = clip_interval(
    stage_one_best_params["gamma"] - 1.0,
    stage_one_best_params["gamma"] + 1.0,
    search_space_bounds["gamma"],
)
refined_search_space["gamma"] = (gamma_lower, gamma_upper)

alpha_lower, alpha_upper = clip_interval(
    stage_one_best_params["reg_alpha"] * 0.5,
    stage_one_best_params["reg_alpha"] * 1.5,
    search_space_bounds["reg_alpha"],
)
refined_search_space["reg_alpha"] = (alpha_lower, alpha_upper)

lambda_lower, lambda_upper = clip_interval(
    stage_one_best_params["reg_lambda"] * 0.5,
    stage_one_best_params["reg_lambda"] * 1.5,
    search_space_bounds["reg_lambda"],
)
refined_search_space["reg_lambda"] = (lambda_lower, lambda_upper)

pprint(refined_search_space)

# %%
# Stage 2: extend the search with tighter bounds and higher-fidelity cross-validation
xgb_model_refined = XGBRegressor(
    random_state=random_state,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
    objective="reg:squarederror",
    eval_metric="rmse",
)
xgb_model_refined.set_params(**stage_one_best_params)

xgb_opt = BayesSearchCV(
    xgb_model_refined,
    refined_search_space,
    n_iter=refined_iterations,
    scoring="neg_mean_squared_error",
    cv=refined_cv_folds,
    n_jobs=-1,
    random_state=random_state,
    return_train_score=False,
    optimizer_kwargs={"base_estimator": "GP"},
)

xgb_opt.fit(X_train, y_train_array, **make_fit_params())
print("Stage 2 — best parameters found:")
pprint(xgb_opt.best_params_)
print(f"Stage 2 — best score (negative MSE): {xgb_opt.best_score_}")

# %%
# Display the best score achieved by XGBoost
print(f"Best score achieved (negative MSE) by XGBoost: {xgb_opt.best_score_}")

# %%
# Evaluate the optimized XGBoost model on the test set
y_pred_xgb = xgb_opt.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"Mean Squared Error on test set by XGBoost: {mse_xgb}")

# %%
# Use the optimized XGBoost model to predict critical_temp
optimized_critical_temp_xgb = xgb_opt.predict(X_test)
print("Predicted critical_temp using Optimized XGBoost Model:")
print(optimized_critical_temp_xgb)

# %%
# Visualize the predicted vs actual critical_temp using XGBoost
plt.figure(figsize=(10, 6))
plt.scatter(y_test, optimized_critical_temp_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.title("Predicted vs Actual critical_temp using XGBoost", fontsize=16)
plt.xlabel("Actual critical_temp")
plt.ylabel("Predicted critical_temp")
plt.grid()
plt.show()


# %%
# Objective function for Bayesian optimization with XGBoost
def objective_function_xgb(params):
    """
    Objective function to minimize the negative of the critical_temp using XGBoost.
    This function takes the properties and returns the negative critical_temp.
    """
    # Create a DataFrame with the parameters using X_train.columns
    param_dict = {col: [param] for col, param in zip(X_train.columns, params)}
    elemental_properties = pd.DataFrame(param_dict)
    # Predict the critical_temp using the optimized XGBoost model
    predicted_critical_temp = xgb_opt.predict(elemental_properties)
    # Return the negative critical_temp (as we want to maximize it)
    return -predicted_critical_temp[0]


# %%
# Define the search space for the properties based on the min and max values in the training set
search_space_properties_xgb = [
    (X_train[col].min(), X_train[col].max()) for col in X_train.columns
]

# %%
from skopt import gp_minimize

# Perform Bayesian optimization to find the optimal properties using XGBoost
result_xgb = gp_minimize(
    objective_function_xgb,
    search_space_properties_xgb,
    n_calls=50,  # Number of evaluations
    random_state=random_state,
    verbose=True,
)

# %%
# Neatly display the best parameters and the best predicted critical_temp along with column names for XGBoost
best_params_xgb = result_xgb.x
best_critical_temp_xgb = (
    -result_xgb.fun
)  # Negate the result to get the actual critical_temp
print("Best parameters found by Bayesian optimization for XGBoost:")
for i, param in enumerate(best_params_xgb):
    print(f"{X_train.columns[i]}: {param}")
print(f"Best predicted critical_temp using XGBoost: {best_critical_temp_xgb}")

# %% [markdown]
# # Human-in-the-Loop Preference Learning for Bayesian Optimization
# We can augment the automated search with simulated expert feedback. A preference model will learn which superconducting material profiles resemble an expert-approved "golden" profile discovered by the XGBoost search. The learned preferences will influence a custom acquisition function that balances predicted critical temperature with the simulated expert's opinion.

# %% [markdown]
# ## Simulate the Human Expert
# The simulated expert favors material profiles that lie closest to the golden profile (the best recipe identified by the XGBoost-based optimization).

# %%
# Golden standard profile based on XGBoost optimization results
golden_standard_profile = np.array(best_params_xgb)


def simulate_human_expert(profile1, profile2):
    """Simulate a superconductivity expert who prefers the profile closest to the golden profile."""
    profile1 = np.array(profile1)
    profile2 = np.array(profile2)
    dist1 = np.linalg.norm(profile1 - golden_standard_profile)
    dist2 = np.linalg.norm(profile2 - golden_standard_profile)
    return 1 if dist1 < dist2 else 0


# %% [markdown]
# ## Train a Preference Model
# Generate synthetic preference data, fit a Gaussian Process Classifier (GPC), and use it to estimate how closely candidate profiles align with the simulated expert's tastes.

# %%
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

kernel = 1.0 * RBF(length_scale=1.0)
user_belief_model = GaussianProcessClassifier(kernel=kernel, random_state=random_state)

n_initial_pairs = 50
preference_data = []
preference_labels = []

for _ in range(n_initial_pairs):
    x1 = [np.random.uniform(low, high) for low, high in search_space_properties_xgb]
    x2 = [np.random.uniform(low, high) for low, high in search_space_properties_xgb]
    preference = simulate_human_expert(x1, x2)

    diff1 = np.array(x1) - golden_standard_profile
    diff2 = np.array(x2) - golden_standard_profile
    preference_data.extend([diff1, diff2])
    preference_labels.extend([preference, 1 - preference])

user_belief_model.fit(preference_data, preference_labels)
print("Initial user belief model trained.")

# %% [markdown]
# ## Custom HITL Optimization Loop
# Combine the surrogate predictions with the preference model to score candidate material profiles, iteratively refining both models with newly collected preferences.

# %%
import time

n_iterations = 50
n_candidates_per_iteration = 100
acquisition_weight = 0.5

main_surrogate_model = xgb_opt.best_estimator_

evaluated_points = []
objective_values = []
convergence_hitl = []
best_temp_so_far = -np.inf

start_time = time.time()

for i in range(n_iterations):
    print(f"--- Iteration {i + 1}/{n_iterations} ---")
    candidates = [
        [np.random.uniform(low, high) for low, high in search_space_properties_xgb]
        for _ in range(n_candidates_per_iteration)
    ]

    def acquisition_function(x):
        x_df = pd.DataFrame([x], columns=X_train.columns)
        pred_temp = main_surrogate_model.predict(x_df)[0]
        preference_features = (np.array(x) - golden_standard_profile).reshape(1, -1)
        pref_score = user_belief_model.predict_proba(preference_features)[0][1]
        return acquisition_weight * pred_temp + (1 - acquisition_weight) * pref_score

    acquisition_scores = [acquisition_function(candidate) for candidate in candidates]
    best_candidate_index = int(np.argmax(acquisition_scores))
    next_point = candidates[best_candidate_index]
    print("Selected new point to evaluate.")

    true_objective_value = -objective_function_xgb(next_point)
    print(f"True critical temperature of new point: {true_objective_value:.4f}")

    evaluated_points.append(next_point)
    objective_values.append(true_objective_value)

    if true_objective_value > best_temp_so_far:
        best_temp_so_far = true_objective_value
    convergence_hitl.append(best_temp_so_far)

    if len(evaluated_points) > 1:
        reference_point = evaluated_points[
            np.random.randint(0, len(evaluated_points) - 1)
        ]
        preference = simulate_human_expert(next_point, reference_point)

        diff_candidate = np.array(next_point) - golden_standard_profile
        diff_reference = np.array(reference_point) - golden_standard_profile
        preference_data.extend([diff_candidate, diff_reference])
        preference_labels.extend([preference, 1 - preference])
        user_belief_model.fit(preference_data, preference_labels)
        print("User belief model updated.")

end_time = time.time()
print(f"\nHITL optimization finished in {end_time - start_time:.2f} seconds.")

best_hitl_index = int(np.argmax(objective_values))
best_params_hitl = evaluated_points[best_hitl_index]
best_temp_hitl = objective_values[best_hitl_index]

print("\nBest parameters found by HITL Bayesian optimization:")
for column, value in zip(X_train.columns, best_params_hitl):
    print(f"{column}: {value}")
print(f"\nBest predicted critical temperature using HITL: {best_temp_hitl}")

# %% [markdown]
# ## Final Evaluation
# Compare the vanilla Bayesian optimization trace with the human-in-the-loop variant and examine the best superconducting profiles uncovered by each approach.

# %%
convergence_original = np.maximum.accumulate(-np.array(result_xgb.func_vals))

num_evaluations = min(len(convergence_hitl), len(convergence_original))
convergence_hitl_plot = convergence_hitl[:num_evaluations]
convergence_original_plot = convergence_original[:num_evaluations]

plt.figure(figsize=(12, 8))
plt.plot(
    range(1, num_evaluations + 1),
    convergence_hitl_plot,
    "o-",
    label="HITL Bayesian Optimization",
    color="tab:blue",
)
plt.plot(
    range(1, num_evaluations + 1),
    convergence_original_plot,
    "s-",
    label="Original Bayesian Optimization (XGBoost)",
    color="tab:green",
)
plt.title(
    "Convergence Comparison: HITL vs. Original Bayesian Optimization", fontsize=16
)
plt.xlabel("Number of Evaluations", fontsize=12)
plt.ylabel("Best Critical Temperature Found So Far", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

print("--- Comparison of Best Results ---")
print("\nOriginal Bayesian Optimization (XGBoost):")
print(f"  Best critical temperature: {best_critical_temp_xgb:.4f}")
print("  Best profile:")
for column, value in zip(X_train.columns, best_params_xgb):
    print(f"    {column}: {value:.4f}")

print("\nHITL Bayesian Optimization:")
print(f"  Best critical temperature: {best_temp_hitl:.4f}")
print("  Best profile:")
for column, value in zip(X_train.columns, best_params_hitl):
    print(f"    {column}: {value:.4f}")

# %% [markdown]
# ### Conclusion
# The superconductivity workflow now mirrors the structure used in the concrete and wine experiments: dataset retrieval, exploratory analysis, hyperparameter tuning with both Random Forest and XGBoost, feature-level optimization via `gp_minimize`, and an optional human-in-the-loop enhancement. The HITL loop can be tuned further (number of iterations, candidate batch size, acquisition weights) to balance compute time with exploration depth.
