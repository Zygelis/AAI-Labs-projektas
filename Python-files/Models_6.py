import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from Data_preparation_6p import *

# Read data
df = pd.read_excel("C:\\Users\\Zygis\\Desktop\\test\\WEOOct2020all.xls", engine="xlrd")

# Prepare data
df_X, df_y = data_preparation(df, all_features)

# Split data
# X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)


# Define 3 models:
# XGBoost model
def XGB_model_fit(X_train, y_train, X_test, y_test):
    # Train model
    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state=0
    )
    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=10,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Get predictions
    predictions = model.predict(X_test)

    return model


# LightGBM model
def LGBM_model_fit(X_train, y_train):
    # Train model
    model = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state=0
    )
    model.fit(X_train, y_train)

    # Get predictions
    predictions = model.predict(X_train)

    return model


# Random Forest model
def RF_model_fit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Select categorical columns with low unique values
    categorical_cols = [
        cname
        for cname in X_train.columns
        if X_train[cname].nunique() < 10 and X_train[cname].dtype == "object"
    ]

    # Select numerical columns
    numerical_cols = [
        cname
        for cname in X_train.columns
        if X_train[cname].dtype in ["int64", "float64"]
    ]

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train[my_cols].copy()
    X_test = X_test[my_cols].copy()

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy="mean")

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Create one preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define model
    model = RandomForestRegressor(n_estimators=900, random_state=0, n_jobs=4)

    # Create pipeline with preprocessor and ML model
    RF_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    RF_pipe.fit(X_train, y_train)

    # Get predictions
    prediction = RF_pipe.predict(X_test)

    return RF_pipe


# Fit and define models
model_xgb = XGB_model_fit(X_train, y_train, X_test, y_test)
model_lgbm = LGBM_model_fit(X_train, y_train)
model_rf = RF_model_fit(df_X, df_y)


# Model quality evaluation
def perform_cross_validation(model, X, y, num_folds=5):
    # Initialize KFold cross-validator
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    # Perform cross-validation and calculate MAE and MSE scores
    scores_mae = -1 * cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_absolute_error"
    )  # Multiply by -1 to Convert back to positive

    scores_mse = -1 * cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_squared_error"
    )  # Multiply by -1 to Convert back to positive4

    # Calculate the mean of MAE and MSE scores
    mean_mae = scores_mae.mean()

    mean_mse = scores_mse.mean()

    return mean_mae, mean_mse


# Compare models and select the best one
def compare_models(X, y):
    models = {"XGB": model_xgb, "LGBM": model_lgbm, "RF": model_rf}

    results_mae = {}
    results_mse = {}

    for name, model in models.items():
        mean_mae, mean_mse = perform_cross_validation(model, X, y)
        results_mae[name] = mean_mae
        results_mse[name] = mean_mse

    best_model_mae = min(results_mae, key=results_mae.get)
    best_model_mse = min(results_mse, key=results_mse.get)

    print("\n" * 2)
    print("Model comparison results:")

    for name, mae_score in results_mae.items():
        mse_score = results_mse[name]
        print(f"{name}: MAE = {mae_score:.4f}, MSE = {mse_score:.4f}")

    print("\n")
    print(
        f"Best model based on MAE: {best_model_mae} with MAE = {results_mae[best_model_mae]:.4f}"
    )
    print(
        f"Best model based on MSE: {best_model_mse} with MSE = {results_mse[best_model_mse]:.4f}"
    )

    return models[best_model_mse]

print("\n")
print("Models_6.py done")
