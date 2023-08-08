import joblib
from Data_preparation_6p import *
from Models_6p import *


# Determine the best model overall
# This usually takes a few minutes to run. The best model is LGBM.
#best_model = compare_models(df_X, df_y)
# If you want to run code faster, comment out the line above and uncomment the line below.
best_model = model_lgbm

# Best model prediction
y_test_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)

# Prediction error on the training and the testing data sets for the best model
print("\n")
print('A)')
print("Prediction error on the training and the testing data sets for the best model with all features:")

mae_test = mean_absolute_error(y_test, y_test_pred)
print("mae_test:", mae_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print("mse_test:", mse_test)
mae_train = mean_absolute_error(y_train, y_train_pred)
print("mae_train:", mae_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print("mse_train:", mse_train)


# Fields used in the model (all_features)
print("\n")
print('B)')
print("Fields used in the model:")
print(all_features)


# Top 5 fields/features that contribute the most to the predictions
mi_scores = make_mi_scores(df_X, df_y)
top_5_features = mi_scores.head(5)
print("\n")
print('C)')
print("Top 5 fields/features that contribute the most to the predictions:")
print(top_5_features)
print("\n")



# Make top 5 features codes a list
top_5_features = top_5_features.index.tolist()

# Add NGDPDPC to top_5_features as a target
top_5_features.append("NGDPDPC")

# Train another predictor that uses those top 5 features
# Prepare data
df_X_top5, df_y_top5 = data_preparation(df, top_5_features)

# Split data
X_train_top5, X_test_top5, y_train_top5, y_test_top5 = train_test_split(df_X_top5, df_y_top5, random_state=0)

# Train another predictor that uses those top 5 features
model_top5 = LGBM_model_fit(X_train_top5, y_train_top5)

# Best model prediction
y_test_pred_top5 = model_top5.predict(X_test_top5)
y_train_pred_top5 = model_top5.predict(X_train_top5)

# Prediction error on the training and the testing data sets
print("\n")
print('D)')
print("Prediction error on the training and the testing data sets with top 5 features")
mae_test_top5 = mean_absolute_error(y_test_top5, y_test_pred_top5)
print("mae_test:", mae_test_top5)
mse_test_top5 = mean_squared_error(y_test_top5, y_test_pred_top5)
print("mse_test:", mse_test_top5)
mae_train_top5 = mean_absolute_error(y_train_top5, y_train_pred_top5)
print("mae_train:", mae_train_top5)
mse_train_top5 = mean_squared_error(y_train_top5, y_train_pred_top5)
print("mse_train:", mse_train_top5)


# Save the model
print("\n")
print('E)')
print("Saved the model as:")
print("model_top5.joblib")
print("\n")

save_path = "6-dalis\\model_top5.joblib"
joblib.dump(model_top5, save_path)

# Load the model
model_loaded = joblib.load("6-dalis\\model_top5.joblib")


# Predictor with best MAE and MSE scores that I have found:
# Features used in the model
features_min_error = ['NGDPDPC', 'BCA', 'LP', 'LUR', 'GGXWDG', 'GGX', 'GGR', 'PPPEX', 'GGXCNL']

# Prepare data
df_X, df_y = data_preparation(df, features_min_error)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

# Train model
model_min_error = LGBM_model_fit(X_train, y_train)

# Predictions for test and train data
y_test_pred_min_error = model_min_error.predict(X_test)
y_train_pred_min_error = model_min_error.predict(X_train)

# Prediction error
mae_test_min_error = mean_absolute_error(y_test, y_test_pred_min_error)
mse_test_min_error = mean_squared_error(y_test, y_test_pred_min_error)
mae_train_min_error = mean_absolute_error(y_train, y_train_pred_min_error)
mse_train_min_error = mean_squared_error(y_train, y_train_pred_min_error)

print("\n")
print('F)')
print("Prediction error on the training and the testing data sets for the best model with minimum error:")
print("mae_test:", mae_test_min_error)
print("mse_test:", mse_test_min_error)
print("mae_train:", mae_train_min_error)
print("mse_train:", mse_train_min_error)
print("\n")
print("Fields used in the model with minimum error:")
print(features_min_error)