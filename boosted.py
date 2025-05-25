# xgboost_pipeline_model.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

# --- Load and prepare data ---
df = pd.read_csv("cleaned_property_data.csv")
df.drop(columns=["Id"], inplace=True, errors="ignore")

# Optional: Feature Engineering
if "Area" in df.columns:
    df["Price_per_sqft"] = df["Price"] / df["Area"]

# Define features and target
target = "Price"
X = df.drop(columns=[target])
y = df[target]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Save feature columns
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Create Pipeline with log-transform of target ---
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

pipeline = TransformedTargetRegressor(regressor=model, transformer=log_transformer)

# --- Train model ---
pipeline.fit(X_train, y_train)

# --- Evaluate ---
def evaluate(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "R2": round(r2_score(y_test, y_pred), 4)
    }

print("\nEvaluation on Test Set:")
print(evaluate(pipeline, X_test, y_test, "XGBoost + Log Target"))

# --- Cross-validation ---
scoring = {
    'R2': 'r2',
    'Neg_RMSE': 'neg_root_mean_squared_error',
    'Neg_MAE': 'neg_mean_absolute_error'
}

def run_cv(model, name="Model"):
    print(f"\nCross-Validation for {name}:")
    for metric, score in scoring.items():
        score_val = cross_val_score(model, X, y, cv=5, scoring=score).mean()
        print(f"{metric}: {round(score_val, 4)}")

run_cv(pipeline, "XGBoost + Log Target")

# --- Hyperparameter tuning ---
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [4, 6],
    'regressor__learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_

print("\nTuned XGBoost Results:")
print(evaluate(best_pipeline, X_test, y_test, "Tuned XGBoost"))
print("Best Parameters:", grid.best_params_)

# --- Save final model ---
with open("model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)
print("Model saved as model.pkl")

# --- Feature Importance ---
xgb_model = best_pipeline.regressor_
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features, y=top_features.index)
plt.title("Top 10 Important Features (XGBoost)")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()
