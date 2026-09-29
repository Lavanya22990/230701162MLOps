import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# 1. Load the CSV file
df = pd.read_csv('Iris.csv')

# 2. Separate features and target
X = df[['SepalLengthCm', 'SepalWidthCm']]
y = df['PetalWidthCm']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        random_state=42
    )
}

# 5. Start MLflow experiment
mlflow.set_experiment("Iris Regression Comparison")

results = []

# 6. Train and track each model
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print results
        print("\nModel:", name)
        print("R-squared Score:", r2)
        print("Mean Squared Error:", mse)

        # Log parameters
        mlflow.log_param("model", name)

        # Log metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)

        # Log model
        if name == "XGBoost":
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

        # Store results
        results.append({
            "Model": name,
            "R2": r2,
            "MSE": mse
        })


# 7. Compare models
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df)

# 8. Find best model based on R2
best_model = results_df.loc[
    results_df["R2"].idxmax()
]

print("\nBest Model:")
print(best_model["Model"])

print("Best R2:", best_model["R2"])
print("Best MSE:", best_model["MSE"])