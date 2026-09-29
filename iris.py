import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


# Create experiment
mlflow.set_experiment("iris.py")


# Start MLflow run
with mlflow.start_run():

    # Parameter
    fit_intercept = False

    df = pd.read_csv('Iris.csv')

    X = df[['SepalLengthCm', 'SepalWidthCm']]
    y = df['PetalWidthCm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 6. Evaluate the model performance
    print("Model Coefficients (Slopes for Feature1, Feature2):", model.coef_)
    print("Model Intercept (Bias):", model.intercept_)
    print("R-squared Score:", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


# 7. Predict a completely new value
# Example: Feature1 = 18, Feature2 = 32

    new_data = pd.DataFrame([[18, 32]], columns=['SepalLengthCm', 'SepalWidthCm'])
    prediction = model.predict(new_data)
    print(f"Prediction for new data: {prediction[0]}")


    # Log parameter
    mlflow.log_param(
        "fit_intercept",
        fit_intercept
    )

    # Log metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("r2_score", r2_score(y_test, y_pred))

    # Log model
    mlflow.sklearn.log_model(
        model,
        name="model"
    )



