import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
data = load_diabetes()

X = data.data
y = data.target


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create MLflow experiment
mlflow.set_experiment("W11D1_Experiment_Tracking")


# Start an MLflow run
with mlflow.start_run() as run:

    # Model parameters
    model_name = "LinearRegression"
    test_size = 0.2
    random_state = 42

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("model", model_name)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    # Log metrics
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    # Log trained model
    mlflow.sklearn.log_model(
        model,
        "linear_regression_model"
    )

    # Display run information
    print("MLflow experiment completed successfully.")
    print(f"Run ID: {run.info.run_id}")
    print(f"Model: {model_name}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")