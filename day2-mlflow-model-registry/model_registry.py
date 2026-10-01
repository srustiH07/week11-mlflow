import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------
# W11D2 - MLflow Model Registry & Versioning
# ---------------------------------------------------------

# Load the Diabetes dataset
data = load_diabetes()

X = data.data
y = data.target


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Set the MLflow experiment
experiment_name = "W11D2_Model_Registry"
mlflow.set_experiment(experiment_name)


# Different hyperparameter values for five experiments
alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]

results = []


print("=" * 60)
print("W11D2 - MLflow Model Registry & Versioning")
print("=" * 60)


# ---------------------------------------------------------
# Run five experiments
# ---------------------------------------------------------

for alpha in alpha_values:

    with mlflow.start_run() as run:

        # Create Ridge regression model
        model = Ridge(alpha=alpha)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)

        # Log the trained model
        mlflow.sklearn.log_model(
            model,
            name="ridge_model"
        )

        # Store information about the run
        results.append(
            {
                "run_id": run.info.run_id,
                "alpha": alpha,
                "mse": mse,
                "r2": r2
            }
        )

        print()
        print(f"Experiment completed")
        print(f"Alpha: {alpha}")
        print(f"Run ID: {run.info.run_id}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")


# ---------------------------------------------------------
# Compare all five experiments
# ---------------------------------------------------------

best_result = min(results, key=lambda result: result["mse"])

print()
print("=" * 60)
print("EXPERIMENT COMPARISON")
print("=" * 60)

for result in results:
    print(
        f"Alpha={result['alpha']:<6} "
        f"MSE={result['mse']:<12.4f} "
        f"R2={result['r2']:.4f}"
    )


print()
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Best Alpha: {best_result['alpha']}")
print(f"Best Run ID: {best_result['run_id']}")
print(f"Best MSE: {best_result['mse']:.4f}")
print(f"Best R2 Score: {best_result['r2']:.4f}")


# ---------------------------------------------------------
# Register the best model
# ---------------------------------------------------------

model_name = "W11D2_Ridge_Best_Model"

model_uri = f"runs:/{best_result['run_id']}/ridge_model"

print()
print("=" * 60)
print("REGISTERING BEST MODEL")
print("=" * 60)

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"Registered Model: {model_name}")
print(f"Model Version: {registered_model.version}")


# ---------------------------------------------------------
# Load the registered model
# ---------------------------------------------------------

registered_model_uri = f"models:/{model_name}/{registered_model.version}"

loaded_model = mlflow.sklearn.load_model(
    registered_model_uri
)

# Test the registered model
sample_predictions = loaded_model.predict(X_test[:5])

print()
print("=" * 60)
print("REGISTERED MODEL LOADED SUCCESSFULLY")
print("=" * 60)

print(f"Model URI: {registered_model_uri}")
print("Sample predictions:")

for prediction in sample_predictions:
    print(f"{prediction:.4f}")


print()
print("=" * 60)
print("W11D2 COMPLETED SUCCESSFULLY")
print("=" * 60)