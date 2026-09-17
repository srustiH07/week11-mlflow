# W11D3 - MLflow Model Serving & REST API

# This script:
# 1. Loads the sklearn Diabetes dataset.
# 2. Runs 5 Random Forest experiments.
# 3. Logs parameters, metrics, and model artifacts to MLflow.
# 4. Selects the best model using the lowest MSE.
# 5. Registers the best model in MLflow Model Registry.
# 6. Prints the registered model information.

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. MLflow configuration
# ---------------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "W11D3_MLflow_Model_Serving_REST_API"
REGISTERED_MODEL_NAME = "W11D3_RandomForest_Best_Model"

mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# 2. Load sklearn Diabetes dataset
# ---------------------------------------------------------

data = load_diabetes()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset loaded successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ---------------------------------------------------------
# 3. Five Random Forest experiments
# ---------------------------------------------------------

experiments = [
    {
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 2
    },
    {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2
    },
    {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
    },
    {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 4
    },
    {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 4
    }
]


# ---------------------------------------------------------
# 4. Run experiments and track them in MLflow
# ---------------------------------------------------------

best_run_id = None
best_mse = float("inf")
best_r2 = None
best_parameters = None

print("\nStarting 5 MLflow experiments...\n")

for index, params in enumerate(experiments, start=1):

    with mlflow.start_run(
        run_name=f"W11D3_RF_Experiment_{index}"
    ):

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log hyperparameters
        mlflow.log_param(
            "n_estimators",
            params["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            params["max_depth"]
        )

        mlflow.log_param(
            "min_samples_split",
            params["min_samples_split"]
        )

        # Log evaluation metrics
        mlflow.log_metric(
            "mean_squared_error",
            mse
        )

        mlflow.log_metric(
            "r2_score",
            r2
        )

        # Create model signature
        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # Log sklearn model artifact
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            signature=signature
        )

        current_run_id = mlflow.active_run().info.run_id

        print(f"Experiment {index}")
        print(f"Run ID: {current_run_id}")
        print(f"n_estimators: {params['n_estimators']}")
        print(f"max_depth: {params['max_depth']}")
        print(f"min_samples_split: {params['min_samples_split']}")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        print("-" * 60)

        # Select the model with the lowest MSE
        if mse < best_mse:
            best_mse = mse
            best_r2 = r2
            best_run_id = current_run_id
            best_parameters = params.copy()


# ---------------------------------------------------------
# 5. Display best experiment
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Best Run ID: {best_run_id}")
print(f"Best n_estimators: {best_parameters['n_estimators']}")
print(f"Best max_depth: {best_parameters['max_depth']}")
print(
    f"Best min_samples_split: "
    f"{best_parameters['min_samples_split']}"
)
print(f"Best MSE: {best_mse:.4f}")
print(f"Best R2: {best_r2:.4f}")


# ---------------------------------------------------------
# 6. Register the best model
# ---------------------------------------------------------

best_model_uri = f"runs:/{best_run_id}/random_forest_model"

print("\nRegistering the best model...")
print(f"Model URI: {best_model_uri}")

registration = mlflow.register_model(
    model_uri=best_model_uri,
    name=REGISTERED_MODEL_NAME
)

print("\n" + "=" * 60)
print("MODEL REGISTRATION SUCCESSFUL")
print("=" * 60)

print(f"Model Name: {REGISTERED_MODEL_NAME}")
print(f"Model Version: {registration.version}")
print(f"Best Run ID: {best_run_id}")
print(f"Best MSE: {best_mse:.4f}")
print(f"Best R2: {best_r2:.4f}")

print("\nW11D3 completed successfully.")
print("Open MLflow UI to compare the 5 experiments.")