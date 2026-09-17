# W11D3 - MLflow Model Serving & REST API

## Objective

The objective of W11D3 is to track multiple machine learning experiments using MLflow, register the best-performing model in the MLflow Model Registry, serve the registered model, and test it through a REST API.

## Tools and Technologies

- Python
- Scikit-learn
- MLflow
- Random Forest Regressor
- MLflow Model Registry
- REST API
- Uvicorn

## Dataset

The Scikit-learn Diabetes dataset was used for the experiments.

The dataset was divided into:

- 80% training data
- 20% testing data

A fixed random state of 42 was used for reproducibility.

## Model

The machine learning model used was:

RandomForestRegressor

The experiments used different combinations of:

- n_estimators
- max_depth
- min_samples_split

## MLflow Experiment Tracking

The MLflow experiment name is:

W11D3_MLflow_Model_Serving_REST_API

Five Random Forest experiments were executed.

Each experiment logged:

- Model hyperparameters
- Mean Squared Error (MSE)
- R2 Score
- Trained model artifact
- Model signature

The best model was selected using the lowest Mean Squared Error.

## Model Registry

The best model was registered in MLflow Model Registry as:

W11D3_RandomForest_Best_Model

The registered model used Version 1.

The model URI was:

models:/W11D3_RandomForest_Best_Model/1

## Model Serving

The registered model was served locally using MLflow Model Serving.

Serving command:

mlflow models serve -m "models:/W11D3_RandomForest_Best_Model/1" -p 5002 --env-manager local

The model server was successfully started at:

http://127.0.0.1:5002

## REST API Testing

The `/invocations` endpoint was tested using PowerShell.

Example request:

Invoke-RestMethod -Uri "http://127.0.0.1:5002/invocations" -Method Post -ContentType "application/json" -Body '{"inputs":[[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]]}'

The REST API successfully returned a prediction:

220.81772161888645

## Workflow

The complete W11D3 workflow was:

Sklearn Dataset
        |
        v
Random Forest Model
        |
        v
5 MLflow Experiments
        |
        v
Compare MSE and R2
        |
        v
Select Best Model
        |
        v
MLflow Model Registry
        |
        v
Serve Registered Model
        |
        v
REST API
        |
        v
Prediction

## Learning Outcomes

After completing W11D3, the following concepts were practiced:

1. MLflow experiment tracking
2. Logging model parameters and metrics
3. Logging sklearn model artifacts
4. Comparing multiple experiments
5. Registering a machine learning model
6. Serving a registered MLflow model
7. Testing model predictions through a REST API
8. Using MLflow as part of a model deployment workflow