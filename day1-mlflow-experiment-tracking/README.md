# Week 11 Day 1 - MLflow Experiment Tracking

## Objective

The objective of this task is to understand and implement experiment tracking using MLflow.

The machine learning model is trained using Linear Regression, and MLflow is used to track the experiment parameters, evaluation metrics, and the trained model.

## Technologies Used

- Python
- Scikit-learn
- MLflow
- SKOPS

## Experiment Details

### Experiment Name

`W11D1_Experiment_Tracking`

### Model

`LinearRegression`

### Parameters

- `test_size = 0.2`
- `random_state = 42`

### Metrics

- `mean_squared_error ≈ 2900.1936`
- `r2_score ≈ 0.4526`

### Logged Model

`linear_regression_model`

## MLflow Run

The experiment was successfully executed and tracked using MLflow.

The MLflow dashboard showed:

- Experiment: `W11D1_Experiment_Tracking`
- Run: `capable-crow-105`
- Status: Finished
- Run ID: `4aa86c20b83a42bf809089d72fa943e1`

## Project Files

```text
day1-mlflow-experiment-tracking/
│
├── mlflow_experiment.py
├── requirements.txt
├── README.md
├── mlflow.db
└── mlruns/
## Learning Outcome

Through this experiment, I learned how to:

1. Create and organize an MLflow experiment.
2. Track model parameters during training.
3. Log evaluation metrics for model performance.
4. Register and log the trained machine learning model.
5. Verify experiment runs through the MLflow dashboard.