# W11D2 - MLflow Model Registry

## Objective

Run multiple Ridge Regression experiments with different hyperparameters, compare their performance in MLflow, register the best-performing model, and serve the registered model through the MLflow Model Server.

## Experiments

Five Ridge Regression experiments were recorded in MLflow using different alpha values.

The experiments logged:

- Alpha hyperparameter
- Mean Squared Error (MSE)
- R² score
- Model artifacts

The best model was selected based on the lowest Mean Squared Error.

## Model Registry

The selected model was registered as:

`W11D2_Ridge_Best_Model`

Registered version:

`Version 1`

## Model Serving

The registered model was served using the MLflow Model Server on port `5001`.

The `/invocations` endpoint was tested using an HTTP POST request.

Example prediction:

`196.3759743666677`

## Workflow

MLflow Experiments  
↓  
Compare Metrics  
↓  
Select Best Ridge Model  
↓  
Register Model  
↓  
Model Version 1  
↓  
MLflow Model Server  
↓  
HTTP Prediction Request