# home-prediction-mlapp

## Table of Contents

- [Initial Setup](#initial-setup)
- [Steps](#steps)
    - [Build Services](#build-services)
    - [Expanding more Models](#expanding-more-models)    
- [Known Bugs](#known-bugs)    

## Initial Setup:
Follow the original steps in creating the `housing` environment from the instructions in conda.  As a reminder `conda create env -f conda_environment.yaml -y` then run `conda activate housing && python setup/create_model.py`


## Steps: 

### Build Services

```docker compose up -d```

Go ahead and try the post on the original model we created from the conda housing environment locally
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features" : {
    "bedrooms": 4,
    "bathrooms": 1.0,
    "sqft_living": 1680,
    "sqft_lot": 5043,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 6,
    "sqft_above": 1680,
    "sqft_basement": 0,
    "yr_built": 1911,
    "yr_renovated": 0,
    "zipcode": 98118,
    "lat": 47.5354,
    "long": -122.273,
    "sqft_living15": 1560,
    "sqft_lot15": 5765
  }}'
```

*BONUS* Before diving into the next section we will quickly walk through an iteration on strict predictions predict.  As I understood it there are required features we'd expect strictly for the model, where as in the usual predict implementation the features are collected around housing input, and then demo data is added on afterwards.  In this case we'll use whats called `predict-strict`. to enforce the model variables be provided in the feature request itself rather than collecting from the sql database.

In order to show this we've created the test: test_fastapi.py

```
> python test/test_fastapi.py
...
=== Testing /predict-strict Endpoint ===
Model requires 41 features
Sending 41 features to strict endpoint
Status Code: 200
Prediction: $384,283.69
Model Version: v20250903_005614
Metadata: {'enriched': False}
...
```

## Expanding more models

Next we'll update another model to play around with.  This should show a folder called `model_artifacts` that appears which would contain all the necessary models we would need. 

**Training**: Here we train a new model using XGBoost, which you can find under ./training/train_xgboost.py for more information

``` docker compose --profile training run trainer python main.py --train```

**Evaluation**

``` docker compose --profile training run trainer python main.py --evaluate --model-version original```

Output: 
```
Original Model Evaluation: 
Training RMSE: $143,467
Test RMSE: $201,671
Test R²: 0.728
Model is OVERFITTING - training error much lower than test error
```

Look in model_artifacts for model Version i.e V[%Y%m%d_%H%M%S]

Note:  You may see overfitting errors, as the trainer is currently built to mimic a data scientist running 

``` docker compose --profile training run trainer python main.py --evaluate --model-version [model_version]```

Output: 
```
Original Model Evaluation: 
Training RMSE: $143,467
Test RMSE: $201,671
Test R²: 0.728
Model is OVERFITTING - training error much lower than test error
```

``` docker compose --profile training run trainer python main.py --evaluate --model-version v20250902_171645```

Output
```
Loading model from output/v20250902_171645/model.pkl...
Training RMSE: $97,694
Test RMSE: $104,207
Test R²: 0.927
Model fit is APPROPRIATE - balanced training/test performance
```

## Known Bugs

## Hot Reloads Default
By default, any time the fastapi service is started it will default to the latest model that was trained (or original if none).  This in theory may always be the latest and greatest model, but in theory there are cases where a rollback would be overly complex.

*Quick Fix*: Updating a file that symlinks, or copies the full model to a main folder when /model/reload is called.  

## Permissions and position of /model/reload
When wanting to point to a new model, we've decided between different points of when a model should be reloaded.  While functionally, /model/reload makes sense to have the server update the model it (1) introduces complexity for permission level and (2) makes the fastapi server the conductor of the ML Model repository which should be handled by a separate service.

*Quick fix*:  Implement a separate fastapi server, or leverage MLflow  to be the manager of the model repo.  This would lend much more control and can still leverage the model/reload/ to acquire am explicit version

*Feature Request*: There are several cases in where multiple model versions would be necessary. Implementing a new feature in which multiple versions can be hosted  would prove valuable.
