import pandas as pd
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
import requests


def evaluate_model(model_version=None):
    """Evaluate the model on a test set and print metrics"""
    if model_version=='original':
        model_path = "model/original/model.pkl"
    else:
        model_path = f"output/{model_version}/model.pkl" 

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        feature_names=model.feature_names_in_
    
    print(f"Loading model from {model_path}...")
    # Load and prepare data
    # sales_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
    #                 'floors', 'sqft_above', 'sqft_basement', 'zipcode']
    data = pd.read_csv("data/kc_house_data.csv", dtype={'zipcode': str})
    demographics = pd.read_csv("data/zipcode_demographics.csv", dtype={'zipcode': str})
    merged = data.merge(demographics, how="left", on="zipcode")
    y = merged.pop('price')
    X = merged[feature_names] 

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)


    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    test_r2 = r2_score(y_test, model.predict(X_test))

    print(f"Training RMSE: ${train_rmse:,.0f}")
    print(f"Test RMSE: ${test_rmse:,.0f}")
    print(f"Test R²: {test_r2:.3f}")

    # Check fit
    ratio = train_rmse / test_rmse
    if ratio < 0.9:
        print("Model is OVERFITTING - training error much lower than test error")
    elif ratio > 1.1:
        print("Model is UNDERFITTING - test error lower than training error")
    else:
        print("Model fit is APPROPRIATE - balanced training/test performance")

