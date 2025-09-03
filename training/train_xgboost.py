#!/usr/bin/env python3
"""
XGBoost training pipeline for house price prediction model.
Designed to run in a separate container from the inference server.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import json
import pickle
import argparse
import loguru
import shutil
import requests
from datetime import datetime


def load_and_prepare_data(data_path: str, demographics_data_path: str):
    """Load and prepare training data"""
    loguru.logger.info(f"Loading data from {data_path}...")
    
    # Load the house sales data
    data = pd.read_csv(data_path)
    demographics = pd.read_csv(demographics_data_path)
    df = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    loguru.logger.info(f"Loaded {len(df)} records")
    
    # Feature engineering
    loguru.logger.info("Performing feature engineering...")
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    y = df.pop('price')
    # Filter to available columns
    available_features = [col for col in df.columns if col not in ('price', 'id', 'date', 'lat', 'long')]

    loguru.logger.info(f"Using {len(available_features)} features: {available_features}")
    X = df[available_features]
    return X, y, available_features


def train_model(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model with given parameters"""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
    
    loguru.logger.info("Training XGBoost model...")
    loguru.logger.info(f"Training set size: {len(X_train)}")
    loguru.logger.info(f"Validation set size: {len(X_val)}")
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],                
        verbose=False
    )
    return model


def evaluate_model(model, X, y, dataset: str = 'test'):
    """Evaluate model and return metrics"""
    if dataset not in ['train', 'test','validation']:
        raise ValueError("dataset must be either 'train' ,'validation', or 'test'")
    
    loguru.logger.info(f"Evaluating model on {dataset} set...")
    
    y_pred = model.predict(X)
    
    metrics = {
        f'{dataset}_rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        f'{dataset}_mae': float(mean_absolute_error(y, y_pred)),
        f'{dataset}_r2': float(r2_score(y, y_pred))
    }
    
    loguru.logger.info(f"{dataset.capitalize()} RMSE: ${metrics[f'{dataset}_rmse']:,.2f}")
    loguru.logger.info(f"{dataset.capitalize()} MAE: ${metrics[f'{dataset}_mae']:,.2f}")
    loguru.logger.info(f"{dataset.capitalize()} R²: {metrics[f'{dataset}_r2']:.4f}")
    
    return metrics

def main(data_path:str = '/app/data/kc_house_data.csv'
               , demographics_path:str = '/app/data/zipcode_demographics.csv'
               , output_dir:str = '/app/output'
               ):

    
    try:
        # Check if data files exist
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        if not os.path.exists(demographics_path):
            raise FileNotFoundError(f"Demographics data not found: {demographics_path}")
        
        # Load and prepare data
        X, y, feature_names = load_and_prepare_data(data_path, demographics_path)
        
        # Train/validation/test split (60/20/20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2
            # , random_state=42 # Intentionally commented out for variability
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25
            # , random_state=42 # Intentionally commented out for variability
        )
        
        # Model parameters
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val, params)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        train_metrics = evaluate_model(model, X_train, y_train, dataset='train')
        metrics.update(train_metrics)

        # Create version-specific directory
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = os.path.join(output_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model in version directory
        model_path = os.path.join(version_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        loguru.logger.info(f"Model saved to: {model_path}")
        
        # Save features in version directory
        features_path = os.path.join(version_dir, "model_features.json")
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        loguru.logger.info(f"Features saved to: {features_path}")
        
        # Also create/update symlinks for "latest" model in root output directory  
        latest_model_path = os.path.join(output_dir, "model.pkl")
        latest_features_path = os.path.join(output_dir, "model_features.json")
        latest_metadata_path = os.path.join(output_dir, "model_metadata.json")
        
        
    
        # Save metadata
        metadata = {
            "model_type": "XGBRegressor",
            "features": feature_names,
            "feature_count": len(feature_names),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "parameters": params,
            "metrics": metrics,
            "version": version
        }
        
        metadata_path = os.path.join(version_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        loguru.logger.info(f"Metadata saved to: {metadata_path}")
        
        loguru.logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        loguru.logger.info(f"Model version: {metadata['version']}")
        loguru.logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        loguru.logger.info(f"Features: {len(feature_names)}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # loguru.logger.info("Top 10 Feature Importances:")
            # for idx, row in feature_importance.head(10).iterrows():
            #     loguru.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
    except Exception as e:
        loguru.logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train XGBoost house price prediction model')
    parser.add_argument('--data-path', default='/app/data/kc_house_data.csv', help='Path to training data CSV')
    parser.add_argument('--demographics-path', default='/app/data/zipcode_demographics.csv', help='Path to demographics data CSV')
    parser.add_argument('--output-dir', default='/app/output', help='Output directory for model files')
    parser.add_argument('--upload-model', action='store_true', help='Whether to upload the model after training')
    args = parser.parse_args()

    exit(main(args.data_path, args.demographics_path, args.output_dir  ))