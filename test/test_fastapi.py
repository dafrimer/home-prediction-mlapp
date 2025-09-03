

import requests
import pandas as pd
import json
import sys
from typing import Dict, Any
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_endpoint(features: Dict[str, Any]):
    """Test the /predict endpoint with automatic enrichment"""
    print("\n=== Testing /predict Endpoint ===")
    print(f"Input features: {json.dumps(features, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=features)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: ${result['prediction']:,.2f}")
            print(f"Model Version: {result['model_version']}")
            print(f"Metadata: {result.get('metadata', {})}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_strict_endpoint(example_features: Dict[str, Any]):
    """Test the /predict-strict endpoint"""
    print("\n=== Testing /predict-strict Endpoint ===")
    
    # For strict endpoint, we need to provide ALL features including demographics
    # Get model info to see what features are required    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        if response.status_code != 200:
            print("Could not get model info")
            return False
        
        required_features = response.json()["features"]
        print(f"Model requires {len(required_features)} features")
        
        # Load demographics to enrich the features manually
        demographics_df = pd.read_csv('data/zipcode_demographics.csv')
        zipcode = int(example_features.get('zipcode', 0))
        demo_row = demographics_df[demographics_df['zipcode'] == zipcode]
        
        # Build complete feature set
        complete_features = {}
        
        if not demo_row.empty:
            # Add demographic features
            for col in demographics_df.columns:
                if col != 'zipcode':
                    complete_features[col] = float(demo_row.iloc[0][col])
        
        # Add housing features
        for feature in required_features:
            if feature in example_features:
                complete_features[feature] = example_features[feature]
        
        payload = {"features": complete_features}
        print(f"Sending {len(complete_features)} features to strict endpoint")
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        return False
    
    try:
        response = requests.post(f"{BASE_URL}/predict-strict", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: ${result['prediction']:,.2f}")
            print(f"Model Version: {result['model_version']}")
            print(f"Metadata: {result.get('metadata', {})}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n=== Testing Model Info ===")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            info = response.json()
            print(f"Model Loaded: {info['loaded']}")
            print(f"Version: {info['version']}")
            print(f"Feature Count: {info['feature_count']}")
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Home Price Prediction API Test Suite")
    print("=" * 60)
    
    # Check if API is running
    if not test_health_check():
        print("\n❌ API is not running. Please start the service first.")
        sys.exit(1)
    
    # Test model info
    test_model_info()
    
    # Load test examples
    try:
        examples_df = pd.read_csv('data/future_unseen_examples.csv')
        print(f"\nLoaded {len(examples_df)} test examples")
    except Exception as e:
        print(f"\n❌ Could not load test data: {e}")
        sys.exit(1)
    
    # Test with first example
    example = {"features" : examples_df.iloc[0].to_dict()}
    
    # Test /predict endpoint
    success_predict = test_predict_endpoint(example)
    
    # Test /predict-strict endpoint
    success_strict = test_predict_strict_endpoint(examples_df.iloc[0].to_dict())
    
    # Test with multiple examples
    print("\n" + "=" * 60)
    print("Testing with multiple examples")
    print("=" * 60)
    
    for i in range(min(3, len(examples_df))):
        print(f"\n--- Example {i+1} ---")
        example = examples_df.iloc[i].to_dict()
        print(f"Property: {int(example['bedrooms'])} bed, {example['bathrooms']} bath, "
              f"{int(example['sqft_living'])} sqft, zipcode {int(example['zipcode'])}")
        
        response = requests.post(f"{BASE_URL}/predict", json=example)
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Price: ${result['prediction']:,.2f}")
        else:
            print(f"Error: {response.status_code}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ Health Check: Passed")
    print(f"{'✅' if success_predict else '❌'} /predict Endpoint")
    print(f"{'✅' if success_strict else '❌'} /predict-strict Endpoint")
    
    if success_predict and success_strict:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()