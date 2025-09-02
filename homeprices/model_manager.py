import pickle
import json
import os
import pandas as pd

class ModelManager:
    def __init__(self):
        self.model = None
        self.features = None
        self.version = "unknown"
        
    def load_model(self, version=None):
        """Load model - check latest symlink, specific version, or fallback"""
        paths = []
        
        if version:
            paths.append(f"model_artifacts/{version}/model.pkl")
        
        paths.extend([
            "model_artifacts/latest/model.pkl",
            "./model/model.pkl"
        ])
        
        for model_path in paths:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Try to load features
                features_path = model_path.replace('model.pkl', 'model_features.json')
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.features = json.load(f)
                
                # Extract version
                if "model_artifacts" in model_path:
                    parts = model_path.split("/")
                    self.version = parts[1] if len(parts) > 1 else "unknown"
                else:
                    self.version = "original"
                
                print(f"Loaded model from {model_path} (version: {self.version})")
                return True
        
        return False
    
    def predict(self, features):
        """Make prediction"""
        if not self.model:
            raise RuntimeError("No model loaded")
        
        df = pd.DataFrame([features])
        
        # Only use features the model knows about
        if hasattr(self.model, 'feature_names_in_'):
            expected = self.model.feature_names_in_
            df = df[[c for c in expected if c in df.columns]]
        elif self.features:
            df = df[[c for c in self.features if c in df.columns]]
        
        return float(self.model.predict(df)[0])
    
    def reload(self):
        return self.load_model()
    
    def is_loaded(self):
        return self.model is not None
    
    def get_version(self):
        return self.version
    
    def get_required_features(self):
        if hasattr(self.model, 'feature_names_in_'):
            return set(self.model.feature_names_in_)
        return set(self.features) if self.features else set()