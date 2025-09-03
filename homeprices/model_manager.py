import pickle
import json
import os
import pandas as pd
from loguru import logger


class ModelManager:
    def __init__(self):
        self.model = None
        self.features = None
        self.version = "unknown"
    
    def list_versions(self) -> list:
        """List available model versions sorted by name"""
        versions_dir = '/app/models/versions/'
        versions_str_list = os.listdir('/app/models/versions/')
        versions_str_list.sort(reverse=True) 
        versions=[]
        for v in versions_str_list:    
            model_version_meta = {}
            model_version_meta['version'] = v
            model_version_meta['pickle_exists'] = os.path.exists(f"/app/models/versions/{v}/model.pkl")
            if os.path.exists(f"/app/models/versions/{v}/model_metadata.json"):
                with open(f"/app/models/versions/{v}/model_metadata.json", 'r') as f:
                    model_version_meta['metadata'] = json.load(f)
            else:
                model_version_meta['metadata'] = {}
            versions.append(model_version_meta)            
        
        return sorted(versions, key=lambda x : x['version'], reverse=True) if versions else []        
    
    def load_model(self, version=None) -> bool:
        """Load model - if version is None, load latest"""
        
        if version == 'original':
            model_path = "/app/models/original/model.pkl"
        elif version:
            model_path = f"/app/models/versions/{version}/model.pkl"
        elif len(os.listdir('/app/models/versions/')) > 0 :
            versions = self.list_versions()
            latest_version = versions[0]
            logger.info(f"Latest model version: {latest_version['version']}")
            model_path = f"/app/models/versions/{latest_version['version']}/model.pkl"            
        else:
            logger.warning("No model found in versions/latest, trying original")
            model_path = "/app/models/original/model.pkl"

        if os.path.exists(model_path):
            logger.info(f"Trying to load model from {model_path}")

            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Try to load features

            
        
            if hasattr(self.model, 'feature_names_in_'):
                logger.debug(f"Feature names in ({len(self.model.feature_names_in_)}): {', '.join(sorted(self.model.feature_names_in_))}")
            features_path = model_path.replace('model.pkl', 'model_features.json')        
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.features = json.load(f)
                logger.debug(f"Feature files in ({len(self.features)}): {', '.join(sorted(self.features))}")
            else:
                raise FileNotFoundError(f"Features file not found: {features_path}")
                
            
            # Extract version
            metadata_path = model_path.replace('model.pkl', 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)                    
                    self.version = metadata.get('version', 'unknown')
                    logger.info(f"Loaded model metadata: Version  {self.version}")

            else:
                self.version = "original"
            
            logger.info(f"Loaded model from {model_path} (version: {self.version})")
            return True
        logger.error(f"No model exist {model_path}")
        return False
    
    def predict(self, features: dict) -> float:
        """Make prediction"""
        if not self.model:
            raise RuntimeError("No model loaded")
        
        df = pd.DataFrame([features])
        
        
        # Only use features the model knows about
        # Model Feature_names_in_ is set by sklearn and xgboost

        if hasattr(self.model, 'feature_names_in_'):
            expected = self.model.feature_names_in_
            df = df[[c for c in expected if c in df.columns]]
        elif self.features:
            df = df[[c for c in self.features if c in df.columns]]
        
        return float(self.model.predict(df)[0])
    
    def reload(self, version=None) -> bool:
        logger.debug(f"Reloading model, version={version}")
        return self.load_model(version)
    
    def is_loaded(self):
        return self.model is not None
    
    def get_version(self):
        return self.version
    
    def get_required_features(self):
        """Get required features for the model"""

        if self.features:
            return set(self.features)
        elif hasattr(self.model, 'feature_names_in_'):
            return set(self.model.feature_names_in_)
        else:
            return set()