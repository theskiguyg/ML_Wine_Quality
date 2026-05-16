import yaml
import joblib
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def save_model(model, save_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
