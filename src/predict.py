from preprocess import get_new_preprocessor
import pandas as pd
import joblib
import os
from sklearn.metrics import r2_score
from utils import load_config, load_model


def start_predict(config_path):
    config = load_config(config_path)
    model = load_model(config['inference_model_path'])

    X_test = pd.read_csv(config['data']['X_test_path'])
    y_test = pd.read_csv(config['data']['y_test_path'])
    X_test = X_test.drop(['Id'], axis=1)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    mode = config['mode']

    if mode == "predict":
        print(f"Test r2: {r2:.4f}. ")
        print("Submission file is not created. ")
    elif mode == 'submit':
        submit_path = config.get('submission_path', 'data/submissions/submission.csv')
        os.makedirs(os.path.dirname(submit_path), exist_ok=True)
        submission = pd.DataFrame({
            'prediction': predictions
        })
        submission.to_csv(submit_path, index=False)
        print(f"Submission saved to {submit_path}")

if __name__ == "__main__":
    start_predict('configs/predict.yaml')