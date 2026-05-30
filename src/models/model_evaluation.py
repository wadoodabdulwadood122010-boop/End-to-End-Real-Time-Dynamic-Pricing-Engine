import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import joblib
from dotenv import load_dotenv
load_dotenv()


dagshub_token = os.getenv("DAGSHUB_TOCKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOCKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "wadoodabdulwadood122010"
repo_name = "End-to-End-Real-Time-Dynamic-Pricing-Engine"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Below code block is for local use
# -------------------------------------------------------------------------------------
#mlflow.set_tracking_uri('https://dagshub.com/wadoodabdulwadood122010/End-to-End-Real-Time-Dynamic-Pricing-Engine.mlflow')
#dagshub.init(repo_owner='wadoodabdulwadood122010', repo_name='End-to-End-Real-Time-Dynamic-Pricing-Engine', mlflow=True)
mlflow.set_experiment('MODEL_EVALUTION')
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    model = joblib.load(file_path)
    return model


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logging.info('Data loaded from %s', file_path)
    return df


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    # FIXED: For Regressors, we only call .predict()
    y_pred = clf.predict(X_test)

    # FIXED: Calculate Regression performance metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics_dict = {
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse
    }
    return metrics_dict


def save_metrics(metrics: dict, file_path: str) -> None:
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # FIXED: Changed 'wb' to 'w' because json.dump handles text, not raw bytes
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    logging.info('Metrics saved to %s', file_path)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)
    logging.info('Model info saved to %s', file_path)


def main():
    with mlflow.start_run() as run:  # Start an MLflow run
        # 1. Load the binary model weights
        clf = load_model('./model/model.pkl')
        
        # 2. Load the evaluation dataset
        x_test_df = load_data('./data/processed/splited_data/x_test.csv')
        y_test_df = load_data('./data/processed/splited_data/y_test.csv')
        
        # 3. Convert DataFrames to NumPy arrays (.values) to match evaluate_model type hints
        # We use .ravel() on y_test to flatten it into a 1D array to avoid scikit-learn warnings
        metrics = evaluate_model(clf, x_test_df.values, y_test_df.values.ravel())
        
        # 4. Save metrics locally
        save_metrics(metrics, 'reports/metrics.json')
        
        # 5. Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # 6. Log model parameters to MLflow if available
        if hasattr(clf, 'get_params'):
            params = clf.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
        
        # 7. Log model artifact directly to MLflow tracking registry
        mlflow.sklearn.log_model(clf, "model")
        
        # 8. Save tracking information locally
        save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
        
        # 9. Log the generated metrics JSON file to MLflow artifacts
        mlflow.log_artifact('reports/metrics.json')


if __name__ == "__main__":
    main()