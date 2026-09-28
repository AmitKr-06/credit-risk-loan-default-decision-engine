import numpy as np
import pandas as pd
import joblib
import json
import os
import yaml
import logging
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, precision_score, f1_score, recall_score

from src.logger import logging
from dotenv import load_dotenv

# Load environment variable from .env file
load_dotenv()

# ---------------------------------------------------------------------------------------
# For Production level code to model evaluation
# ---------------------------------------------------------------------------------------

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AmitKr-06"
repo_name = "credit-risk-loan-default-decision-engine"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

MODEL_NAME = "credit_risk_model"

def load_params(params_path: str) -> dict:
    """Load parameters from YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_model(file_path: str):
    """Load the trained model from a file. """
    try:
        model = joblib.load(file_path)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file. """
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_threshold(file_path: str) -> float:
    """Load the decision threshold chosen on validation data."""
    try:
        with open(file_path, 'r') as file:
            threshold = json.load(file)['threshold']
        logging.info('Threshold loaded: %.4f', threshold)
        return threshold
    except Exception as e:
        logging.error('Unexpected error while loading the threshold: %s', e)
        raise


def evaluate_model(clf, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)   # apply our decision threshold

        metrics_dict = {
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'pr_auc': float(average_precision_score(y_test, y_proba)),
            'brier': float(brier_score_loss(y_test, y_proba)),
            'log_loss': float(log_loss(y_test, y_proba)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'flagged_rate': float(y_pred.mean()),
            'mean_predicted': float(y_proba.mean()),
            'actual_rate': float(y_test.mean()),
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, version: str, file_path: str) -> None:
    """Save the model run ID, path and registry version to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path, 'model_version': version}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_experiment("credit-risk-pipeline")
    with mlflow.start_run() as run:   # Start an MLflow run
        try:
            params = load_params('params.yaml')
            threshold = load_threshold('./models/threshold.json')

            clf = load_model('./models/calibrated_model.pkl')
            test_df = load_data('./data/processed/test.csv')
            X_test = test_df.drop(columns=['Default'])
            y_test = test_df['Default']

            # Evaluate model and save metrics locally
            metrics = evaluate_model(clf, X_test, y_test, threshold)
            save_metrics(metrics, 'reports/metrics.json')

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters and run settings to MLflow
            for param_name, param_value in params['model_building'].items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param('calibration_method', params['model_calibration']['method'])
            mlflow.log_param('threshold', threshold)
            mlflow.log_param('n_features', X_test.shape[1])
            mlflow.set_tag('model_type', 'LightGBM + sigmoid calibration')

            # Log model to MLflow and register it (creates a new version each run)
            sample = X_test.head(100).astype('float64')
            signature = infer_signature(sample, clf.predict(sample))
            model_info = mlflow.sklearn.log_model(clf, name="model", signature=signature,
                                      input_example=sample.head(5),
                                      serialization_format="cloudpickle",
                                      registered_model_name=MODEL_NAME)

            logging.info("Model logged to MLflow successfully!")
            logging.info(f"Model ID: {model_info.model_id}")
            logging.info(f"Registered Model Version: {model_info.registered_model_version}")

            # Point the 'staging' alias to the new version
            version = str(model_info.registered_model_version)
            if model_info.registered_model_version:
                client = mlflow.MlflowClient()
                client.set_registered_model_alias(name=MODEL_NAME, alias="staging", version=version)
                logging.info(f"Model version {version} set as staging.")

            # Log helper files (the API needs the preprocessor too)
            mlflow.log_artifact('models/threshold.json')
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('models/preprocessor.pkl')
            mlflow.log_artifact('params.yaml')

            # Save model info for downstream stages
            save_model_info(run.info.run_id, "model", version, 'reports/experiment_info.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")
            raise   # so DVC marks the stage as failed


if __name__ == '__main__':
    main()