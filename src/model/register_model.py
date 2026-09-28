import os
import yaml
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from src.logger import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


def get_version_by_alias(client: MlflowClient, alias: str):
    """Return the model version that holds an alias (None if the alias does not exist)."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def get_run_metric(client: MlflowClient, run_id: str, metric: str) -> float:
    """Read one logged metric from an MLflow run."""
    return client.get_run(run_id).data.metrics[metric]


def promote_to_production(min_roc_auc: float, tolerance: float) -> bool:
    """Promote the staging model to production if it passes the quality gate."""
    try:
        client = MlflowClient()

        staging = get_version_by_alias(client, 'staging')
        if staging is None:
            logging.error('No model with the staging alias found')
            return False

        new_auc = get_run_metric(client, staging.run_id, 'roc_auc')
        logging.info('Staging version %s: roc_auc = %.4f', staging.version, new_auc)

        # Gate 1: minimum quality
        if new_auc < min_roc_auc:
            logging.error('Rejected: roc_auc %.4f is below the minimum %.4f', new_auc, min_roc_auc)
            return False

        # Gate 2: must not be worse than the current production model
        production = get_version_by_alias(client, 'production')
        if production is not None:
            prod_auc = get_run_metric(client, production.run_id, 'roc_auc')
            logging.info('Production version %s: roc_auc = %.4f', production.version, prod_auc)
            if new_auc < prod_auc - tolerance:
                logging.error('Rejected: staging model is worse than production')
                return False

        # Promote: point the production alias to the staging version
        client.set_registered_model_alias(MODEL_NAME, 'production', staging.version)
        logging.info('Version %s promoted to production', staging.version)
        return True
    except Exception as e:
        logging.error('Unexpected error during model promotion: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')['model_registration']
        promoted = promote_to_production(params['min_roc_auc'], params['tolerance'])
        if not promoted:
            raise SystemExit(1)   # non-zero exit lets CI/CD stop the deployment
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f'Error: {e}')
        raise


if __name__ == '__main__':
    main()