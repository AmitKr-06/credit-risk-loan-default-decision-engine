import pandas as pd
import numpy as np
import os
import json
import yaml
import joblib
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_recall_curve, roc_auc_score, brier_score_loss
from src.logger import logging


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
    """Load the trained model from a file."""
    try:
        model = joblib.load(file_path)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading the data: %s', e)
        raise


def calibrate_model(model, X_val: pd.DataFrame, y_val: pd.Series, method: str):
    """Fix the probabilities using the validation data (the base model stays frozen)."""
    try:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrated.fit(X_val, y_val)
        logging.info('Model calibrated using %s method', method)
        return calibrated
    except Exception as e:
        logging.error('Unexpected error during calibration: %s', e)
        raise


def find_f1_threshold(y_val: pd.Series, val_proba: np.ndarray) -> dict:
    """Pick the threshold that gives the best F1 score on validation."""
    try:
        precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best = int(np.argmax(f1[:-1]))   # last point has no threshold
        result = {
            'threshold': float(thresholds[best]),
            'val_precision': float(precision[best]),
            'val_recall': float(recall[best]),
            'val_f1': float(f1[best]),
            'val_roc_auc': float(roc_auc_score(y_val, val_proba)),
            'val_brier': float(brier_score_loss(y_val, val_proba)),
            'val_mean_predicted': float(val_proba.mean()),
            'val_actual_rate': float(y_val.mean()),
        }
        logging.info('F1-optimal threshold: %.4f', result['threshold'])
        return result
    except Exception as e:
        logging.error('Unexpected error while finding the threshold: %s', e)
        raise


def save_artifacts(calibrated, threshold_info: dict, model_dir: str) -> None:
    """Save the calibrated model and the threshold."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(calibrated, os.path.join(model_dir, 'calibrated_model.pkl'))
        with open(os.path.join(model_dir, 'threshold.json'), 'w') as file:
            json.dump(threshold_info, file, indent=4)
        logging.info('Calibrated model and threshold saved to %s', model_dir)
    except Exception as e:
        logging.error('Unexpected error while saving the artifacts: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')['model_calibration']

        model = load_model('./models/model.pkl')
        val_df = load_data('./data/processed/val.csv')
        X_val = val_df.drop(columns=['Default'])
        y_val = val_df['Default']

        # Calibrate on val, then choose the threshold on the calibrated probabilities
        calibrated = calibrate_model(model, X_val, y_val, params['method'])
        val_proba = calibrated.predict_proba(X_val)[:, 1]
        threshold_info = find_f1_threshold(y_val, val_proba)

        save_artifacts(calibrated, threshold_info, 'models')
    except Exception as e:
        logging.error('Failed to complete the model calibration process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()