import pandas as pd
import numpy as np
import os
import yaml
import joblib
import logging
from lightgbm import LGBMClassifier
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


def load_data(data_path: str) -> pd.DataFrame:
    """Load a processed CSV file."""
    try:
        df = pd.read_csv(data_path)
        logging.info('Data loaded from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading the data: %s', e)
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> LGBMClassifier:
    """Train LightGBM with the tuned parameters."""
    try:
        logging.info('Training LightGBM on %s rows and %s features', X_train.shape[0], X_train.shape[1])
        model = LGBMClassifier(**params, random_state = 42, verbose = -1)
        model.fit(X_train, y_train)
        logging.info('Model training completed')
        return model
    except Exception as e:
        logging.error('Unexpected error during model training: %s', e)
        raise

def save_model(model, model_path: str) -> None:
    """Save the trained model to disk."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok = True)
        joblib.dump(model, model_path)
        logging.info('Model saved to %s', model_path)
    except Exception as e:
        logging.error('Unexpected error while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')['model_building']

        train_df = load_data('./data/processed/train.csv')
        #Split features and target
        X_train = train_df.drop(columns = ['Default'])
        y_train = train_df['Default']

        model = train_model(X_train, y_train, params)
        save_model(model, 'models/model.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()