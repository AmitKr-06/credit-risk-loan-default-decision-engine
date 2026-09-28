import pandas as pd
import numpy as np
import os
import yaml
import logging
from src.logger import logging

# Mapping for employment length text -> number
EMP_LENGTH_MAP = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                  '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
                  '10+ years': 10}

def load_params(params_path: str)-> dict:
    """ Load Parameters from YAML file. """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str)-> pd.DataFrame:
    """ Load data from a CSV file. """
    try:
        df = pd.read_csv(data_url, low_memory = False)  # low_memory avoids mixed-type warning
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning only. No statistics are learned here (avoids leakage)."""
    try:
        logging.info('Pre-Processing...')
        # Drop useless columns
        df = df.drop(columns = ['id', 'desc', 'title', 'experience_c'])
        # Convert "Dec-2015" text to a real date
        df['issue_d'] = pd.to_datetime(df['issue_d'], format = '%b-%Y')
        # Convert emp_length text to a number (unknown statys NaN)
        df['emp_length_n'] = df['emp_length'].map(EMP_LENGTH_MAP)
        df = df.drop(columns = ['emp_length'])
        # 999 in dti_n is a fake value, so mark it as missing
        df.loc[df['dti_n'] >= 900, 'dti_n'] = np.nan
        logging.info('Data Preprocessing completed')
        return df
    except KeyError as e:
        logging.error('Missing column in the DataFrame: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def time_split(df: pd.DataFrame, train_end: int, val_year: int, test_start: int):
    """Split by issue year so we never train on the future."""
    try:
        year = df['issue_d'].dt.year
        train_df = df[year <= train_end].copy()

        val_df = df[year == val_year].copy()
        test_df = df[year >= test_start].copy()

        logging.info('Split Shapes: %s %s %s', train_df.shape, val_df.shape, test_df.shape)
        return train_df, val_df, test_df
    except Exception as e:
        logging.error('Unexpected error during time split: %s', e)
        raise

def save_data(train_df, val_df, test_df, data_path: str) -> None:
    """ Save the train, val, and test datasets."""
    try:
        interim_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_path, exist_ok = True)
        train_df.to_csv(os.path.join(interim_path, 'train.csv'), index = False)
        val_df.to_csv(os.path.join(interim_path, 'val.csv'), index = False)
        test_df.to_csv(os.path.join(interim_path, 'test.csv'), index = False)

        logging.debug('Train, Val adn Test data saved to %s', interim_path)
    except Exception as e:
        logging.error('Unexpected error while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')['data_ingestion']
        df = load_data('data/raw/loan_data.csv')
        df = preprocess_data(df)
        train_df, val_df, test_df = time_split(df, params['train_end_year'],
                                               params['val_year'], params['test_start_year'])

        save_data(train_df, val_df, test_df, data_path = './data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()