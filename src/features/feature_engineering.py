import joblib
import pandas as pd
import numpy as np
import os
import yaml
import logging
from sklearn.preprocessing import OneHotEncoder
from src.logger import logging

def load_params(params_path: str) -> dict:
    """ Load parameters from YAML file. """
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
    """Load a CSV file (issue_d is parsed back into a date)."""
    try:
        df = pd.read_csv(data_path, parse_dates = ['issue_d'])
        logging.info('Data loaded from %s', data_path)
        return df
    except Exception as e:
        logging.error('Unexpected error while loading the data: %s', e)
        raise

def fit_preprocessor(train_df: pd.DataFrame, params: dict) -> dict:
    """Learn all statistics from TRAIN only (no leakage)."""
    try:
        pre = {}
        # Median and cap for dti_n
        pre['dti_median'] = train_df['dti_n'].median()
        pre['dti_cap'] = train_df['dti_n'].quantile(params['dti_cap_quantile'])

        # One-hot encoder for low-cardinality columns
        pre['ohe_cols'] = ['purpose', 'home_ownership_n']
        pre['ohe'] = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
        pre['ohe'].fit(train_df[pre['ohe_cols']])

        # Smoothed target encoding maps for state and zip code
        pre['te_maps'] = {}
        smoothing = params['te_smoothing']
        global_mean = train_df['Default'].mean()

        for c in ['addr_state', 'zip_code']:
            agg = train_df.groupby(c)['Default'].agg(['mean', 'count'])
            smoothed = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
            pre['te_maps'][c] = (smoothed, global_mean)

        logging.info('Preprocessor fitted on train data')
        return pre
    except Exception as e:
        logging.error('Unexpected error while the preprocessor: %s', e)
        raise

def transform(df: pd.DataFrame, pre: dict, final_features: list) -> pd.DataFrame:
    """Apply the learned preprocessing to any split (train/ val/ test/ live data)."""
    try:
        out = df.copy()

        # Fill the missing zip code with a placeholder
        out['zip_code'] = out['zip_code'].fillna('Unknown')
        # Fill missing dti with train median, then cap outliers
        out['dti_n'] = out['dti_n'].fillna(pre['dti_median']).clip(upper = pre['dti_cap'])
        # Log of revenue reduces the extreme skew
        out['revenue_log'] = np.log1p(out['revenue'])
        # Loan amount compared to income
        out['loan_to_income'] = out['loan_amnt'] / out['revenue']
        # Month of laon issue
        out['issue_month'] = out['issue_d'].dt.month
        # Unknown employment length becomes -1
        out['emp_length_n'] = out['emp_length_n'].fillna(-1)

        # One-hot encode purpose and home ownership
        ohe_df = pd.DataFrame(pre['ohe'].transform(out[pre['ohe_cols']]),
                              columns = pre['ohe'].get_feature_names_out(pre['ohe_cols']),
                              index = out.index)

        out = pd.concat([out, ohe_df], axis = 1)

        # Target encode state and zip (unseen values get the global mean)
        for c, (mapping, g_mean) in pre['te_maps'].items():
            out[c + '_te'] = out[c].map(mapping).fillna(g_mean)

        # Keep only the final features (+ target if present)
        cols = final_features + (['Default'] if 'Default' in out.columns else [])
        logging.info('Transformed data shape: %s', out[cols].shape)
        return out[cols]
    except KeyError as e:
        logging.error('Missing column in the DataFrame: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during transformation: %s', e)
        raise

def save_data(train_df, val_df, test_df, data_path: str) -> None:
    """Save the processed datasets."""
    try:
        processed_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_path, exist_ok = True)
        train_df.to_csv(os.path.join(processed_path, 'train.csv'), index = False)
        val_df.to_csv(os.path.join(processed_path, 'val.csv'), index = False)
        test_df.to_csv(os.path.join(processed_path, 'test.csv'), index = False)

        logging.debug('Processed data saved to %s', processed_path)
    except Exception as e:
        logging.error('Unexpected error while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')['feature_engineering']

        train_df = load_data('./data/interim/train.csv')
        val_df = load_data('./data/interim/val.csv')
        test_df = load_data('./data/interim/test.csv')

        # Learn on train only, then apply to all splits
        pre = fit_preprocessor(train_df, params)
        train_out = transform(train_df, pre, params['final_features'])
        val_out = transform(val_df, pre, params['final_features'])
        test_out = transform(test_df, pre, params['final_features'])

        # Save the preprocessor (neede later by the API)
        os.makedirs('models', exist_ok = True)
        joblib.dump(pre, 'models/preprocessor.pkl')

        save_data(train_df, val_df, test_df, data_path = './data')
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()