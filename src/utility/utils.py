import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utility.config import logger

def create_and_check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def join_file_path(path, filename):
    return os.path.join(path, filename)

def check_file_exists(filepath):
    return os.path.exists(filepath)

def read_csv_file_as_dataframe(file_path):
    logger.debug(f"Reading file {file_path}")
    return pd.read_csv(file_path, sep=";", encoding="utf-8")

def remove_columns(df, columns_to_remove):
    logger.debug(f"Removing columns {columns_to_remove}")
    return df.drop(columns=columns_to_remove)

def save_csv_file(df, filepath):
    logger.debug(f"Saving file {filepath}")
    df.to_csv(filepath, sep=";", index=False)

def read_csv_file_as_dataframe_with_date_index(file_path, parse_date, index_col):
    return pd.read_csv(file_path, sep=";", parse_dates=[parse_date], index_col=index_col)

def split_dataset_into_train_and_test(dataset, train_start_idx, train_end_idx, test_start_idx, test_end_idx):
    train_data = dataset[train_start_idx:train_end_idx]
    test_data = dataset[test_start_idx:test_end_idx]
    return train_data, test_data

def create_sequences_to_forecasting(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def convert_string_to_datetime_and_add_index_position(df, train_start, train_end, test_start, test_end):
    train_start_idx = df.index.get_loc(pd.to_datetime(train_start), method='nearest')
    train_end_idx = df.index.get_loc(pd.to_datetime(train_end), method='nearest') + 1
    test_start_idx = df.index.get_loc(pd.to_datetime(test_start), method='nearest')
    test_end_idx = df.index.get_loc(pd.to_datetime(test_end), method='nearest') + 1

    return train_start_idx, train_end_idx, test_start_idx, test_end_idx

def get_future_text(future_hours):
    hour_mapping = {
        1: "1 hora",
        12: "12 horas",
        24: "1 dia",
        48: "2 dias"
    }
    return hour_mapping.get(future_hours, f"{future_hours} horas")

def mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true + epsilon
    ape = np.abs((y_true - y_pred) / y_true)
    return np.mean(ape) * 100

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape_score = mape(y_true, y_pred)
    smape_score = smape(y_true, y_pred)
    return mae, mse, rmse, r2, mape_score, smape_score
