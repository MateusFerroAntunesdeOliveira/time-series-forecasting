import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings

from statsmodels.tsa.arima.model import ARIMA

from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME
import utility.load_constants as constants
import utility.utils as utils

def find_best_arima_params(train_data):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_pdq = None

    warnings.filterwarnings("ignore")

    for param in pdq:
        try:
            temp_model = ARIMA(train_data, order=param)
            temp_result = temp_model.fit()
            if temp_result.aic < best_aic:
                best_aic = temp_result.aic
                best_pdq = param
        except:
            continue

    return best_pdq, best_aic

def apply_arima_forecasting(future_hours):
    df = utils.read_csv_file_as_dataframe_with_date_index(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME), constants.reading_date_column, constants.reading_date_column)
    series = df[constants.target_column]

    # Split the dataset into training and test datasets
    train_start_idx, train_end_idx, test_start_idx, test_end_idx = utils.convert_string_to_datetime_and_add_index_position(df, constants.train_start, constants.train_end, constants.test_start, constants.test_end)

    if train_start_idx < 0 or train_end_idx < 0 or test_start_idx < 0 or test_end_idx < 0:
        raise ValueError("One or more index positions are out of bounds.")

    train_data, test_data = utils.split_dataset_into_train_and_test(series, train_start_idx, train_end_idx, test_start_idx, test_end_idx)

    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")

    # Find the best ARIMA parameters
    best_pdq, best_aic = find_best_arima_params(train_data)
    logger.info(f"Best ARIMA parameters: {best_pdq}")
    logger.info(f"Best AIC: {best_aic}")

    # Fit the ARIMA model with best parameters found
    model = ARIMA(train_data, order=best_pdq)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))

    # Calculate performance metrics for the test data
    mae_test, mse_test, rmse_test, r2_test, mape_test, smape_test = utils.calculate_metrics(test_data, predictions)
    logger.info(f"Test MAE: {mae_test:.4f}")
    logger.info(f"Test MSE: {mse_test:.4f}")
    logger.info(f"Test RMSE: {rmse_test:.4f}")
    logger.info(f"Test R2: {r2_test:.4f}")
    logger.info(f"Test MAPE: {mape_test:.4f}")
    logger.info(f"Test SMAPE: {smape_test:.4f}")

    # Plot the results for the test period
    plt.figure(figsize=(14, 6))
    plt.plot(test_data.index, test_data.values, label='Valor Real')
    plt.plot(test_data.index, predictions, label='Valor Previsto', linestyle='dashed')
    plt.title('Previsão de Fator de Capacidade com ARIMA')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Valor Fator de Capacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the results for the last week of May 2024
    last_week_start = pd.to_datetime("2024-05-22 00:00:00")
    last_week_end = pd.to_datetime("2024-05-28 23:00:00")

    last_week_real = df[constants.target_column][last_week_start:last_week_end]
    last_week_predictions = model_fit.forecast(steps=len(last_week_real))

    plt.figure(figsize=(14, 6))
    plt.plot(last_week_real.index, last_week_real.values, label='Valor Real')
    plt.plot(last_week_real.index, last_week_predictions, label='Valor Previsto', linestyle='dashed')
    plt.title('Previsão de Fator de Capacidade para a Última Semana de Maio de 2024')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Valor Fator de Capacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Future prediction for one day ahead
    future_predictions = model_fit.forecast(steps=future_hours)

    # Combine last week's real values, predictions and future predictions
    extended_predictions = np.concatenate([last_week_predictions, future_predictions])

    # Create index for future dates
    future_dates = pd.date_range(start=last_week_end + pd.Timedelta(hours=1), periods=future_hours, freq='H')
    extended_index = last_week_real.index.append(future_dates)

    # Plot the results including the future prediction
    future_text = utils.get_future_text(future_hours)
    plt.figure(figsize=(14, 6))
    plt.plot(extended_index[:len(last_week_real)], last_week_real.values, label='Valor Real')
    plt.plot(extended_index, extended_predictions, label='Valor Previsto e Previsão Futura', linestyle='dashed')
    plt.title(f'Previsão de Fator de Capacidade para a Última Semana de Maio de 2024 e {future_text} Futuro')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Valor Fator de Capacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_data, test_data
