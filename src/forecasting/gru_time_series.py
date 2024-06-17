import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import GRU, Dense

from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME
import utility.load_constants as constants
import utility.utils as utils

def apply_gru_forecasting(future_hours):
    file_path = utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME)
    
    df = utils.read_csv_file_as_dataframe_with_date_index(file_path, constants.reading_date_column, constants.reading_date_column)
    series = df[constants.target_column]

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    # Split the dataset into training and test datasets
    train_start_idx, train_end_idx, test_start_idx, test_end_idx = utils.convert_string_to_datetime_and_add_index_position(df, constants.train_start, constants.train_end, constants.test_start, constants.test_end)

    if train_start_idx < 0 or train_end_idx < 0 or test_start_idx < 0 or test_end_idx < 0:
        raise ValueError("One or more index positions are out of bounds.")

    train_data, test_data = utils.split_dataset_into_train_and_test(scaled_data, train_start_idx, train_end_idx, test_start_idx, test_end_idx)

    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")

    # Define the length of the input sequences
    sequence_length = constants.gru_sequence_length

    # Create train and test sequences
    X_train, y_train = utils.create_sequences_to_forecasting(train_data, sequence_length)
    X_test, y_test = utils.create_sequences_to_forecasting(test_data, sequence_length)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Reshape the input sequences in the form accepted by the GRU - [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define the GRU model
    model = Sequential([
        GRU(units=constants.gru_units, input_shape=(X_train.shape[1], 1)),
        Dense(1)  # Output layer predicts 1 step ahead
    ])

    # Compile & Train the model
    model.compile(optimizer=constants.gru_optimizer, loss=constants.gru_loss_function)
    model.fit(X_train, y_train, epochs=constants.gru_epochs, batch_size=constants.gru_batch_size, verbose=1)

    # Evaluate the model
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Train Score: {train_score:.4f}")
    logger.info(f"Test Score: {test_score:.4f}")

    # Make predictions for test data
    predictions = model.predict(X_test)

    # Invert the normalization to get the actual values
    predictions = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate performance metrics for the test data
    mae_test = mean_absolute_error(y_test_inv, predictions)
    mse_test = mean_squared_error(y_test_inv, predictions)
    rmse_test = np.sqrt(mse_test)

    logger.info(f"Test MAE: {mae_test:.4f}")
    logger.info(f"Test MSE: {mse_test:.4f}")
    logger.info(f"Test RMSE: {rmse_test:.4f}")

    # Plot the results for the test period
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[test_start_idx + sequence_length:test_end_idx], y_test_inv, label='Valor Real')
    plt.plot(df.index[test_start_idx + sequence_length:test_end_idx], predictions, label='Valor Previsto', linestyle='dashed')
    plt.title('Previsão de Fator de Capacidade com GRU')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Valor Fator de Capacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the results for the last week of May 2024
    last_week_start = pd.to_datetime("2024-05-22 00:00:00")
    last_week_end = pd.to_datetime("2024-05-28 23:00:00")
    last_week_start_idx = df.index.get_loc(last_week_start, method='nearest')
    last_week_end_idx = df.index.get_loc(last_week_end, method='nearest') + 1

    last_week_real = df[constants.target_column][last_week_start:last_week_end]
    last_week_predictions = predictions[-len(last_week_real):]

    plt.figure(figsize=(14, 6))
    plt.plot(last_week_real.index, last_week_real.values, label='Valor Real')
    plt.plot(last_week_real.index, last_week_predictions, label='Valor Previsto', linestyle='dashed')
    plt.title('Previsão de Fator de Capacidade para a Última Semana de Maio de 2024 com GRU')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Valor Fator de Capacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Future prediction for future_hours ahead
    future_predictions = []
    last_sequence = test_data[-sequence_length:].reshape((1, sequence_length, 1))

    for _ in range(future_hours):
        next_pred = model.predict(last_sequence)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_pred

    # Invert the normalization for future predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Combine last week's real values, predictions and future predictions
    extended_predictions = np.concatenate([last_week_predictions.flatten(), future_predictions.flatten()])

    # Create index for future dates
    future_dates = pd.date_range(start=last_week_end + pd.Timedelta(hours=1), periods=future_hours, freq='H')
    extended_index = last_week_real.index.append(future_dates)

    # Plot the results including the future prediction
    future_text = utils.get_future_text(future_hours)
    plt.figure(figsize=(14, 6))
    plt.plot(extended_index[:len(last_week_real)], last_week_real.values, label='Valor Real')
    plt.plot(extended_index, extended_predictions, label='Valor Previsto e Previsão Futura', linestyle='dashed')
    plt.title(f'Previsão de Fator de Capacidade para a Última Semana de Maio de 2024 e {future_text} Futuro com GRU')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Valor Fator de Capacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()
