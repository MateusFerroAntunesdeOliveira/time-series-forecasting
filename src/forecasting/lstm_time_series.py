import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME
import utility.load_constants as constants
import utility.utils as utils

def load_data(file_path, parse_date, index_col):
    return pd.read_csv(file_path, sep=";", parse_dates=[parse_date], index_col=index_col)

def split_dataset(dataset, train_start_idx, train_end_idx, test_start_idx, test_end_idx):
    train_data = dataset[train_start_idx:train_end_idx]
    test_data = dataset[test_start_idx:test_end_idx]
    return train_data, test_data

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def convert_string_to_datetime_and_add_index_position(df, train_start, train_end, test_start, test_end):
    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    train_start_idx = df.index.get_loc(train_start, method='nearest')
    train_end_idx = df.index.get_loc(train_end, method='nearest') + 1
    test_start_idx = df.index.get_loc(test_start, method='nearest')
    test_end_idx = df.index.get_loc(test_end, method='nearest') + 1

    return train_start_idx, train_end_idx, test_start_idx, test_end_idx

def get_future_text(future_hours):
    hour_mapping = {
        1: "1 hora",
        12: "12 horas",
        24: "1 dia",
        48: "2 dias"
    }
    return hour_mapping.get(future_hours, f"{future_hours} horas")

def apply_lstm_forecasting(future_hours):
    df = load_data(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME), constants.reading_date_column, constants.reading_date_column)
    series = df[constants.target_column]

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    # Split the dataset into training and test datasets
    train_start_idx, train_end_idx, test_start_idx, test_end_idx = convert_string_to_datetime_and_add_index_position(df, constants.train_start, constants.train_end, constants.test_start, constants.test_end)

    if train_start_idx < 0 or train_end_idx < 0 or test_start_idx < 0 or test_end_idx < 0:
        raise ValueError("One or more index positions are out of bounds.")

    train_data, test_data = split_dataset(scaled_data, train_start_idx, train_end_idx, test_start_idx, test_end_idx)

    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")

    # Define the length of the input sequences
    sequence_length = constants.lstm_sequence_length

    # Create train and test sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Reshape the input sequences in the form accepted by the LSTM - [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define the LSTM model
    model = Sequential([
        LSTM(input_shape=(X_train.shape[1], 1), units=constants.lstm_units),
        Dense(future_hours),  # Output layer predicts 'future_hours' steps ahead
    ])

    # Compile & Train the model
    model.compile(optimizer=constants.lstm_optimizer, loss=constants.lstm_loss_function)
    model.fit(X_train, y_train, epochs=constants.lstm_epochs, batch_size=constants.lstm_batch_size, verbose=1)

    # Evaluate the model
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Train Score: {train_score:.4f}")
    logger.info(f"Test Score: {test_score:.4f}")

    # Make predictions
    predictions = model.predict(X_test)

    # Invert the normalization to get the actual values
    predictions = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test)

    # Plot the results for the test period
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[test_start_idx + sequence_length:test_end_idx], y_test_inv, label='Valor Real')
    plt.plot(df.index[test_start_idx + sequence_length:test_end_idx], predictions[:, 0], label='Valor Previsto', linestyle='dashed')
    plt.title('Previsão de Fator de Capacidade com LSTM')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Val_fatorcapacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the results for the last week of May 2024
    last_week_start = pd.to_datetime("2024-05-22 00:00:00")
    last_week_end = pd.to_datetime("2024-05-28 23:00:00")
    last_week_start_idx = df.index.get_loc(last_week_start, method='nearest')
    last_week_end_idx = df.index.get_loc(last_week_end, method='nearest') + 1

    last_week_real = df[constants.target_column][last_week_start:last_week_end]
    last_week_predictions = predictions[-len(last_week_real):, 0]

    plt.figure(figsize=(14, 6))
    plt.plot(last_week_real.index, last_week_real.values, label='Valor Real')
    plt.plot(last_week_real.index, last_week_predictions, label='Valor Previsto', linestyle='dashed')
    plt.title('Previsão de Fator de Capacidade para a Última Semana de Maio de 2024')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Val_fatorcapacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Future prediction for one day ahead
    last_sequence = test_data[-sequence_length:]  # Last sequence from the test data
    last_sequence = last_sequence.reshape((1, sequence_length, 1))

    future_predictions = []
    for _ in range(future_hours):  # Predicting 'future_hours' ahead
        next_pred = model.predict(last_sequence)
        future_predictions.append(next_pred[0, 0])  # Take the first value of the prediction
        next_pred = next_pred[0, 0].reshape((1, 1, 1))
        last_sequence = np.append(last_sequence[:, 1:, :], next_pred, axis=1)

    # Invert the normalization for future predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Combine last week's real values, predictions and future predictions
    extended_predictions = np.concatenate([last_week_predictions, future_predictions.flatten()])

    # Create index for future dates
    future_dates = pd.date_range(start=last_week_end + pd.Timedelta(hours=1), periods=future_hours, freq='H')
    extended_index = last_week_real.index.append(future_dates)

    # Plot the results including the future prediction
    future_text = get_future_text(future_hours)
    plt.figure(figsize=(14, 6))
    plt.plot(extended_index[:len(last_week_real)], last_week_real.values, label='Valor Real')
    plt.plot(extended_index, extended_predictions, label='Valor Previsto e Previsão Futura', linestyle='dashed')
    plt.title(f'Previsão de Fator de Capacidade para a Última Semana de Maio de 2024 e {future_text} Futuro')
    plt.xlabel('Tempo (horas)')
    plt.ylabel('Val_fatorcapacidade (MW/MW)')
    plt.legend()
    plt.grid(True)
    plt.show()
