import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

from utility.load_constants import encoding_dimension, encoding_activation, decoding_activation, optimizer, loss_function, epochs, batch_size
from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME
import utility.utils as utils

def load_data(file_path):
    return utils.read_csv_file_as_dataframe(file_path)

def preprocess_data(df):
    # Select numerical columns for autoencoder
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    data = df[numerical_columns]
    return data

def apply_encoder(df):
    logger.debug(f"Applying Autoencoder on DataFrame with shape: {df.shape}")

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    input_dim = df_scaled.shape[1]

    # Define the autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoding_layer = Dense(encoding_dimension, activation=encoding_activation)(input_layer)
    decoding_layer = Dense(input_dim, activation=decoding_activation)(encoding_layer)

    # Create the Autoencoder / Encoder Model
    autoencoder = Model(inputs=input_layer, outputs=decoding_layer)
    encoder = Model(inputs=input_layer, outputs=encoding_layer)

    # Compile and Train the Autoencoder
    autoencoder.compile(optimizer=optimizer, loss=loss_function)
    autoencoder.fit(df_scaled, df_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    # Get the encoded (reduced) data
    encoded_data = encoder.predict(df_scaled)
    logger.debug(f"Encoded data shape: {encoded_data.shape}")

    return encoded_data

def visualize_encoded_data(encoded_data):
    if encoded_data.shape[1] < 2:
        logger.error("Encoded data has less than 2 dimensions, cannot visualize")
        return
    
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
    plt.title("Dados Codificados")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.show()

def apply_autoencoder_analysis():
    logger.info("Applying Autoencoder Analysis")
    df = load_data(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    
    data = preprocess_data(df)
    encoded_data = apply_encoder(data)
    visualize_encoded_data(encoded_data)
    
    logger.info("Autoencoder Analysis Completed")
