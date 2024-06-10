import sys
import os

import pandas as pd

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

from data_processing.data_processing_config import INPUT_PATH, OUTPUT_PATH, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME, logger

# File List to be processed - Will be used to train the script & adding the first 5 months of 2024 - Will be used to test the script
filenames = [f"FATOR_CAPACIDADE-2_{year}_{month:02d}.csv" for year in range(2022, 2024) for month in range(1, 13)]
filenames.extend([f"FATOR_CAPACIDADE-2_2024_{month:02d}.csv" for month in range(1, 6)])

# Columns to be removed
columns_to_remove = ["id_subsistema", "nom_subsistema", "id_estado", "nom_localizacao", "val_latitudesecoletora", "val_longitudesecoletora", "val_latitudepontoconexao", "val_longitudepontoconexao", "id_ons", "ceg"]

# Complex name and State Name to be filtered
complex_name = "Conj. Alex"
state_name = "CE"

# Column to be used as the reading date - primary key
reading_date_column = "din_instante"

# Dataframe to store all the processed data
df_all = pd.DataFrame()

def create_and_check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def join_file_path(path, filename):
    return os.path.join(path, filename)

def check_file_exists(filepath):
    return os.path.exists(filepath)

def read_csv_file(filepath):
    logger.debug(f"Reading file {filepath}")
    return pd.read_csv(filepath, sep=";", encoding="utf-8")

def filter_by_state(df, state_name):
    logger.debug(f"Filtering data by state {state_name}")
    return df[df["id_estado"] == state_name]

def remove_columns(df, columns_to_remove):
    logger.debug(f"Removing columns {columns_to_remove}")
    return df.drop(columns=columns_to_remove)

def filter_by_complex_name(df, complex_name):
    logger.debug(f"Filtering data by complex name {complex_name}")
    return df[df["nom_usina_conjunto"] == complex_name]

def remove_duplicates(df, column_name):
    logger.debug(f"Removing duplicates based on column {column_name}")
    return df.drop_duplicates(subset=[column_name])

def save_csv_file(df, filepath):
    logger.debug(f"Saving file {filepath}")
    df.to_csv(filepath, sep=";", index=False)

def merge_files(df_all, df):
    logger.debug("Merging dataframes")
    return pd.concat([df_all, df], ignore_index=True)

def load_processed_data():
    global df_all
    merged_file_path = join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME)
    if check_file_exists(merged_file_path):
        logger.info(f"Loading the merged file {merged_file_path}")
        df_all = read_csv_file(merged_file_path)
    else:
        logger.info("Merged file not found. Loading individual processed files.")
        processed_files = [f"processed_{filename}" for filename in filenames]
        for filename in processed_files:
            processed_file_path = join_file_path(OUTPUT_PATH, filename)
            if check_file_exists(processed_file_path):
                df_processed = read_csv_file(processed_file_path)
                df_all = pd.concat([df_all, df_processed], ignore_index=True)
    logger.info("Processed data loaded successfully.")

def process_files():
    global df_all
    logger.info("Starting file processing...")

    # Create the output directories if they don't exist
    create_and_check_directory(OUTPUT_PATH)
    create_and_check_directory(OUTPUT_MERGED_PATH)

    # Load the already processed data
    load_processed_data()

    for filename in filenames:
        input_file_path = join_file_path(INPUT_PATH, filename)
        processed_file_path = join_file_path(OUTPUT_PATH, f"processed_{filename}")

        if check_file_exists(processed_file_path):
            logger.info(f"File {filename} already processed. Skipping...")
            continue

        if check_file_exists(input_file_path):
            try:
                df = read_csv_file(input_file_path)
                df = filter_by_state(df, state_name)
                df = remove_columns(df, columns_to_remove)
                df = filter_by_complex_name(df, complex_name)
                df = remove_duplicates(df, reading_date_column)
                save_csv_file(df, processed_file_path)
                logger.info(f"File {filename} processed and saved.")

                # Merge the processed data to the main dataframe
                df_all = merge_files(df_all, df)

            except Exception as e:
                logger.error(f"Error processing file {filename}. Error: {e}")
        else:
            logger.error(f"File {filename} not found.")

    # Save the merged dataframe to a CSV file
    save_csv_file(df_all, join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    logger.info("Merged file saved successfully.")
    logger.info("Processing finished.")

# TODO:
# Mover a parte de Log para um arquivo de configuração de log específico
