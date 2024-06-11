import sys
import os

import pandas as pd

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

from data_processing.config import logger, INPUT_PATH, OUTPUT_PATH, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME, OUTPUT_FILTERED_MERGED_FILENAME
from data_processing.load_constants import filenames, columns_to_remove, complex_name, state_name, reading_date_column
import utility.utils as utils

# df_all to store the merged dataframes with all columns
df_all = pd.DataFrame()
# df_all_filtered to store the merged dataframes with filtered columns
df_all_filtered = pd.DataFrame()

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
    global df_all, df_all_filtered
    merged_file_path = utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME)
    filtered_merged_file_path = utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME)
    
    if utils.check_file_exists(merged_file_path):
        logger.info(f"Loading the merged file {merged_file_path}")
        df_all = read_csv_file(merged_file_path)

    if utils.check_file_exists(filtered_merged_file_path):
        logger.info(f"Loading the filtered merged file {filtered_merged_file_path}")
        df_all_filtered = read_csv_file(filtered_merged_file_path)

    else:
        logger.info("Merged file not found. Loading individual processed files.")
        processed_files = [f"processed_{filename}" for filename in filenames]
        for filename in processed_files:
            processed_file_path = utils.join_file_path(OUTPUT_PATH, filename)
            if utils.check_file_exists(processed_file_path):
                df_processed = read_csv_file(processed_file_path)
                df_all = merge_files(df_all, df_processed)
                df_all_filtered = merge_files(df_all_filtered, df_processed)

    logger.info("Processed data loaded successfully.")

def process_files():
    global df_all, df_all_filtered
    logger.info("Starting file processing...")

    # Create the output directories if they don't exist
    utils.create_and_check_directory(OUTPUT_PATH)
    utils.create_and_check_directory(OUTPUT_MERGED_PATH)

    # Load the already processed data
    load_processed_data()

    for filename in filenames:
        input_file_path = utils.join_file_path(INPUT_PATH, filename)
        processed_file_path = utils.join_file_path(OUTPUT_PATH, f"processed_{filename}")
        processed_filtered_file_path = utils.join_file_path(OUTPUT_PATH, f"processed_filtered_{filename}")

        if utils.check_file_exists(processed_file_path):
            logger.info(f"File {filename} already processed. Skipping...")
            continue

        if utils.check_file_exists(input_file_path):
            try:
                df = read_csv_file(input_file_path)
                df = filter_by_state(df, state_name)
                df = filter_by_complex_name(df, complex_name)
                df = remove_duplicates(df, reading_date_column)
                save_csv_file(df, processed_file_path)

                df_filtered = remove_columns(df, columns_to_remove)
                save_csv_file(df, processed_filtered_file_path)

                logger.info(f"File {filename} processed and saved.")

                # Merge the processed data to the main dataframe
                df_all = merge_files(df_all, df)
                df_all_filtered = merge_files(df_all_filtered, df_filtered)

            except Exception as e:
                logger.error(f"Error processing file {filename}. Error: {e}")
        else:
            logger.error(f"File {filename} not found.")

    # Save the merged dataframe to a CSV file
    save_csv_file(df_all, utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME))
    save_csv_file(df_all_filtered, utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME))
    logger.info("Merged file saved successfully.")
    logger.info("Processing finished.")
