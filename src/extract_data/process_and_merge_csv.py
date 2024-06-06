import sys
import os

import pandas as pd

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

from extract_data.process_and_merge_csv_config import INPUT_CSV_FILE_PATH, OUTPUT_CSV_FILE_PATH, OUTPUT_MERGED_CSV_FILE_PATH, OUTPUT_MERGED_CSV_FILE_NAME

# File List to be processed - Will be used to train the script
filenames = [f"FATOR_CAPACIDADE-2_{year}_{month:02d}.csv" for year in range(2022, 2024) for month in range(1, 13)]

# Adding the first 5 months of 2024 - Will be used to test the script
filenames.extend([f"FATOR_CAPACIDADE-2_2024_{month:02d}.csv" for month in range(1, 6)])

# Columns to be removed
columns_to_remove = ["id_subsistema", "nom_subsistema", "id_estado", "nom_localizacao", "val_latitudesecoletora", 
                     "val_longitudesecoletora", "val_latitudepontoconexao", "val_longitudepontoconexao", "id_ons", "ceg"]

# Complex name and State Name to be filtered
complex_name = "Conj. Alex"
state_name = "CE"

# Dataframe to store all the processed data
df_all = pd.DataFrame()

def join_file_path(path, filename):
    return os.path.join(path, filename)

def check_file_exists(filepath):
    return os.path.exists(filepath)

def read_csv_file(filepath):
    return pd.read_csv(filepath, sep=";", encoding="utf-8")

def filter_by_state(df, state_name):
    return df[df["id_estado"] == state_name]

def remove_columns(df, columns_to_remove):
    return df.drop(columns=columns_to_remove)

def filter_by_complex_name(df, complex_name):
    return df[df["nom_usina_conjunto"] == complex_name]

def save_csv_file(df, filepath):
    df.to_csv(filepath, sep=";", index=False)

def load_processed_data():
    global df_all
    merged_file_path = join_file_path(OUTPUT_MERGED_CSV_FILE_PATH, OUTPUT_MERGED_CSV_FILE_NAME)
    if check_file_exists(merged_file_path):
        df_all = read_csv_file(merged_file_path)
        print("Merged file loaded.")
        return

    processed_files = [f"processed_{filename}" for filename in filenames]
    for filename in processed_files:
        processed_file_path = join_file_path(OUTPUT_CSV_FILE_PATH, filename)
        if check_file_exists(processed_file_path):
            df_processed = read_csv_file(processed_file_path)
            df_all = pd.concat([df_all, df_processed], ignore_index=True)

def process_files():
    global df_all
    # Load the already processed data
    load_processed_data()

    for filename in filenames:
        input_file_path = join_file_path(INPUT_CSV_FILE_PATH, filename)
        processed_file_path = join_file_path(OUTPUT_CSV_FILE_PATH, f"processed_{filename}")

        if check_file_exists(processed_file_path):
            print(f"{filename} already processed. Skipping to the next file...")
            continue

        if check_file_exists(input_file_path):
            df = read_csv_file(input_file_path)
            df = filter_by_state(df, state_name)
            df = remove_columns(df, columns_to_remove)
            df = filter_by_complex_name(df, complex_name)
            save_csv_file(df, processed_file_path)
            print(f"{filename} processed and saved.")

            # Merge the processed data to the main dataframe
            df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            print(f"File {filename} not found.")

    # Save the merged dataframe to a CSV file
    save_csv_file(df_all, join_file_path(OUTPUT_MERGED_CSV_FILE_PATH, OUTPUT_MERGED_CSV_FILE_NAME))

    print("\nProcessing finished.\n")
