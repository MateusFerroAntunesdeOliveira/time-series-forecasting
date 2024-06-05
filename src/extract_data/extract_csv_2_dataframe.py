import pandas as pd
import os

input_folder = "../readings/"
output_folder = "../processed/"

os.makedirs(output_folder, exist_ok=True)

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

full_df = pd.DataFrame()

def extract_dataframe_handler():
    for filename in filenames:
        # Full path of the input file
        input_path = os.path.join(input_folder, filename)
        processed_path = os.path.join(output_folder, f"processed_{filename}")
        
        if os.path.exists(processed_path):
            print(f"{filename} already processed. Skipping to the next file...")
            continue

        if os.path.exists(input_path):
            # Read the CSV file
            df = pd.read_csv(input_path, sep=";")
            
            # Filter by state name
            df_ce = df[df["id_estado"] == state_name]
            
            # Remove unnecessary columns
            df_ce = df_ce.drop(columns=columns_to_remove)
            
            # Filter by complex name
            df_alex_complex = df_ce[df_ce["nom_usina_conjunto"] == complex_name]
            
            # Full path of the output file
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the processed file
            df_alex_complex.to_csv(output_path, sep=";", index=False)
            
            print(f"{output_filename} saved.")
        else:
            print(f"File {filename} not found.")

    print("\nProcessing finished.\n")
