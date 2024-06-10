
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
