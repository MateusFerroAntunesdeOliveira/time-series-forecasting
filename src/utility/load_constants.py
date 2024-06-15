# File List to be processed - Will be used to train
filenames = [f"FATOR_CAPACIDADE-2_{year}_{month:02d}.csv" for year in range(2022, 2024) for month in range(1, 13)]

# Adding the first 5 months of 2024 - Will be used to test
filenames.extend([f"FATOR_CAPACIDADE-2_2024_{month:02d}.csv" for month in range(1, 6)])

# Columns to be removed
columns_to_remove = ["id_subsistema", "nom_subsistema", "id_estado", "nom_estado", "nom_pontoconexao", "nom_localizacao", "val_latitudesecoletora", 
                    "val_longitudesecoletora", "val_latitudepontoconexao", "val_longitudepontoconexao", "nom_modalidadeoperacao", "nom_tipousina",
                    "nom_usina_conjunto", "id_ons", "ceg"]

# Complex name and State Name to be filtered
complex_name = "Conj. Alex"
state_name = "CE"

# Column to be used as the reading date - primary key
reading_date_column = "din_instante"

# Column to be used as the value to be predicted
target_column = "val_fatorcapacidade"

# DateTime Format
datetime_format = "%Y-%m-%d %H:%M:%S"

# Train and Test Periods
train_start = '2022-01-01'
train_end = '2023-12-31'
test_start = '2024-01-01'
test_end = '2024-05-31'

# Autoencoder Parameters
encoding_dimension = 2
encoding_activation = "relu"
decoding_activation = "sigmoid"
optimizer = "adam"
loss_function = "mse"
epochs = 50
batch_size = 256

# LSTM Parameters
lstm_sequence_length = 12
lstm_units = 50
lstm_activation_function = "linear"
lstm_optimizer = "adam"
lstm_loss_function = "mean_squared_error"
lstm_epochs = 10
lstm_batch_size = 32

# Gru Parameters
gru_sequence_length = 12
gru_units = 50
gru_activation_function = "linear"
gru_optimizer = "adam"
gru_loss_function = "mean_squared_error"
gru_epochs = 10
gru_batch_size = 32
