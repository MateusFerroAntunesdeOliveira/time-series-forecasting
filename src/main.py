import pandas as pd
import os

input_folder = './readings/'
output_folder = './processed/'

os.makedirs(output_folder, exist_ok=True)

# Lista de arquivos a serem processados
# filenames = [f'FATOR_CAPACIDADE-2_{year}_{month:02d}.csv' for year in range(2022, 2024) for month in range(1, 13)]
filenames = ["FATOR_CAPACIDADE-2_2024_02.csv"]

# Colunas a serem removidas
columns_to_remove = ['id_subsistema', 'nom_subsistema', 'id_estado', 'nom_localizacao', 'val_latitudesecoletora', 
                     'val_longitudesecoletora', 'val_latitudepontoconexao', 'val_longitudepontoconexao', 'id_ons', 'ceg']

# Nome do complexo a ser filtrado
complex_name = 'Conj. Alex'

# Nome do estado a ser filtrado
state_name = 'CE'

for filename in filenames:
    # Caminho completo do arquivo de entrada
    input_path = os.path.join(input_folder, filename)
    processed_path = os.path.join(output_folder, f'processed_{filename}')
    
    if os.path.exists(processed_path):
        print(f'{filename} já foi processado. Pulando para o próximo arquivo.')
        continue

    if os.path.exists(input_path):
        # Ler o arquivo CSV
        df = pd.read_csv(input_path, sep=';')
        
        # Filtrar pelo estado CE
        df_ce = df[df['id_estado'] == state_name]
        
        # Remover colunas desnecessárias
        df_ce = df_ce.drop(columns=columns_to_remove)
        
        # Filtrar pelo nome do complexo
        df_alex = df_ce[df_ce['nom_usina_conjunto'] == complex_name]
        
        # Caminho completo do arquivo de saída
        output_filename = f'processed_{filename}'
        output_path = os.path.join(output_folder, output_filename)
        
        # Salvar o resultado em um novo arquivo CSV
        df_alex.to_csv(output_path, sep=';', index=False)
        
        print(f'{output_filename} salvo com sucesso.')
    else:
        print(f'Arquivo {filename} não encontrado.')

print('Processamento concluído.')
