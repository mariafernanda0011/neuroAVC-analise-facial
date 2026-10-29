import os
import numpy as np
import pandas as pd
from scipy.spatial import procrustes

def aplicar_procrustes_em_dataset(nome_csv_entrada, nome_csv_saida):

    # Definindo caminhos
    caminho_base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    caminho_landmarks_dir = os.path.join(caminho_base_data, 'landmarks')
    
    arquivo_entrada = os.path.join(caminho_landmarks_dir, nome_csv_entrada)
    arquivo_saida = os.path.join(caminho_landmarks_dir, nome_csv_saida)

    if not os.path.exists(arquivo_entrada):
        print(f"❌ Erro: O arquivo '{arquivo_entrada}' não foi encontrado.")
        return

    print(f"\n==================================================")
    print(f"🔄 Processando Procrustes para: {nome_csv_entrada}")
    print(f"==================================================")
    
    # Carregando o arquivo CSV
    df = pd.read_csv(arquivo_entrada)

    # Separa as features (coordenadas x e y) do rótulo
    X = df.drop(columns=['rotulo']).values
    y = df['rotulo'].values

    num_amostras = X.shape[0]
    num_pontos = 478  # Total de landmarks do MediaPipe

    # Calcula a média das coordenadas de todos os rostos para servir de referência geométrica ideal
    X_reshaped = X.reshape(num_amostras, num_pontos, 2)
    referencia = np.mean(X_reshaped, axis=0)

    print(f"Aplicando alinhamento de Procrustes em {num_amostras} amostras...")

    X_procrustes_list = []

    for i in range(num_amostras):
        rosto_atual = X_reshaped[i]
        # Alinha o rosto atual em relação ao rosto de referência médio
        _, rosto_alinhado, _ = procrustes(referencia, rosto_atual)
        # Achata a matriz de volta para 1D (x1, y1, x2, y2...)
        X_procrustes_list.append(rosto_alinhado.flatten())

    # Converte de volta para array numpy
    X_proc_array = np.array(X_procrustes_list)

    # Reconstrói as colunas no formato original para salvar em CSV
    colunas_x_y = []
    for i in range(num_pontos):
        colunas_x_y.append(f'x_{i}')
        colunas_x_y.append(f'y_{i}')

    df_procrustes = pd.DataFrame(X_proc_array, columns=colunas_x_y)
    df_procrustes['rotulo'] = y

    # Salva o novo dataset padronizado na pasta landmarks
    df_procrustes.to_csv(arquivo_saida, index=False)

    print(f"✅ Alinhamento de Procrustes concluído com sucesso!")
    print(f"📂 Arquivo salvo em: '{arquivo_saida}'")

if __name__ == "__main__":
    # 1. Processa o dataset original
    aplicar_procrustes_em_dataset("dataset_landmarks.csv", "dataset_landmarks_procrustes.csv")
    
    # 2. Processa o dataset augmentado
    aplicar_procrustes_em_dataset("dataset_landmarks_aug.csv", "dataset_landmarks_aug_procrustes.csv")