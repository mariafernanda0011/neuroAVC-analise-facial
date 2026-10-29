import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extrair_landmarks_do_dataset(nome_dataset, nome_csv_saida):
    print(f"\n==================================================")
    print(f"🚀 Iniciando processamento do dataset: {nome_dataset}")
    print(f"==================================================")

    # 1. Definir caminhos baseando-se na estrutura do projeto
    caminho_base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    caminho_pasta_dataset = os.path.join(caminho_base_data, nome_dataset)
    caminho_zip = os.path.join(caminho_pasta_dataset, f"{nome_dataset}.zip")
    caminho_landmarks_dir = os.path.join(caminho_base_data, 'landmarks')

    # Garante que a pasta de saída dos CSVs existe
    os.makedirs(caminho_landmarks_dir, exist_ok=True)

    # 2. Verificar se a pasta descompactada existe, senão descompacta o .zip
    if not os.path.exists(os.path.join(caminho_pasta_dataset, 'normal')):
        if os.path.exists(caminho_zip):
            print(f"📦 Descompactando {nome_dataset}.zip...")
            with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                zip_ref.extractall(caminho_pasta_dataset)
            print("✅ Descompactação concluída!")
        else:
            print(f"❌ Erro: O arquivo {caminho_zip} não foi encontrado.")
            return

    # 3. Garantir que o modelo do Face Landmarker (.task) está baixado na raiz do script ou projeto
    model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
    if not model_path in globals() and not os.path.exists(model_path):
        # Tenta procurar na raiz do projeto também
        model_path_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'face_landmarker.task'))
        if os.path.exists(model_path_raiz):
            model_path = model_path_raiz
        else:
            print("📥 Baixando o modelo oficial do MediaPipe Face Landmarker...")
            import urllib.request
            url_modelo = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url_modelo, model_path)
            print("✅ Modelo baixado com sucesso!")

    # 4. Configurar o detector do MediaPipe
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    pastas = {'normal': 0, 'anormal': 1}
    dados = []

    print("🔍 Iniciando extração de Landmarks faciais...")

    for nome_pasta, rotulo in pastas.items():
        caminho_pasta = os.path.join(caminho_pasta_dataset, nome_pasta)

        if not os.path.exists(caminho_pasta):
            print(f"⚠️ ATENÇÃO: A pasta '{caminho_pasta}' não foi encontrada.")
            continue

        arquivos = [f for f in os.listdir(caminho_pasta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📂 Processando pasta '{nome_pasta}': {len(arquivos)} imagens encontradas.")

        for arquivo in arquivos:
            caminho_img = os.path.join(caminho_pasta, arquivo)

            try:
                image = mp.Image.create_from_file(caminho_img)
                detection_result = detector.detect(image)

                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    features = []
                    for lm in landmarks:
                        features.append(lm.x)
                        features.append(lm.y)

                    features.append(rotulo)
                    dados.append(features)
            except Exception as e:
                continue

    # 5. Salvar o DataFrame resultante na pasta data/landmarks/
    if len(dados) > 0:
        
        # Descobre dinamicamente quantos pontos foram extraídos para criar as colunas certas
        num_pontos = (len(dados[0]) - 1) // 2

        colunas = []
        for i in range(num_pontos):
            colunas.append(f'x_{i}')
            colunas.append(f'y_{i}')
        colunas.append('rotulo')

        df = pd.DataFrame(dados, columns=colunas)
        caminho_csv_final = os.path.join(caminho_landmarks_dir, nome_csv_saida)
        df.to_csv(caminho_csv_final, index=False)

        print("\n" + "="*50)
        print(f"✅ Extração finalizada com sucesso para {nome_dataset}!")
        print(f"Total de rostos processados: {len(df)}")
        print(f"Arquivo salvo em: '{caminho_csv_final}'")
        print("="*50)
    else:
        print(f"\n❌ Nenhum rosto foi processado para {nome_dataset}.")

if __name__ == "__main__":
    # Processa o dataset original
    extrair_landmarks_do_dataset("dataset_paralisia_facial", "dataset_landmarks.csv")
    
    # Processa o dataset augmentado
    extrair_landmarks_do_dataset("dataset_paralisia_facial_aug", "dataset_landmarks_aug.csv")