import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# 1. Garantir que a pasta assets existe no mesmo nível
os.makedirs('assets', exist_ok=True)

# 2. Mapeamento dos 4 datasets
datasets_config = {
    'original_sem_procrustes': 'data/landmarks/dataset_landmarks.csv',
    'original_procrustes': 'data/landmarks/dataset_landmarks_procrustes.csv',
    'augmentado_sem_procrustes': 'data/landmarks/dataset_landmarks_aug.csv',
    'augmentado_procrustes': 'data/landmarks/dataset_landmarks_aug_procrustes.csv'
}

tabelas_comparativas = []

for nome_config, caminho_csv in datasets_config.items():
    print(f"\n{'='*80}")
    print(f"Processando Dataset: {nome_config} ({caminho_csv})")
    print(f"{'='*80}")
    
    if not os.path.exists(caminho_csv):
        print(f"[Aviso] O arquivo '{caminho_csv}' não foi encontrado. Certifique-se de que ele está no diretório correto.")
        continue
        
    df = pd.read_csv(caminho_csv)
    X = df.drop(columns=['rotulo'])
    y = df['rotulo']
    
    # Divisão idêntica em Treino (70%) e Teste (30%) com random_state fixo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # --- RANDOM FOREST ---
    print("-> Treinando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # --- SVM ---
    print("-> Treinando Support Vector Machine (SVM RBF)...")
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    svm_pipeline.fit(X_train, y_train)
    y_pred_svm = svm_pipeline.predict(X_test)
    
    # --- SALVAR OS MODELOS ---
    # Salva os dois clasificadores para cada variação de dataset/procrustes
    joblib.dump(rf_model, f"modelo_rf_{nome_config}.pkl")
    joblib.dump(svm_pipeline, f"modelo_svm_{nome_config}.pkl")
    print(f"[Sucesso] Modelos salvos para: {nome_config}")
    
    # --- MÉTRICAS ---
    acc_rf = accuracy_score(y_test, y_pred_rf) * 100
    acc_svm = accuracy_score(y_test, y_pred_svm) * 100
    
    # --- MATRIZES DE CONFUSÃO (Salvas em assets/) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
    axes[0].set_title(f"Random Forest (Acurácia: {acc_rf:.2f}%)")
    axes[0].set_ylabel('Rótulo Real')
    axes[0].set_xlabel('Rótulo Previsto')
    
    sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
    axes[1].set_title(f"SVM RBF (Acurácia: {acc_svm:.2f}%)")
    axes[1].set_ylabel('Rótulo Real')
    axes[1].set_xlabel('Rótulo Previsto')
    
    plt.suptitle(f"Matrizes de Confusão - Config: {nome_config}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"assets/matriz_confusao_{nome_config}.png")
    plt.close()
    print(f"[Gráfico] Matriz de confusão salva em: assets/matriz_confusao_{nome_config}.png")
    
    # --- EXTRAÇÃO DE IMPORTÂNCIA DE ATRIBUTOS (Random Forest) ---
    # Geramos o gráfico dos top 10 landmarks para datasets augmentados ou com procrustes
    importancias = rf_model.feature_importances_
    num_pontos = 478
    importancia_pontos = []
    for i in range(num_pontos):
        imp_x = importancias[2 * i]
        imp_y = importancias[2 * i + 1]
        importancia_pontos.append(imp_x + imp_y)
        
    df_importancia = pd.DataFrame({
        'landmark_id': range(num_pontos),
        'importancia': importancia_pontos
    }).sort_values(by='importancia', ascending=False)
    
    top_10 = df_importancia.head(10).sort_values(by='importancia', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh([str(i) for i in top_10['landmark_id']], top_10['importancia'], color='dodgerblue')
    plt.title(f"Top 10 Landmarks Mais Relevantes (RF) - {nome_config}", fontsize=14, fontweight='bold')
    plt.xlabel("Importância Acumulada (X + Y)", fontsize=12)
    plt.ylabel("ID do Landmark do MediaPipe", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"assets/top10_landmarks_{nome_config}.png")
    plt.close()
    print(f"[Gráfico] Gráfico de importância salvo em: assets/top10_landmarks_{nome_config}.png\n")

print("Treinamento e avaliações concluídos para todos os datasets!")