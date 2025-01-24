import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

import numpy as np

def draw_kmeans_plot(virus_data_with_embs, feature, k_min = 2, k_max = 40, step = 2):
    """
    Выполняет кластеризацию K-Means с разным количеством кластеров и вычисляет метрики качества.

    Параметры:
    - virus_data_with_embs (pd.DataFrame): DataFrame, содержащий эмбеддинги и истинные метки.
    - feature (str): Название колонки с истинными метками классов.
    - k_min (int): Минимальное количество кластеров для K-Means.
    - k_max (int): Максимальное количество кластеров для K-Means.

    Возвращает:
    - optimal_k (int): Оптимальное количество кластеров на основе Silhouette Score.
    """
    # Проверка наличия необходимых колонок
    if 'embeds' not in virus_data_with_embs.columns:
        raise ValueError("DataFrame должен содержать колонку 'embeds' с эмбеддингами.")
    if feature not in virus_data_with_embs.columns:
        raise ValueError(f"DataFrame должен содержать колонку '{feature}' с истинными метками.")

    # Извлечение эмбеддингов и истинных меток
    X = np.stack(virus_data_with_embs['embeds'].values)
    true_labels = virus_data_with_embs[feature].values

    print("Число уникальных меток", len(set(true_labels)))

    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Применение PCA для 2D-визуализации
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(X_scaled)
    virus_data_with_embs['x'] = reduced_embeddings[:, 0]
    virus_data_with_embs['y'] = reduced_embeddings[:, 1]

    # Подготовка списков для хранения метрик
    ari_scores = []
    nmi_scores = []
    silhouette_scores = []

    # Диапазон значений k
    k_values = range(k_min, k_max + 1, step)

    print("Выполнение кластеризации K-Means и вычисление метрик:")
    for k in k_values:
        # Кластеризация K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Вычисление метрик
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)

        # Сохранение метрик
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        silhouette_scores.append(silhouette_avg)

        print(f"K={k}: ARI={ari:.4f}, NMI={nmi:.4f}, Silhouette Score={silhouette_avg:.4f}")

    # Визуализация метрик
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(k_values, ari_scores, marker='o')
    plt.title('Adjusted Rand Index (ARI) vs K')
    plt.xlabel('Количество кластеров K')
    plt.ylabel('ARI')
    plt.xticks(k_values)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(k_values, nmi_scores, marker='o', color='orange')
    plt.title('Normalized Mutual Information (NMI) vs K')
    plt.xlabel('Количество кластеров K')
    plt.ylabel('NMI')
    plt.xticks(k_values)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(k_values, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score vs K')
    plt.xlabel('Количество кластеров K')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # # Определение оптимального k по максимальному Silhouette Score
    # optimal_k = k_values[np.argmax(silhouette_scores)]
    # print(f"Оптимальное количество кластеров по Silhouette Score: K={optimal_k}")

    # # Кластеризация с оптимальным k
    # kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    # cluster_labels_opt = kmeans_opt.fit_predict(X_scaled)

    # # Добавление меток кластеров в DataFrame
    # virus_data_with_embs['cluster'] = cluster_labels_opt

    # # Визуализация кластеров
    # plt.figure(figsize=(8, 6))
    # unique_clusters = np.unique(cluster_labels_opt)
    # colors = plt.cm.get_cmap('tab10', len(unique_clusters))

    # for cluster in unique_clusters:
    #     subset = virus_data_with_embs[virus_data_with_embs['cluster'] == cluster]
    #     plt.scatter(subset['x'], subset['y'], label=f'Кластер {cluster}', alpha=0.6, s=50)

    # plt.title(f'K-Means Clustering with K={optimal_k}')
    # plt.xlabel('PCA Dimension 1')
    # plt.ylabel('PCA Dimension 2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # return optimal_k