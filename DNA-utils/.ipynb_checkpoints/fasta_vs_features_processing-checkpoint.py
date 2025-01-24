import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def join_fasta_with_features(fasta_records, features_df, label_column):
    """
    Объединяет данные из словаря FASTA и таблицы признаков.

    :param fasta_records: Словарь {ID: ДНК-последовательность}.
    :param features_df: DataFrame с признаками (включает колонку 'Virus GENBANK accession').
    :return: Объединенный DataFrame с sequence, label_column, split и ID.
    """
    # Преобразуем словарь FASTA в DataFrame
    fasta_df = pd.DataFrame({
        'Virus GENBANK accession': fasta_records.keys(),
        'sequence': fasta_records.values()
    })

    # Объединяем с таблицей признаков по колонке 'Virus GENBANK accession'
    joined_df = pd.merge(features_df, fasta_df, on='Virus GENBANK accession', how='inner')

    # Оставляем только нужные колонки
    return joined_df[['sequence', label_column, 'split', 'Isolate ID']]