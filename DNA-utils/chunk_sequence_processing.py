import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def create_chunked_dataset(df, tokenizer, label_column, max_chunk_length=512, max_chunks=10):
    """
    Разбивает последовательности на чанки и создает расширенный датасет.

    :param df: DataFrame с данными (sequence, label_column, split, Isolate ID).
    :param tokenizer: Токенизатор модели.
    :param max_chunk_length: Максимальная длина чанка.
    :param max_chunks: Максимальное количество чанков для длинных последовательностей.
    :return: Новый DataFrame с чанками.
    """
    chunked_data = []

    for _, row in tqdm(df.iterrows()):
        sequence = row['sequence']
        isolate_id = row['Isolate ID']
        split = row['split']
        label = row[label_column]

        # Токенизация последовательности
        tokens = tokenizer.encode(sequence, add_special_tokens=True, truncation=False)
        num_tokens = len(tokens)

        # Разбиение на чанки
        chunks = []
        for i in range(0, num_tokens, max_chunk_length):
            chunks.append(tokens[i:i + max_chunk_length])

        # Случайный выбор max_chunks чанков, если их больше max_chunks
        if len(chunks) > max_chunks:
            chunks = random.sample(chunks, max_chunks)

        for chunk in chunks:
            chunk_tokens = chunk + [0] * (max_chunk_length - len(chunk))  # Паддинг
            chunked_data.append({
                'chunk': torch.tensor(chunk_tokens, dtype=torch.long),
                'label': label,
                'split': split,
                'isolate_id': isolate_id
            })

    # Создаем список из чанков
    return chunked_data