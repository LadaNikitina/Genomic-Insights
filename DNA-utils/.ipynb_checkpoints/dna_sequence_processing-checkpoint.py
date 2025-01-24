import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length):
        """
        Инициализация датасета.

        :param sequences: Список последовательностей.
        :param tokenizer: Токенизатор (например, из Hugging Face Transformers).
        :param max_length: Максимальная длина последовательности после токенизации.
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Получение одного элемента датасета.

        :param idx: Индекс последовательности.
        :return: Словарь с `input_ids` и `attention_mask`.
        """
        sequence = self.sequences[idx]
        
        # Токенизация без усечения
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',  # Паддинг до max_length
            max_length=self.max_length,
            truncation=False  # Отключаем автоматическое усечение
        )
        input_ids = encoding['input_ids'].squeeze(0)  # Удаляем лишнее измерение
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Если длина последовательности превышает max_length, выбираем случайный кусочек
        if input_ids.size(0) > self.max_length:
            # Вычисляем допустимый диапазон начала окна
            max_start = input_ids.size(0) - self.max_length
            start_idx = random.randint(0, max_start)
            input_ids = input_ids[start_idx:start_idx + self.max_length]
            attention_mask = attention_mask[start_idx:start_idx + self.max_length]
        # Если длина меньше max_length, паддинг уже применен

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }