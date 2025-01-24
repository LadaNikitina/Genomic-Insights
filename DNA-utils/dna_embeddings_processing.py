from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

from transformers.models.bert.configuration_bert import BertConfig

model_name = "zhihan1996/DNABERT-2-117M"

config = BertConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config).half().to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def compute_embeddings(chunked_data, model, batch_size=4, device="cuda"):
    """
    Вычисляет эмбеддинги для чанков в данных.

    :param chunked_data: Список словарей, где каждый словарь содержит chunk, label, split и isolate_id.
    :param model: Модель (например, DNABERT).
    :param batch_size: Размер батча.
    :param device: Устройство ("cuda" или "cpu").
    :return: Список словарей с эмбеддингами и дополнительной информацией (label, split, isolate_id).
    """
    # Создаем DataLoader для чанков
    class ChunkDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = ChunkDataset(chunked_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Перенос модели на устройство
    model.to(device)
    model.eval()

    embeddings_with_metadata = []

    # Отключение вычисления градиентов
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches"):
            # Извлекаем данные
            input_ids = torch.stack([item for item in batch['chunk']]).to(device)  # (batch_size, seq_length)
            labels = [item for item in batch['label']]
            splits = [item for item in batch['split']]
            isolate_ids = [item for item in batch['isolate_id'].tolist()]

            # Использование смешанной точности (опционально)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=(input_ids != 0))  # (batch_size, seq_length, hidden_size)

            # Получение последнего скрытого слоя
            embeddings = outputs[0]  # (batch_size, seq_length, hidden_size)

            # Среднее пуллирование с учетом маски внимания
            attention_mask = (input_ids != 0).unsqueeze(-1)  # (batch_size, seq_length, 1)
            sum_embeddings = torch.sum(embeddings * attention_mask, dim=1)  # (batch_size, hidden_size)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)  # (batch_size, 1)
            mean_embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)

            # Перенос на CPU и преобразование в NumPy
            mean_embeddings_np = mean_embeddings.cpu().numpy()

            # Сохраняем эмбеддинги с мета-данными
            for i, embedding in enumerate(mean_embeddings_np):
                embeddings_with_metadata.append({
                    'embedding': embedding,
                    'label': labels[i],
                    'split': splits[i],
                    'isolate_id': isolate_ids[i]
                })

    # Опционально: очистка памяти CUDA
    torch.cuda.empty_cache()

    return embeddings_with_metadata