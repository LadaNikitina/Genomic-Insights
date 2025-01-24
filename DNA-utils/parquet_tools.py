import pandas as pd

def save_to_parquet(data, file_path):
    """
    Сохраняет список словарей в Parquet.

    :param data: Список словарей.
    :param file_path: Путь к Parquet-файлу.
    """
    df = pd.DataFrame(data)
    df.to_parquet(file_path, index=False)

def load_from_parquet(file_path):
    """
    Загружает данные из Parquet.

    :param file_path: Путь к Parquet-файлу.
    :return: DataFrame с данными.
    """
    return pd.read_parquet(file_path)