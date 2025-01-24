import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def prepare_data(chunked_dataset):
    """
    Подготовка данных для обучения.

    :param chunked_dataset: Список словарей с эмбеддингами, метками, сплитами.
    :return: Данные для train, val и test.
    """
    # Преобразуем в DataFrame для удобства
    df = pd.DataFrame(chunked_dataset)

    # Преобразуем метки в числовой формат
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    # Разделение на train, val и test
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'val']
    test_data = df[df['split'] == 'test']

    # Получаем эмбеддинги и метки
    X_train = np.vstack(train_data['embedding'])
    y_train = train_data['label_encoded']

    X_val = np.vstack(val_data['embedding'])
    y_val = val_data['label_encoded']

    X_test = np.vstack(test_data['embedding'])
    y_test = test_data['label_encoded']

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder

def train_catboost(X_train, y_train, X_val, y_val):
    """
    Обучение CatBoostClassifier.

    :param X_train: Матрица признаков для обучения.
    :param y_train: Метки классов для обучения.
    :param X_val: Матрица признаков для валидации.
    :param y_val: Метки классов для валидации.
    :return: Обученная модель.
    """
    model = CatBoostClassifier(
        iterations=2_000,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        verbose=50,
        task_type="GPU"
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=20)
    return model

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Обучение LogisticRegression.

    :param X_train: Матрица признаков для обучения.
    :param y_train: Метки классов для обучения.
    :param X_val: Матрица признаков для валидации.
    :param y_val: Метки классов для валидации.
    :return: Обученная модель.
    """
    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Оценка модели.

    :param model: Обученная модель.
    :param X_test: Матрица признаков тестового набора.
    :param y_test: Метки классов тестового набора.
    :param label_encoder: Объект LabelEncoder для декодирования меток.
    """
    print("Классы в y_test:", np.unique(y_test))
    print("Классы в label_encoder:", label_encoder.classes_)

    y_pred = model.predict(X_test)

    # Для CatBoost, результат может быть в виде вероятностей
    if hasattr(model, 'predict_proba'):
        y_pred = np.argmax(model.predict_proba(X_test), axis=1)

    print("Классификационный отчет:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))