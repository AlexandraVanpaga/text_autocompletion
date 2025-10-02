"""
Предобработка и очистка текстовых данных.
"""

import re
import pandas as pd

from config import PATHS


def preprocess_text(text: str) -> str:
    """
    Функция предобработки твита.
    
    Args:
        text: исходный текст твита
        
    Returns:
        очищенный текст
    """
    # Удаляем юзернеймы
    text = re.sub(r'@\w+\s+', '', text)
    # Удаляем ссылки
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text)
    # Lowercase
    text = text.lower()
    # Убираем всё кроме латиницы, цифр, важных знаков препинания и пробелов
    text = re.sub(r"[^a-z0-9\s\.,!?\']", "", text)
    # Нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_split_tweets(raw_path: str) -> pd.DataFrame:
    """
    Загружает raw файл и разделяет на отдельные твиты.
    
    Args:
        raw_path: путь к исходному файлу
        
    Returns:
        DataFrame с колонкой 'text'
    """
    # Читаем текст из файла
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Разделяем текст на твиты: каждый твит начинается с @username
    tweets_raw = re.split(r'(?<=\n)(?=@\w+\s)|^(?=@\w+\s)', text, flags=re.MULTILINE)
    
    # Убираем пустые строки и лишние пробелы
    tweets_cleaned = [tw.strip() for tw in tweets_raw if tw.strip()]
    
    # Создаём DataFrame
    df = pd.DataFrame(tweets_cleaned, columns=["text"])
    
    print(f"Количество твитов после разделения: {len(df)}")
    print("\nПервые 5 твитов (исходные):")
    print(df.head())
    
    return df


def clean_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает твиты с помощью предобработки.
    
    Args:
        df: DataFrame с колонкой 'text'
        
    Returns:
        DataFrame с добавленной колонкой 'text_clean'
    """
    # Применяем предобработку
    df["text_clean"] = df["text"].apply(preprocess_text)
    
    # Убираем пустые строки после очистки
    df = df[df["text_clean"].str.strip() != ""]
    
    print(f"\nКоличество твитов после очистки: {len(df)}")
    print("\nПервые 5 твитов (очищенные):")
    print(df[["text_clean"]].head())
    
    return df


def main():
    """Основная функция для запуска процесса очистки датасета."""
    
    print("=" * 60)
    print("Начинаем обработку датасета")
    print("=" * 60)
    
    # Шаг 1: Загружаем и разделяем твиты
    df = load_and_split_tweets(PATHS['raw_data'])
    
    # Шаг 2: Очищаем твиты
    df_clean = clean_tweets(df)
    
    # Шаг 3: Сохраняем только очищенный текст
    df_final = df_clean[["text_clean"]].rename(columns={"text_clean": "text"})
    df_final.to_csv(PATHS['clean_data'], index=False, encoding="utf-8")
    
    print("\n" + "=" * 60)
    print(f"✓ Датасет успешно очищен и сохранён в:\n  {PATHS['clean_data']}")
    print(f"✓ Итоговое количество твитов: {len(df_final)}")
    print("=" * 60)


if __name__ == "__main__":
    main()