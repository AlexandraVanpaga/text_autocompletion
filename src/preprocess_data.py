import re
import pandas as pd

# Пути к файлам
raw_path = r"C:\Users\Alexandra\Desktop\text_autocompletion\data\raw_dataset.csv"
clean_path = r"C:\Users\Alexandra\Desktop\text_autocompletion\data\clean_dataset.csv"

# Читаем текст из raw файла
with open(raw_path, "r", encoding="utf-8") as f:
    text = f.read()

# Разделяем текст на твиты: каждый твит начинается с @username и до следующего юзернейма
tweets_raw = re.split(r'(?<=\n)(?=@\w+\s)|^(?=@\w+\s)', text, flags=re.MULTILINE)

# Убираем пустые строки и лишние пробелы
tweets_cleaned = [tw.strip() for tw in tweets_raw if tw.strip()]

# Функция предобработки твита
def preprocess_text(text: str) -> str:
    # Удаляем username в начале твита (@username )
    text = re.sub(r'^@\w+\s+', '', text)
    # Удаляем ссылки
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text)
    # Lowercase
    text = text.lower()
    # Убираем всё кроме латиницы, цифр и пробелов
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Очищаем твиты
tweets_cleaned = [preprocess_text(tw) for tw in tweets_cleaned if preprocess_text(tw)]

# Создаём DataFrame только с очищенным текстом
df_clean = pd.DataFrame(tweets_cleaned, columns=["text"])

# Сохраняем чистый датасет
df_clean.to_csv(clean_path, index=False, encoding="utf-8")
print(f"✅ Датасет очищен и сохранён в {clean_path}")
print(f"📊 Количество твитов после очистки: {len(df_clean)}")
