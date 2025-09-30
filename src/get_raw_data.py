import os
import requests

def download_file(url: str, save_path: str):
    # Создаём папку, если её ещё нет
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Скачиваю данные из {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # если ошибка — выбросит исключение

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Файл сохранён: {save_path}")


if __name__ == "__main__":
    url = "https://code.s3.yandex.net/deep-learning/tweets.txt"
    save_path = r"C:\Users\Alexandra\Desktop\text_autocompletion\data\raw_dataset.csv"

    download_file(url, save_path)