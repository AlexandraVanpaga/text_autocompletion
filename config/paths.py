"""
Пути к директориям и файлам проекта.
"""

import os

# Базовая директория проекта
BASE_DIR = r'C:\Users\Alexandra\Desktop\text_autocompletion'

PATHS = {
    # Данные
    'raw_data': os.path.join(BASE_DIR, 'data', 'raw_dataset.csv'),
    'clean_data': os.path.join(BASE_DIR, 'data', 'clean_dataset.csv'),
    'split_dir': os.path.join(BASE_DIR, 'data', 'split'),
    
    # Datasets
    'train_dataset': os.path.join(BASE_DIR, 'data', 'split', 'train_dataset.pkl'),
    'val_dataset': os.path.join(BASE_DIR, 'data', 'split', 'val_dataset.pkl'),
    'test_dataset': os.path.join(BASE_DIR, 'data', 'split', 'test_dataset.pkl'),
    'tokenizer': os.path.join(BASE_DIR, 'data', 'split', 'tokenizer.pkl'),
    
    # Модели
    'models_dir': os.path.join(BASE_DIR, 'models'),
    'best_model': os.path.join(BASE_DIR, 'models', 'best_lstm_model.pt'),


    # Результаты
    'results_dir': os.path.join(BASE_DIR, 'results'),
    'test_examples': os.path.join(BASE_DIR, 'models', 'test_generation_examples.json'),
    'gpt2_examples': os.path.join(BASE_DIR, 'models', 'gpt2_generation_examples.json'),
}

