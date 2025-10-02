"""
Токенизация и разбиение данных на train/val/test.
"""

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

from config import PATHS, MODEL_CONFIG, TRAINING_CONFIG
from src.dataset import TextGenerationDataset


# =============================================================================
# Функции для работы с данными
# =============================================================================

def split_data(df: pd.DataFrame, text_column: str = 'text'):
    df_shuffled = df.sample(frac=1, random_state=TRAINING_CONFIG['random_state']).reset_index(drop=True)
    
    train_df, temp_df = train_test_split(
        df_shuffled,
        test_size=(TRAINING_CONFIG['val_ratio'] + TRAINING_CONFIG['test_ratio']),
        random_state=TRAINING_CONFIG['random_state']
    )
    
    val_size_adjusted = TRAINING_CONFIG['val_ratio'] / (TRAINING_CONFIG['val_ratio'] + TRAINING_CONFIG['test_ratio'])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_adjusted),
        random_state=TRAINING_CONFIG['random_state']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def create_and_save_datasets(df: pd.DataFrame, text_column: str = 'text', device: str = 'cuda'):
    os.makedirs(PATHS['split_dir'], exist_ok=True)
    
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CPU")
    
    print("\nРазбиение данных...")
    train_df, val_df, test_df = split_data(df, text_column)
    
    print("\nЗагрузка Fast токенизатора...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    print("\nСоздание datasets...")
    train_dataset = TextGenerationDataset(train_df[text_column].tolist(), tokenizer, MODEL_CONFIG['max_length'], MODEL_CONFIG['min_length'])
    val_dataset = TextGenerationDataset(val_df[text_column].tolist(), tokenizer, MODEL_CONFIG['max_length'], MODEL_CONFIG['min_length'])
    test_dataset = TextGenerationDataset(test_df[text_column].tolist(), tokenizer, MODEL_CONFIG['max_length'], MODEL_CONFIG['min_length'])
    
    print("\nСоздание dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True,
                              num_workers=TRAINING_CONFIG['num_workers'], pin_memory=(device.type == 'cuda' and TRAINING_CONFIG['pin_memory']))
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False,
                            num_workers=TRAINING_CONFIG['num_workers'], pin_memory=(device.type == 'cuda' and TRAINING_CONFIG['pin_memory']))
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False,
                             num_workers=TRAINING_CONFIG['num_workers'], pin_memory=(device.type == 'cuda' and TRAINING_CONFIG['pin_memory']))
    
    print("\nСохранение...")
    with open(PATHS['train_dataset'], 'wb') as f: pickle.dump(train_dataset, f)
    with open(PATHS['val_dataset'], 'wb') as f: pickle.dump(val_dataset, f)
    with open(PATHS['test_dataset'], 'wb') as f: pickle.dump(test_dataset, f)
    with open(PATHS['tokenizer'], 'wb') as f: pickle.dump(tokenizer, f)
    
    print(f"✓ Сохранено в: {PATHS['split_dir']}")
    return train_loader, val_loader, test_loader, tokenizer, device


def main():
    print("=" * 60)
    print("Создание datasets для обучения модели")
    print("=" * 60)
    
    print(f"\nЗагрузка данных из: {PATHS['clean_data']}")
    df = pd.read_csv(PATHS['clean_data'], encoding='utf-8')
    print(f"Загружено строк: {len(df)}")
    
    train_loader, val_loader, test_loader, tokenizer, device = create_and_save_datasets(df=df, text_column='text', device='cuda')
    
    print("\n" + "=" * 60)
    print("✓ Готово!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Device: {device}")
    print("=" * 60)


if __name__ == "__main__":
    main()