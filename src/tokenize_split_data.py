"""
Токенизация и разбиение данных на train/val/test.
"""

import os
import pickle
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

from config import PATHS, MODEL_CONFIG, TRAINING_CONFIG



# Dataset с Fast Tokenizer


class TextGenerationDataset(Dataset):
    """
    Dataset для генерации текста с поддержкой быстрой токенизации.
    Разбивает текст на prefix и target для обучения модели автодополнения.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: BertTokenizerFast,
        max_length: int = 64,
        min_length: int = 16
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
        self.samples = []
        self.word_boundaries = []
        
        # Батчевая токенизация - быстрее
        print("Токенизация батчами...")
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Батчевая токенизация
            encodings = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                return_tensors=None
            )
            
            for tokens in encodings['input_ids']:
                if len(tokens) >= min_length:
                    # Находим границы слов
                    tokenized_text = self.tokenizer.convert_ids_to_tokens(tokens)
                    word_starts = [0]
                    
                    for j, token in enumerate(tokenized_text[1:], 1):
                        if not token.startswith('##') and token != '[SEP]':
                            word_starts.append(j)
                    
                    if len(tokens) - 1 not in word_starts:
                        word_starts.append(len(tokens) - 1)
                    
                    self.samples.append(tokens)
                    self.word_boundaries.append(word_starts)
        
        print(f"Загружено {len(self.samples)} строк")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.samples[idx]
        word_starts = self.word_boundaries[idx]
        seq_len = len(tokens)
        
        target_split = int(seq_len * 0.75)
        split_point = min(word_starts, key=lambda x: abs(x - target_split))
        split_point = max(split_point, 1)
        
        prefix = tokens[:split_point]
        target = tokens[split_point:]
        
        prefix_len = int(self.max_length * 0.75)
        target_len = self.max_length - prefix_len
        
        if len(prefix) < prefix_len:
            prefix = prefix + [self.tokenizer.pad_token_id] * (prefix_len - len(prefix))
        elif len(prefix) > prefix_len:
            valid_boundaries = [w for w in word_starts if w <= prefix_len]
            if valid_boundaries:
                cut_point = max(valid_boundaries)
                prefix = tokens[:cut_point]
                target = tokens[cut_point:]
                if len(prefix) < prefix_len:
                    prefix = prefix + [self.tokenizer.pad_token_id] * (prefix_len - len(prefix))
        
        original_target_len = len(target)
        if len(target) < target_len:
            target = target + [self.tokenizer.pad_token_id] * (target_len - len(target))
        else:
            target = target[:target_len]
        
        attention_mask = [1] * min(original_target_len, target_len) + [0] * max(0, target_len - original_target_len)
        
        return (
            torch.tensor(prefix, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float)
        )



# Функции для работы с данными


def split_data(
    df: pd.DataFrame,
    text_column: str = 'text'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разбивает DataFrame на train, validation и test выборки.
    """
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



def create_and_save_datasets(
    df: pd.DataFrame,
    text_column: str = 'text',
    device: torch.device = None
):
    """
    Создаёт datasets, dataloaders и сохраняет их на диск.
    
    Args:
        df: DataFrame с текстами
        text_column: название колонки с текстом
        device: torch.device объект (не строка!)
    """
   
    
    # Создаём директорию
    os.makedirs(PATHS['split_dir'], exist_ok=True)
    
    # Создаем torch.device объект
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU")
    
    print("\nРазбиение данных...")
    train_df, val_df, test_df = split_data(df, text_column)
    
    print("\nЗагрузка Fast токенизатора...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    print("\nСоздание datasets...")
    train_texts = train_df[text_column].tolist()
    val_texts = val_df[text_column].tolist()
    test_texts = test_df[text_column].tolist()
    
    print("Train dataset:")
    train_dataset = TextGenerationDataset(
        train_texts, 
        tokenizer, 
        MODEL_CONFIG['max_length'], 
        MODEL_CONFIG['min_length']
    )
    
    print("Val dataset:")
    val_dataset = TextGenerationDataset(
        val_texts, 
        tokenizer, 
        MODEL_CONFIG['max_length'], 
        MODEL_CONFIG['min_length']
    )
    
    print("Test dataset:")
    test_dataset = TextGenerationDataset(
        test_texts, 
        tokenizer, 
        MODEL_CONFIG['max_length'], 
        MODEL_CONFIG['min_length']
    )
    
    print("\nСоздание dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda' and TRAINING_CONFIG['pin_memory'])
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda' and TRAINING_CONFIG['pin_memory'])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda' and TRAINING_CONFIG['pin_memory'])
    )
    
    print("\nСохранение...")
    with open(PATHS['train_dataset'], 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open(PATHS['val_dataset'], 'wb') as f:
        pickle.dump(val_dataset, f)
    
    with open(PATHS['test_dataset'], 'wb') as f:
        pickle.dump(test_dataset, f)
    
    with open(PATHS['tokenizer'], 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"✓ Сохранено в: {PATHS['split_dir']}")
    
    return train_loader, val_loader, test_loader, tokenizer, device


# Главная функция


def main():
    """Основная функция для создания и сохранения datasets."""
    
    print("=" * 60)
    print("Создание datasets для обучения модели")
    print("=" * 60)
    
    print(f"\nЗагрузка данных из: {PATHS['clean_data']}")
    df = pd.read_csv(PATHS['clean_data'], encoding='utf-8')
    print(f"Загружено строк: {len(df)}")
    
    # Передаем torch.device объект 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, tokenizer, device = create_and_save_datasets(
        df=df,
        text_column='text',
        device=device  
    )
 
    
    print("\n" + "=" * 60)
    print("✓ Готово!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Device: {device}")
    print("=" * 60)


if __name__ == "__main__":
    main()