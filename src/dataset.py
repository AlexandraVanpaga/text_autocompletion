import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from transformers import BertTokenizerFast


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
        
        # Разделяем на prefix (75%) и target (25%)
        target_split = int(seq_len * 0.75)
        split_point = min(word_starts, key=lambda x: abs(x - target_split))
        split_point = max(split_point, 1)
        
        prefix = tokens[:split_point]
        target = tokens[split_point:]
        
        prefix_len = int(self.max_length * 0.75)
        target_len = self.max_length - prefix_len
        
        # Паддинг/обрезка prefix
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
        
        # Паддинг/обрезка target
        original_target_len = len(target)
        if len(target) < target_len:
            target = target + [self.tokenizer.pad_token_id] * (target_len - len(target))
        else:
            target = target[:target_len]
        
        # Создаём attention mask
        attention_mask = [1] * min(original_target_len, target_len) + [0] * max(0, target_len - original_target_len)
        
        return (
            torch.tensor(prefix, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float)
        )