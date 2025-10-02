"""
LSTM модель для генерации продолжения текста.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CONFIG


class LSTMTextGenerator(nn.Module):
    """
    LSTM модель для генерации продолжения текста (1/4 последовательности).
    """
    
    def __init__(
        self,
        vocab_size: int = None,
        embedding_dim: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        pad_token_id: int = None
    ):
        """
        Args:
            vocab_size: размер словаря (30522 для BERT)
            embedding_dim: размерность embedding'ов
            hidden_size: размер hidden state LSTM
            num_layers: количество LSTM слоёв
            dropout: dropout rate
            pad_token_id: ID токена padding
        """
        super(LSTMTextGenerator, self).__init__()
        
        # Используем значения из конфига, если параметры не переданы
        self.vocab_size = vocab_size or MODEL_CONFIG['vocab_size']
        self.embedding_dim = embedding_dim or MODEL_CONFIG['embedding_dim']
        self.hidden_size = hidden_size or MODEL_CONFIG['hidden_size']
        self.num_layers = num_layers or MODEL_CONFIG['num_layers']
        dropout = dropout if dropout is not None else MODEL_CONFIG['dropout']
        self.pad_token_id = pad_token_id if pad_token_id is not None else MODEL_CONFIG['pad_token_id']
        
        # Embedding слой
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.pad_token_id
        )
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout перед выходным слоем
        self.dropout = nn.Dropout(dropout)
        
        # Выходной fully connected слой
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов модели умным способом - через Xavier."""
        # Embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # FC layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] - входные токены
            hidden: опциональное начальное hidden state
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] - logits для каждой позиции
            hidden: (h_n, c_n) - финальное hidden state
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        
        # LSTM: [batch_size, seq_len, hidden_size]
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # FC layer: [batch_size, seq_len, vocab_size]
        logits = self.fc(lstm_out)
        
        return logits, hidden
    
    def generate(
        self,
        prefix: torch.Tensor,
        max_length: int,
        temperature: float = None,
        top_k: int = None,
        repetition_penalty: float = None,
        pad_token_id: int = None,
        eos_token_id: int = 102  # [SEP] token для BERT
    ) -> torch.Tensor:
        """
        Генерирует продолжение текста с защитой от повторений.
        
        Args:
            prefix: [batch_size, prefix_len] - префикс (3/4 текста)
            max_length: длина генерации (1/4 текста)
            temperature: температура для sampling (выше = разнообразнее)
            top_k: размер top-k filtering
            repetition_penalty: штраф за повторения (>1 = сильнее штраф)
            pad_token_id: ID padding токена
            eos_token_id: ID токена конца последовательности
        
        Returns:
            generated: [batch_size, max_length] - сгенерированные токены
        """
        # Используем значения из конфига, если параметры не переданы
        temperature = temperature if temperature is not None else MODEL_CONFIG['temperature']
        top_k = top_k if top_k is not None else MODEL_CONFIG['top_k']
        repetition_penalty = repetition_penalty if repetition_penalty is not None else MODEL_CONFIG['repetition_penalty']
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        
        self.eval()
        batch_size = prefix.size(0)
        device = prefix.device
        
        # Получаем hidden state от префикса (3/4 текста)
        with torch.no_grad():
            _, hidden = self.forward(prefix)
        
        # Инициализируем генерацию с последнего токена префикса
        current_token = prefix[:, -1].unsqueeze(1)  # [batch_size, 1]
        generated = []
        
        # Словарь для отслеживания сгенерированных токенов (для repetition penalty)
        generated_tokens = {i: [] for i in range(batch_size)}
        
        for _ in range(max_length):
            with torch.no_grad():
                # Forward pass для текущего токена
                logits, hidden = self.forward(current_token, hidden)  # [batch_size, 1, vocab_size]
                logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Применяем temperature
                logits = logits / temperature
                
                # Repetition penalty (защита от повторений)
                for i in range(batch_size):
                    for token in generated_tokens[i]:
                        logits[i, token] /= repetition_penalty
                
                # Top-k filtering (оставляем только топ-k токенов)
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_values)
                    logits = logits_filtered
                
                # Softmax для получения вероятностей
                probs = F.softmax(logits, dim=-1)
                
                # Sampling (случайный выбор с учётом вероятностей)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                
                # Сохраняем токен
                generated.append(next_token)
                
                # Обновляем историю для repetition penalty
                for i in range(batch_size):
                    token_id = next_token[i].item()
                    if token_id != pad_token_id:
                        generated_tokens[i].append(token_id)
                
                # Следующий токен становится текущим
                current_token = next_token
        
        # Собираем результат
        generated = torch.cat(generated, dim=1)  # [batch_size, max_length]
        
        return generated
    
    def get_num_params(self) -> int:
        """Возвращает количество обучаемых параметров модели."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)