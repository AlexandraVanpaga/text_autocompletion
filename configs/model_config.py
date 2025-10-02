"""
Конфигурация модели LSTM.
"""

MODEL_CONFIG = {
    # Архитектура модели
    'vocab_size': 30522,  # BERT tokenizer
    'embedding_dim': 256,
    'hidden_size': 512,
    'num_layers': 2,
    'dropout': 0.2,
    'pad_token_id': 0,
    
    # Параметры последовательности
    'max_length': 64,
    'min_length': 16,
    
    # Параметры генерации
    'temperature': 0.8,
    'top_k': 50,
    'repetition_penalty': 1.2,
}