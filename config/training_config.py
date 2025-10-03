"""
Конфигурация обучения.
"""

TRAINING_CONFIG = {
    # Гиперпараметры обучения
    'batch_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'num_epochs': 5,
    
    # Оптимизация
    'grad_clip': 5.0,
    'scheduler_factor': 0.5,
    'scheduler_patience': 2,
    
    # Данные
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'random_state': 42,
    
    # DataLoader
    'num_workers': 0,
    'pin_memory': True,
    
    # Evaluation
    'num_rouge_samples': 2000,
    'num_examples': 10,
}