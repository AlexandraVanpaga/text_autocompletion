"""
Обучение LSTM модели с ROUGE метриками.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from rouge_score import rouge_scorer

from lstm_model import LSTMTextGenerator
from config import PATHS, MODEL_CONFIG, TRAINING_CONFIG


# =============================================================================
# Загрузка данных
# =============================================================================

os.makedirs(PATHS['models_dir'], exist_ok=True)

print("Загрузка данных...")
with open(PATHS['train_dataset'], 'rb') as f:
    train_dataset = pickle.load(f)
with open(PATHS['val_dataset'], 'rb') as f:
    val_dataset = pickle.load(f)
with open(PATHS['tokenizer'], 'rb') as f:
    tokenizer = pickle.load(f)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=TRAINING_CONFIG['batch_size'], 
    shuffle=True, 
    num_workers=TRAINING_CONFIG['num_workers']
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=TRAINING_CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=TRAINING_CONFIG['num_workers']
)


# =============================================================================
# Создание модели
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = LSTMTextGenerator(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=MODEL_CONFIG['embedding_dim'],
    hidden_size=MODEL_CONFIG['hidden_size'],
    num_layers=MODEL_CONFIG['num_layers'],
    dropout=MODEL_CONFIG['dropout'],
    pad_token_id=tokenizer.pad_token_id
).to(device)

print(f"Параметров: {model.get_num_params():,}\n")


# =============================================================================
# Оптимизация
# =============================================================================

optimizer = optim.Adam(
    model.parameters(), 
    lr=TRAINING_CONFIG['learning_rate'], 
    weight_decay=TRAINING_CONFIG['weight_decay']
)
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=TRAINING_CONFIG['scheduler_factor'], 
    patience=TRAINING_CONFIG['scheduler_patience']
)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# =============================================================================
# Обучение
# =============================================================================

best_val_loss = float('inf')

for epoch in range(1, TRAINING_CONFIG['num_epochs'] + 1):
    # Train
    model.train()
    train_loss = 0
    for prefix, target, mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
        prefix, target, mask = prefix.to(device), target.to(device), mask.to(device)
        
        optimizer.zero_grad()
        _, hidden = model(prefix)
        decoder_input = torch.cat([prefix[:, -1:], target[:, :-1]], dim=1)
        logits, _ = model(decoder_input, hidden)
        
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = (loss * mask.reshape(-1)).sum() / mask.reshape(-1).sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Val с ROUGE
    model.eval()
    val_loss = 0
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    with torch.no_grad():
        for prefix, target, mask in val_loader:
            prefix, target, mask = prefix.to(device), target.to(device), mask.to(device)
            
            # Loss
            _, hidden = model(prefix)
            decoder_input = torch.cat([prefix[:, -1:], target[:, :-1]], dim=1)
            logits, _ = model(decoder_input, hidden)
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss = (loss * mask.reshape(-1)).sum() / mask.reshape(-1).sum()
            val_loss += loss.item()
            
            # ROUGE
            if len(rouge_scores['rouge1']) < TRAINING_CONFIG['num_rouge_samples']:
                generated = model.generate(
                    prefix[:32],
                    max_length=target.size(1),
                    temperature=MODEL_CONFIG['temperature'],
                    top_k=MODEL_CONFIG['top_k'],
                    repetition_penalty=MODEL_CONFIG['repetition_penalty'],
                    pad_token_id=tokenizer.pad_token_id
                )
                
                for i in range(min(32, prefix.size(0))):
                    target_text = tokenizer.decode(target[i].cpu().tolist(), skip_special_tokens=True)
                    generated_text = tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True)
                    
                    if len(target_text.strip()) > 0 and len(generated_text.strip()) > 0:
                        scores = scorer.score(target_text, generated_text)
                        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    val_loss /= len(val_loader)
    perplexity = np.exp(val_loss)
    scheduler.step(val_loss)
    
    # Средние ROUGE
    avg_rouge1 = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0
    avg_rouge2 = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0
    avg_rougeL = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0
    
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, PPL={perplexity:.2f}")
    print(f"  ROUGE-1={avg_rouge1:.4f}, ROUGE-2={avg_rouge2:.4f}, ROUGE-L={avg_rougeL:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), PATHS['best_model'])
        print("  Модель сохранена")

print("\nОбучение завершено!")