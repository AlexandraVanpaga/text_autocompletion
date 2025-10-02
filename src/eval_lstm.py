"""
Оценка LSTM модели на test set.
"""

import os
import pickle
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rouge_score import rouge_scorer

from src.lstm_model import LSTMTextGenerator
from config import PATHS, MODEL_CONFIG, TRAINING_CONFIG


# =============================================================================
# Загрузка данных и модели
# =============================================================================

print("Загрузка test set...")
with open(PATHS['test_dataset'], 'rb') as f:
    test_dataset = pickle.load(f)
with open(PATHS['tokenizer'], 'rb') as f:
    tokenizer = pickle.load(f)

test_loader = DataLoader(
    test_dataset, 
    batch_size=TRAINING_CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=0
)
print(f"Test samples: {len(test_dataset)}")

# Загружаем модель
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

model.load_state_dict(torch.load(PATHS['best_model']))
model.eval()

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# =============================================================================
# Оценка на test set
# =============================================================================

test_loss = 0
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

print("\nОценка на test set...")

with torch.no_grad():
    for prefix, target, mask in tqdm(test_loader, desc="Testing"):
        prefix, target, mask = prefix.to(device), target.to(device), mask.to(device)
        
        # Loss
        _, hidden = model(prefix)
        decoder_input = torch.cat([prefix[:, -1:], target[:, :-1]], dim=1)
        logits, _ = model(decoder_input, hidden)
        
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = (loss * mask.reshape(-1)).sum() / mask.reshape(-1).sum()
        test_loss += loss.item()
        
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

test_loss /= len(test_loader)
test_perplexity = np.exp(test_loss)

# Средние метрики
avg_rouge1 = np.mean(rouge_scores['rouge1'])
avg_rouge2 = np.mean(rouge_scores['rouge2'])
avg_rougeL = np.mean(rouge_scores['rougeL'])

std_rouge1 = np.std(rouge_scores['rouge1'])
std_rouge2 = np.std(rouge_scores['rouge2'])
std_rougeL = np.std(rouge_scores['rougeL'])

# Результаты
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ НА TEST SET")
print("="*80)
print(f"Test Loss:       {test_loss:.4f}")
print(f"Test Perplexity: {test_perplexity:.2f}")
print(f"\nROUGE Scores (на {len(rouge_scores['rouge1'])} примерах):")
print(f"  ROUGE-1: {avg_rouge1:.4f} (±{std_rouge1:.4f})")
print(f"  ROUGE-2: {avg_rouge2:.4f} (±{std_rouge2:.4f})")
print(f"  ROUGE-L: {avg_rougeL:.4f} (±{std_rougeL:.4f})")
print("="*80)


# =============================================================================
# Примеры генерации
# =============================================================================

print("\nПРИМЕРЫ ГЕНЕРАЦИИ ИЗ TEST SET")
print("="*80)

num_examples = TRAINING_CONFIG['num_examples']
examples_shown = 0
examples_data = []

with torch.no_grad():
    for prefix, target, mask in test_loader:
        if examples_shown >= num_examples:
            break
        
        prefix = prefix.to(device)
        target = target.to(device)
        
        generated = model.generate(
            prefix,
            max_length=target.size(1),
            temperature=MODEL_CONFIG['temperature'],
            top_k=MODEL_CONFIG['top_k'],
            repetition_penalty=MODEL_CONFIG['repetition_penalty'],
            pad_token_id=tokenizer.pad_token_id
        )
        
        for i in range(min(prefix.size(0), num_examples - examples_shown)):
            prefix_text = tokenizer.decode(prefix[i].cpu().tolist(), skip_special_tokens=True)
            target_text = tokenizer.decode(target[i].cpu().tolist(), skip_special_tokens=True)
            generated_text = tokenizer.decode(generated[i].cpu().tolist(), skip_special_tokens=True)
            
            if len(target_text.strip()) > 0 and len(generated_text.strip()) > 0:
                scores = scorer.score(target_text, generated_text)
                
                print(f"\nПример {examples_shown + 1}:")
                print(f"Префикс: {prefix_text}")
                print(f"Ground Truth: {target_text}")
                print(f"Генерация: {generated_text}")
                print(f"ROUGE: R1={scores['rouge1'].fmeasure:.3f}, "
                      f"R2={scores['rouge2'].fmeasure:.3f}, "
                      f"RL={scores['rougeL'].fmeasure:.3f}")
                print("─"*80)
                
                # Сохраняем как словарь
                examples_data.append({
                    'example_num': examples_shown + 1,
                    'prefix': prefix_text,
                    'ground_truth': target_text,
                    'generated': generated_text,
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure
                })
                
                examples_shown += 1
            
            if examples_shown >= num_examples:
                break

# Сохраняем примеры в JSON
with open(PATHS['test_examples'], 'w', encoding='utf-8') as f:
    json.dump(examples_data, f, ensure_ascii=False, indent=2)

print(f"\nПримеры сохранены в: {PATHS['test_examples']}")
print("Оценка завершена")