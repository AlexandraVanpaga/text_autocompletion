"""
Сравнение с baseline: DistilGPT2.
"""

import os
import pickle
import json
from src.dataset import TextGenerationDataset 
import numpy as np
import torch
import torch.nn as nn
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PATHS, MODEL_CONFIG, TRAINING_CONFIG


# =============================================================================
# Параметры для GPT2
# =============================================================================

gpt2_params = {
    'model_name': 'distilgpt2',
    'num_samples': 100,
    'max_new_tokens': 10,
    'split_ratio': 0.75
}


# =============================================================================
# Загрузка данных
# =============================================================================

print("Загрузка данных...")
with open(PATHS['val_dataset'], 'rb') as f:
    val_dataset = pickle.load(f)
with open(PATHS['tokenizer'], 'rb') as f:
    tokenizer = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =============================================================================
# Загрузка DistilGPT2
# =============================================================================

print("\nЗагрузка DistilGPT2...")
tokenizer_gpt = AutoTokenizer.from_pretrained(gpt2_params['model_name'])
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

model_gpt = AutoModelForCausalLM.from_pretrained(gpt2_params['model_name']).to(device)

generator = pipeline(
    "text-generation",
    model=model_gpt,
    tokenizer=tokenizer_gpt,
    device=0 if torch.cuda.is_available() else -1
)

rouge = evaluate.load("rouge")
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_gpt.pad_token_id, reduction='none')


# =============================================================================
# Функция генерации
# =============================================================================

def generate_continuation(text, max_new_tokens=10):
    """Генерирует продолжение текста (последние 25%)."""
    words = text.split()
    split_idx = int(len(words) * gpt2_params['split_ratio'])
    
    input_text = " ".join(words[:split_idx])
    target_text = " ".join(words[split_idx:])
    
    result = generator(
        input_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=MODEL_CONFIG['top_k'],
        temperature=MODEL_CONFIG['temperature']
    )
    
    generated_full = result[0]["generated_text"]
    generated_tail = generated_full[len(input_text):].strip()
    
    return input_text, target_text, generated_tail


# =============================================================================
# Подсчет loss на validation set
# =============================================================================

print("\nПодсчет loss на validation set...")
val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False, num_workers=0)

model_gpt.eval()
val_loss = 0

with torch.no_grad():
    for prefix, target, mask in tqdm(val_loader, desc="Computing loss"):
        # Токенизируем prefix для GPT2
        prefix_text_batch = [tokenizer.decode(p.tolist(), skip_special_tokens=True) for p in prefix]
        target_text_batch = [tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in target]
        
        # Объединяем prefix + target для GPT2
        full_text_batch = [p + " " + t for p, t in zip(prefix_text_batch, target_text_batch)]
        
        # Токенизируем для GPT2
        inputs = tokenizer_gpt(full_text_batch, return_tensors='pt', padding=True, truncation=True, max_length=MODEL_CONFIG['max_length'])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = model_gpt(**inputs, labels=inputs['input_ids'])
        val_loss += outputs.loss.item()

val_loss /= len(val_loader)
val_perplexity = np.exp(val_loss)

print(f"\nGPT2 Loss: {val_loss:.4f}")
print(f"GPT2 Perplexity: {val_perplexity:.2f}")


# =============================================================================
# Оценка ROUGE на validation set
# =============================================================================

print("\nОценка ROUGE на validation set...")

num_samples = gpt2_params['num_samples']
sample_indices = np.random.choice(len(val_dataset), num_samples, replace=False)

all_preds, all_refs = [], []

for idx in tqdm(sample_indices, desc="DistilGPT2 Generation"):
    prefix, target, _ = val_dataset[idx]
    
    prefix_text = tokenizer.decode(prefix.tolist(), skip_special_tokens=True)
    target_text = tokenizer.decode(target.tolist(), skip_special_tokens=True)
    full_text = prefix_text + " " + target_text
    
    _, gt, pred = generate_continuation(full_text, max_new_tokens=gpt2_params['max_new_tokens'])
    
    all_preds.append(pred)
    all_refs.append(gt)

# ROUGE метрики
gpt_rouge = rouge.compute(predictions=all_preds, references=all_refs)

print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ DistilGPT2 НА VALIDATION SET")
print("="*80)
print(f"Loss:       {val_loss:.4f}")
print(f"Perplexity: {val_perplexity:.2f}")
print(f"ROUGE-1: {gpt_rouge['rouge1']:.4f}")
print(f"ROUGE-2: {gpt_rouge['rouge2']:.4f}")
print(f"ROUGE-L: {gpt_rouge['rougeL']:.4f}")
print("="*80)


# =============================================================================
# Примеры генерации
# =============================================================================

print("\nПРИМЕРЫ ГЕНЕРАЦИИ DistilGPT2:")
print("="*80)

examples_data = []
for i in range(min(5, num_samples)):
    idx = sample_indices[i]
    prefix, target, _ = val_dataset[idx]
    
    prefix_text = tokenizer.decode(prefix.tolist(), skip_special_tokens=True)
    target_text = tokenizer.decode(target.tolist(), skip_special_tokens=True)
    full_text = prefix_text + " " + target_text
    
    inp, gt, pred = generate_continuation(full_text, max_new_tokens=gpt2_params['max_new_tokens'])
    
    print(f"\nПример {i+1}:")
    print(f"Префикс: {inp}")
    print(f"Ground Truth: {gt}")
    print(f"GPT2: {pred}")
    print("─"*80)
    
    # Сохраняем как словарь
    examples_data.append({
        'example_num': i + 1,
        'prefix': inp,
        'ground_truth': gt,
        'gpt2_generated': pred
    })

# Сохраняем примеры в JSON
with open(PATHS['gpt2_examples'], 'w', encoding='utf-8') as f:
    json.dump(examples_data, f, ensure_ascii=False, indent=2)

print(f"\nПримеры сохранены в: {PATHS['gpt2_examples']}")
print("Готово")