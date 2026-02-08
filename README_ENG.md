# Text Autocomplete for Mobile Applications

A text autocomplete system designed for mobile apps, comparing a custom LSTM model against DistilGPT2 as a baseline.

---

## Description

The project implements and benchmarks two text generation approaches for mobile-friendly autocomplete:
- **LSTM** — a lightweight custom-trained sequence model, optimized for on-device inference
- **DistilGPT2** — a distilled transformer baseline, pretrained on a large English corpus

Despite DistilGPT2's larger pretraining corpus, the custom LSTM delivers competitive and in some cases more contextually appropriate completions, while being significantly smaller and faster — making it the preferred choice for mobile deployment.

### Key Features:
- Custom LSTM model trained from scratch on target corpus (Twitter conversational text)
- DistilGPT2 baseline for comparison
- ROUGE-based evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
- Perplexity and cross-entropy loss tracking
- Generation examples with qualitative analysis
- Mobile-first design: lightweight model, fast inference

---

## Project Structure

```
text_autocompletion/
├── data/
│   ├── raw/                          # Raw Twitter text data
│   └── processed/                    # Tokenized and split data
│           ├── train/                # Training set
│           ├── val/                  # Validation set
│           └── test/                 # Test set
├── src/
│   ├── get_raw_data.py               # Raw data download
│   ├── preprocess_data.py            # Text cleaning and preprocessing
│   ├── tokenize_split_data.py        # Tokenization and train/val/test split
│   ├── lstm_model.py                 # LSTM model architecture
│   ├── lstm_train.py                 # LSTM training loop
│   ├── eval_lstm.py                  # LSTM evaluation (loss, ROUGE)
│   └── eval_transformer_pipeline.py  # DistilGPT2 evaluation
├── models/
│   ├── lstm_best.pt                  # Best LSTM checkpoint
│   ├── gpt2_generation_examples.json # DistilGPT2 generation examples
│   └── test_generation_examples.json # LSTM test generation examples
├── results/                          # Plots, metrics, analysis
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- transformers (HuggingFace)
- rouge-score
- numpy, pandas

### Hardware
- Training was performed on NVIDIA GeForce RTX 3060 (27M parameters)
- Evaluation also tested on Google Colab T4

### Installing Dependencies

```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Preprocessing

Cleans and normalizes raw Twitter text data.

```bash
python -m src.preprocess_data
```

### 3. Tokenization and Splitting

Tokenizes the corpus and splits into train/val/test sets.

```bash
python -m src.tokenize_split_data
```

### 4. LSTM Model Definition

Defines the LSTM architecture with embedding layer and output projection.

```bash
python -m src.lstm_model
```

### 5. LSTM Training

Trains the LSTM on the training set with validation-based checkpointing.

```bash
python -m src.lstm_train
```

### 6. LSTM Evaluation

Computes loss, perplexity, and ROUGE scores on the test set. Generates text completion examples.

```bash
python -m src.eval_lstm
```

### 7. DistilGPT2 Evaluation

Runs the same evaluation pipeline on DistilGPT2 for direct comparison.

```bash
python -m src.eval_transformer_pipeline
```

---

## Model Architecture

### LSTM (Primary Model)

```
Input Tokens → Embedding Layer → LSTM Layers → Output Projection → Next Token Prediction
```

- **Parameters:** 27 million
- Lightweight: no pretraining required, trains entirely on the target corpus
- Optimized for fast inference on mobile devices
- Competitive quality with DistilGPT2 on this task

### DistilGPT2 (Baseline)

- Pretrained on a large English corpus
- Used as a reference point for evaluating LSTM performance
- Heavier and slower than LSTM — not suitable for on-device mobile deployment without further distillation or quantization

---

## 📈 Results

### Metrics Comparison

| Metric | LSTM (Test) | DistilGPT2 (Val) |
|--------|-------------|------------------|
| **Loss** | **4.4529** | 7.4887 |
| **Perplexity** | **85.88** | 1787.65 |
| **ROUGE-1** | **0.0728 (±0.1084)** | 0.0645 |
| **ROUGE-2** | **0.0063 (±0.0438)** | 0.0060 |
| **ROUGE-L** | **0.0678 (±0.1028)** | 0.0591 |

- LSTM achieves **lower perplexity (85.88 vs 1787.65)** and **lower loss (4.45 vs 7.49)**, indicating better fit to the target domain
- ROUGE scores are similar across both models (~0.06-0.07), confirming comparable text-matching ability
- Despite similar ROUGE, LSTM completions are often more contextually appropriate

### Generation Examples (LSTM on Test Set)

| Prefix | Ground Truth | LSTM Output | ROUGE-1 |
|--------|--------------|-------------|---------|
| "ok so somehow ashwee managed to make a whole bottle of purple punky explode..." | "maybe it's a sign i need sleep? but i dun wanna." | "good night, it was a great day! my birthday. going to sleep" | 0.250 |
| "i know, i know... but i can't get my" | "hair wet!!!" | "phone on my twitter." | 0.000 |
| "easy week then 98 minutes to go..." | "...... i just wanted to give yall smiles." | ". it's so nice to see how you can be a kid" | 0.105 |
| "i am jealous. all the shows in my area were far away..." | "see this show have a blast!" | "see the movie on bday party" | 0.167 |

### Generation Examples (DistilGPT2 on Validation Set)

| Prefix | Ground Truth | DistilGPT2 Output |
|--------|--------------|-------------------|
| "brill poor rabbit. maybe we should get a frog as well" | "to eat the flies?" | ", then we can make a frog ourselves." |
| "ah yes, space is so limited here! must work with what i have!" | "yours is looking great, too!" | "if i have to try it, i will not" |
| "cool. did i see something about a free lunch..." | "rewards the weather is gorgeous" | ". I can't remember if i noticed a sign" |

Full generation examples are saved in:
- `models/test_generation_examples.json` (LSTM)
- `models/gpt2_generation_examples.json` (DistilGPT2)

---

## Analysis

### Why ROUGE Scores Are Low

Low ROUGE-1 across both models (~0.06-0.07) reflects the fundamental difficulty of the task: predicting the exact continuation of **conversational Twitter text** is challenging even for humans. Tweets are often stream-of-consciousness, non-logical, and highly context-dependent. The outputs are contextually plausible but rarely match the ground truth word-for-word — which is expected for open-ended generation on informal text.

### LSTM vs DistilGPT2

| Aspect | LSTM | DistilGPT2 |
|--------|------|------------|
| **Loss** | 4.45 (better) | 7.49 |
| **Perplexity** | 85.88 (better) | 1787.65 |
| **ROUGE-1** | 0.0728 (slightly better) | 0.0645 |
| **Contextual quality** | Often more appropriate | Sometimes generic |
| **Model size** | Small | Large |
| **Speed** | Fast | Slow |
| **Pretraining** | None (trained on target domain only) | Large corpus |

- **LSTM** outperforms DistilGPT2 on loss and perplexity, indicating better adaptation to the target domain (informal Twitter text)
- **DistilGPT2** benefits from pretraining but is not fine-tuned on this specific corpus, leading to higher perplexity
- **For mobile deployment**, LSTM is the clear winner — smaller size, faster inference, better domain fit

---

## Conclusions

- LSTM delivers relatively sensible text completions on Twitter autocomplete, though perfect matching is inherently difficult on informal, stream-of-consciousness text
- DistilGPT2, thanks to pretraining on a large corpus, produces comparable results but with higher loss and perplexity on this specific domain
- Metrics show similar ROUGE (~0.06-0.07 for both), but LSTM achieves significantly better perplexity (85.88 vs 1787.65)
- For mobile applications, LSTM is preferable due to smaller size and faster inference, especially when further trained on more coherent text and a larger corpus
- Training on RTX 3060 (27M parameters) was significantly faster than on Colab T4 for both training and evaluation

---

## Potential Improvements

- **Larger and more coherent training corpus** — training LSTM on formal conversational text (e.g., customer service chats, SMS) would improve coherence and reduce perplexity
- **Fine-tuning DistilGPT2** — adapting the transformer to the target domain (Twitter) could close the perplexity gap
- **Quantization** — applying INT8 or dynamic quantization to either model would further reduce inference time on mobile
- **Beam search** — replacing greedy decoding with beam search could produce more fluent and diverse completions
- **User-specific adaptation** — personalizing the model on per-user typing history for more relevant suggestions
- **Domain adaptation** — training on multiple text domains (SMS, email, social media) to generalize better across use cases

---

## Technologies

- **PyTorch** — training framework and LSTM implementation
- **Transformers** — DistilGPT2 model (HuggingFace)
- **rouge-score** — ROUGE metric computation
- **numpy / pandas** — data processing
- **matplotlib** — training curves and visualization
EOF
Salida

# Text Autocomplete for Mobile Applications

A text autocomplete system designed for mobile apps, comparing a custom LSTM model against DistilGPT2 as a baseline.

---

## Description

The project implements and benchmarks two text generation approaches for mobile-friendly autocomplete:
- **LSTM** — a lightweight custom-trained sequence model, optimized for on-device inference
- **DistilGPT2** — a distilled transformer baseline, pretrained on a large English corpus

Despite DistilGPT2's larger pretraining corpus, the custom LSTM delivers competitive and in some cases more contextually appropriate completions, while being significantly smaller and faster — making it the preferred choice for mobile deployment.

### Key Features:
- Custom LSTM model trained from scratch on target corpus (Twitter conversational text)
- DistilGPT2 baseline for comparison
- ROUGE-based evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
- Perplexity and cross-entropy loss tracking
- Generation examples with qualitative analysis
- Mobile-first design: lightweight model, fast inference

---

## Project Structure

```
text_autocompletion/
├── data/
│   ├── raw/                          # Raw Twitter text data
│   └── processed/                    # Tokenized and split data
│           ├── train/                # Training set
│           ├── val/                  # Validation set
│           └── test/                 # Test set
├── src/
│   ├── get_raw_data.py               # Raw data download
│   ├── preprocess_data.py            # Text cleaning and preprocessing
│   ├── tokenize_split_data.py        # Tokenization and train/val/test split
│   ├── lstm_model.py                 # LSTM model architecture
│   ├── lstm_train.py                 # LSTM training loop
│   ├── eval_lstm.py                  # LSTM evaluation (loss, ROUGE)
│   └── eval_transformer_pipeline.py  # DistilGPT2 evaluation
├── models/
│   ├── lstm_best.pt                  # Best LSTM checkpoint
│   ├── gpt2_generation_examples.json # DistilGPT2 generation examples
│   └── test_generation_examples.json # LSTM test generation examples
├── results/                          # Plots, metrics, analysis
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- transformers (HuggingFace)
- rouge-score
- numpy, pandas

### Hardware
- Training was performed on NVIDIA GeForce RTX 3060 (27M parameters)
- Evaluation also tested on Google Colab T4

### Installing Dependencies

```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Preprocessing

Cleans and normalizes raw Twitter text data.

```bash
python -m src.preprocess_data
```

### 3. Tokenization and Splitting

Tokenizes the corpus and splits into train/val/test sets.

```bash
python -m src.tokenize_split_data
```

### 4. LSTM Model Definition

Defines the LSTM architecture with embedding layer and output projection.

```bash
python -m src.lstm_model
```

### 5. LSTM Training

Trains the LSTM on the training set with validation-based checkpointing.

```bash
python -m src.lstm_train
```

### 6. LSTM Evaluation

Computes loss, perplexity, and ROUGE scores on the test set. Generates text completion examples.

```bash
python -m src.eval_lstm
```

### 7. DistilGPT2 Evaluation

Runs the same evaluation pipeline on DistilGPT2 for direct comparison.

```bash
python -m src.eval_transformer_pipeline
```

---

## Model Architecture

### LSTM (Primary Model)

```
Input Tokens → Embedding Layer → LSTM Layers → Output Projection → Next Token Prediction
```

- **Parameters:** 27 million
- Lightweight: no pretraining required, trains entirely on the target corpus
- Optimized for fast inference on mobile devices
- Competitive quality with DistilGPT2 on this task

### DistilGPT2 (Baseline)

- Pretrained on a large English corpus
- Used as a reference point for evaluating LSTM performance
- Heavier and slower than LSTM — not suitable for on-device mobile deployment without further distillation or quantization

---

## 📈 Results

### Metrics Comparison

| Metric | LSTM (Test) | DistilGPT2 (Val) |
|--------|-------------|------------------|
| **Loss** | **4.4529** | 7.4887 |
| **Perplexity** | **85.88** | 1787.65 |
| **ROUGE-1** | **0.0728 (±0.1084)** | 0.0645 |
| **ROUGE-2** | **0.0063 (±0.0438)** | 0.0060 |
| **ROUGE-L** | **0.0678 (±0.1028)** | 0.0591 |

- LSTM achieves **lower perplexity (85.88 vs 1787.65)** and **lower loss (4.45 vs 7.49)**, indicating better fit to the target domain
- ROUGE scores are similar across both models (~0.06-0.07), confirming comparable text-matching ability
- Despite similar ROUGE, LSTM completions are often more contextually appropriate

### Generation Examples (LSTM on Test Set)

| Prefix | Ground Truth | LSTM Output | ROUGE-1 |
|--------|--------------|-------------|---------|
| "ok so somehow ashwee managed to make a whole bottle of purple punky explode..." | "maybe it's a sign i need sleep? but i dun wanna." | "good night, it was a great day! my birthday. going to sleep" | 0.250 |
| "i know, i know... but i can't get my" | "hair wet!!!" | "phone on my twitter." | 0.000 |
| "easy week then 98 minutes to go..." | "...... i just wanted to give yall smiles." | ". it's so nice to see how you can be a kid" | 0.105 |
| "i am jealous. all the shows in my area were far away..." | "see this show have a blast!" | "see the movie on bday party" | 0.167 |

### Generation Examples (DistilGPT2 on Validation Set)

| Prefix | Ground Truth | DistilGPT2 Output |
|--------|--------------|-------------------|
| "brill poor rabbit. maybe we should get a frog as well" | "to eat the flies?" | ", then we can make a frog ourselves." |
| "ah yes, space is so limited here! must work with what i have!" | "yours is looking great, too!" | "if i have to try it, i will not" |
| "cool. did i see something about a free lunch..." | "rewards the weather is gorgeous" | ". I can't remember if i noticed a sign" |

Full generation examples are saved in:
- `models/test_generation_examples.json` (LSTM)
- `models/gpt2_generation_examples.json` (DistilGPT2)

---

## Analysis

### Why ROUGE Scores Are Low

Low ROUGE-1 across both models (~0.06-0.07) reflects the fundamental difficulty of the task: predicting the exact continuation of **conversational Twitter text** is challenging even for humans. Tweets are often stream-of-consciousness, non-logical, and highly context-dependent. The outputs are contextually plausible but rarely match the ground truth word-for-word — which is expected for open-ended generation on informal text.

### LSTM vs DistilGPT2

| Aspect | LSTM | DistilGPT2 |
|--------|------|------------|
| **Loss** | 4.45 (better) | 7.49 |
| **Perplexity** | 85.88 (better) | 1787.65 |
| **ROUGE-1** | 0.0728 (slightly better) | 0.0645 |
| **Contextual quality** | Often more appropriate | Sometimes generic |
| **Model size** | Small | Large |
| **Speed** | Fast | Slow |
| **Pretraining** | None (trained on target domain only) | Large corpus |

- **LSTM** outperforms DistilGPT2 on loss and perplexity, indicating better adaptation to the target domain (informal Twitter text)
- **DistilGPT2** benefits from pretraining but is not fine-tuned on this specific corpus, leading to higher perplexity
- **For mobile deployment**, LSTM is the clear winner — smaller size, faster inference, better domain fit

---

## Conclusions

- LSTM delivers relatively sensible text completions on Twitter autocomplete, though perfect matching is inherently difficult on informal, stream-of-consciousness text
- DistilGPT2, thanks to pretraining on a large corpus, produces comparable results but with higher loss and perplexity on this specific domain
- Metrics show similar ROUGE (~0.06-0.07 for both), but LSTM achieves significantly better perplexity (85.88 vs 1787.65)
- For mobile applications, LSTM is preferable due to smaller size and faster inference, especially when further trained on more coherent text and a larger corpus
- Training on RTX 3060 (27M parameters) was significantly faster than on Colab T4 for both training and evaluation

---

## Potential Improvements

- **Larger and more coherent training corpus** — training LSTM on formal conversational text (e.g., customer service chats, SMS) would improve coherence and reduce perplexity
- **Fine-tuning DistilGPT2** — adapting the transformer to the target domain (Twitter) could close the perplexity gap
- **Quantization** — applying INT8 or dynamic quantization to either model would further reduce inference time on mobile
- **Beam search** — replacing greedy decoding with beam search could produce more fluent and diverse completions
- **User-specific adaptation** — personalizing the model on per-user typing history for more relevant suggestions
- **Domain adaptation** — training on multiple text domains (SMS, email, social media) to generalize better across use cases

---

## Technologies

- **PyTorch** — training framework and LSTM implementation
- **Transformers** — DistilGPT2 model (HuggingFace)
- **rouge-score** — ROUGE metric computation
- **numpy / pandas** — data processing
- **matplotlib** — training curves and visualization
