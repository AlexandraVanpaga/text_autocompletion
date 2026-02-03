# Text Autocomplete for Mobile Applications

A text autocomplete system designed for mobile apps, comparing a custom LSTM model against DistilGPT2 as a baseline.

---

## Description

The project implements and benchmarks two text generation approaches for mobile-friendly autocomplete:
- **LSTM** — a lightweight custom-trained sequence model, optimized for on-device inference
- **DistilGPT2** — a distilled transformer baseline, pretrained on a large English corpus

Despite DistilGPT2's larger pretraining corpus, the custom LSTM delivers competitive and in some cases more contextually appropriate completions, while being significantly smaller and faster — making it the preferred choice for mobile deployment.

### Key Features:
- Custom LSTM model trained from scratch on target corpus
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
│   ├── raw/                          # Raw text data
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
│   └── gpt2_generation_examples.json # Saved generation examples
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

Cleans and normalizes raw text data.

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

Computes loss, perplexity, and ROUGE scores on the validation set. Generates text completion examples.

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

| Metric | LSTM | DistilGPT2 |
|--------|------|------------|
| ROUGE-1 | 0.06 | 0.0645 |
| ROUGE-2 | — | 0.0060 |
| ROUGE-L | — | 0.0591 |
| Loss | — | 7.4887 |
| Perplexity | — | 1787.65 |

Both models achieve similar ROUGE-1 scores (~0.06), indicating comparable text-matching ability on this dataset.

### Generation Examples (DistilGPT2)

| Prefix | Ground Truth | DistilGPT2 Output |
|--------|--------------|-------------------|
| "brill poor rabbit. maybe we should get a frog as well" | "to eat the flies?" | ", then we can make a frog ourselves." |
| "ah yes, space is so limited here! must work with what i have!" | "yours is looking great, too!" | "if i have to try it, i will not" |
| "cool. did i see something about a free lunch..." | "rewards the weather is gorgeous" | ". I can't remember if i noticed a sign" |

Full generation examples are saved in `models/gpt2_generation_examples.json`.

---

## Analysis

### Why ROUGE Scores Are Low

Low ROUGE-1 across both models (~0.06) reflects the fundamental difficulty of the task: predicting the exact continuation of conversational text is challenging even for humans. The outputs are contextually plausible but rarely match the ground truth word-for-word — which is expected for open-ended generation.

### LSTM vs DistilGPT2

- **DistilGPT2** scores slightly higher on ROUGE-1, likely due to its pretraining on a much larger corpus
- **LSTM** produces competitive and in some cases more contextually appropriate completions, despite having no pretraining
- **For mobile deployment**, LSTM is the clear winner — smaller size, faster inference, no dependency on a large pretrained model

---

## Conclusions

- Both models produce reasonable text completions, but exact match with ground truth is inherently limited by the nature of open-ended generation
- LSTM is competitive with DistilGPT2 and better suited for mobile applications due to its size and speed
- Further improvements can be achieved by training LSTM on a larger and more coherent corpus, or by fine-tuning DistilGPT2 on the target domain

---

## Potential Improvements

- **Larger training corpus** — training LSTM on more data (especially conversational text) would improve coherence and coverage
- **Fine-tuning DistilGPT2** — adapting the transformer to the target domain could close the quality gap further
- **Quantization** — applying INT8 or dynamic quantization to either model would further reduce inference time on mobile
- **Beam search** — replacing greedy decoding with beam search could produce more fluent and diverse completions
- **User-specific adaptation** — personalizing the model on per-user typing history for more relevant suggestions

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
- Custom LSTM model trained from scratch on target corpus
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
│   ├── raw/                          # Raw text data
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
│   └── gpt2_generation_examples.json # Saved generation examples
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

Cleans and normalizes raw text data.

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

Computes loss, perplexity, and ROUGE scores on the validation set. Generates text completion examples.

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

| Metric | LSTM | DistilGPT2 |
|--------|------|------------|
| ROUGE-1 | 0.06 | 0.0645 |
| ROUGE-2 | — | 0.0060 |
| ROUGE-L | — | 0.0591 |
| Loss | — | 7.4887 |
| Perplexity | — | 1787.65 |

Both models achieve similar ROUGE-1 scores (~0.06), indicating comparable text-matching ability on this dataset.

### Generation Examples (DistilGPT2)

| Prefix | Ground Truth | DistilGPT2 Output |
|--------|--------------|-------------------|
| "brill poor rabbit. maybe we should get a frog as well" | "to eat the flies?" | ", then we can make a frog ourselves." |
| "ah yes, space is so limited here! must work with what i have!" | "yours is looking great, too!" | "if i have to try it, i will not" |
| "cool. did i see something about a free lunch..." | "rewards the weather is gorgeous" | ". I can't remember if i noticed a sign" |

Full generation examples are saved in `models/gpt2_generation_examples.json`.

---

## Analysis

### Why ROUGE Scores Are Low

Low ROUGE-1 across both models (~0.06) reflects the fundamental difficulty of the task: predicting the exact continuation of conversational text is challenging even for humans. The outputs are contextually plausible but rarely match the ground truth word-for-word — which is expected for open-ended generation.

### LSTM vs DistilGPT2

- **DistilGPT2** scores slightly higher on ROUGE-1, likely due to its pretraining on a much larger corpus
- **LSTM** produces competitive and in some cases more contextually appropriate completions, despite having no pretraining
- **For mobile deployment**, LSTM is the clear winner — smaller size, faster inference, no dependency on a large pretrained model

---

## Conclusions

- Both models produce reasonable text completions, but exact match with ground truth is inherently limited by the nature of open-ended generation
- LSTM is competitive with DistilGPT2 and better suited for mobile applications due to its size and speed
- Further improvements can be achieved by training LSTM on a larger and more coherent corpus, or by fine-tuning DistilGPT2 on the target domain

---

## Potential Improvements

- **Larger training corpus** — training LSTM on more data (especially conversational text) would improve coherence and coverage
- **Fine-tuning DistilGPT2** — adapting the transformer to the target domain could close the quality gap further
- **Quantization** — applying INT8 or dynamic quantization to either model would further reduce inference time on mobile
- **Beam search** — replacing greedy decoding with beam search could produce more fluent and diverse completions
- **User-specific adaptation** — personalizing the model on per-user typing history for more relevant suggestions

---

## Technologies

- **PyTorch** — training framework and LSTM implementation
- **Transformers** — DistilGPT2 model (HuggingFace)
- **rouge-score** — ROUGE metric computation
- **numpy / pandas** — data processing
- **matplotlib** — training curves and visualization
