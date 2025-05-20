# CyberAgent: Detecting Botnet Activity Using Fine-Tuned LLaMA 2

This project presents a hybrid approach to detecting botnet activity in network traffic using a combination of traditional machine learning and a fine-tuned large language model (LLM). The solution leverages the CTU-13 dataset and applies both Random Forest and a LoRA fine-tuned LLaMA 2 7B chat model for classification and explanation of network flows.

---

## 🚀 Project Overview

The goal is to build an intelligent agent capable of:

- Classifying network flows as **normal** or **botnet activity**
- Generating **natural language explanations** for each classification
- Providing a **command-line interface** (UI) for manual or batch evaluation
- Supporting **real-time demonstration** and model evaluation

---

## 📂 Repository Structure

```
.
├── data/
│   ├── Filtered/            # Filtered CTU-13 flows (CSV)
│   └── FineTuning/          # Training data for LLaMA2 (JSONL)
├── models/                  # Saved fine-tuned model checkpoints
├── scripts/
│   ├── build_training_prompts.py  # Generates instruction-output training pairs
│   ├── train.py                   # Fine-tunes LLaMA2 with LoRA
│   ├── predict.py                 # CLI-based real-time inference tool
│   └── evaluate_models.py        # Compares baseline and SOTA performance
└── README.md
```

---

## 📊 Dataset

- **Source**: [CTU-13 Botnet Dataset](https://www.stratosphereips.org/datasets-ctu13)
- Labeled flows: `From-Botnet` vs. `From-Normal`
- Used to train both classic ML models and generate prompts for LLaMA2

---

## 🧠 Models

### 🔹 Baseline: Random Forest
- Trained on numerical features from the CTU-13 flows
- Fast and interpretable, but lacks contextual understanding

### 🔸 SOTA: LLaMA 2 7B + LoRA
- Fine-tuned on flow metadata → instruction/output format
- Supports 4-bit quantization (nf4)
- Generates classification + reasoning text per flow

---

## 🛠️ Installation

### ✅ Prerequisites

- Python 3.10+
- Git, pip
- GPU recommended (for training/inference)

### 🔧 1. Clone the Project

```bash
git clone https://github.com/Anthonios-Deeb/AI-Cyber-Agent-for-Good
cd cyberagent-llama2
```

### 📦 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Training the LLM Agent

To fine-tune the LLaMA 2 model with LoRA:

```bash
python scripts/fine_tune_llama.py
```

- Model checkpoints saved to: `models/llama2-finetuned-cyberagent`

---

## 📈 Evaluating Models

To compare Random Forest (baseline) vs. fine-tuned LLaMA:

```bash
python scripts/evaluate_models.py
```

- Evaluates both models on the same test set
- Metrics: accuracy, precision, recall, F1

---

## 💡 Real-Time Inference (Terminal UI)

Run the real-time CLI predictor:

```bash
python scripts/predict.py
```

Options:
- `1`: Enter flows manually
- `2`: Auto-sample random flows from CTU-13 and predict

---

## 📸 Live Demo

- Terminal screenshots included in presentation
- Examples shown for both `normal` and `botnet` responses with LLM explanations

---

## ⚙️ LoRA Fine-Tuning Config

| Param         | Value       |
|---------------|-------------|
| `r`           | 16          |
| `alpha`       | 32          |
| Target modules| `q_proj`, `v_proj` |
| `dropout`     | 0.05        |
| Optimizer     | `paged_adamw_8bit` |
| Steps         | 500         |
| Precision     | `fp16`      |

---

## 📊 Baseline vs. SOTA Summary

| Model           | Pros                                 | Cons                                     |
|-----------------|---------------------------------------|------------------------------------------|
| Random Forest   | Fast, easy to interpret              | No explanation, relies on fixed features |
| LLaMA 2 + LoRA  | Explanations, adaptable, generalizes | Slower, needs GPU                        |

---

## 🔗 Useful Links

- [CTU-13 Dataset](https://www.stratosphereips.org/datasets-ctu13)
- [LLaMA 2 HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
