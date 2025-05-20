"""
This script fine-tunes the LLaMA 2 7B Chat model using LoRA and 4-bit quantization.
It loads a custom dataset of cyber traffic flows, applies LoRA for efficient adaptation,
and saves the fine-tuned model for downstream inference tasks.
"""

# PyTorch for tensor operations and model training
import torch
# Hugging Face datasets for loading JSONL fine-tuning data
from datasets import load_dataset
# Hugging Face Transformers for model/tokenizer setup and training configuration
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
# PEFT library for Low-Rank Adaptation (LoRA) setup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load the fine-tuning dataset from a JSONL file containing instruction-response pairs
print("Loading dataset...")
dataset = load_dataset("json", data_files="../data/FineTuning/training_data.jsonl", split="train")

# Load the tokenizer for the LLaMA 2 chat model (HF version), with remote code trust enabled
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    use_fast=True,
    trust_remote_code=True
)

# Set pad token to EOS token to avoid padding mismatch during training
tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization for faster and smaller model loading
print("Loading 4-bit quantized model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load the LLaMA 2 7B Chat model with 4-bit quantization and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepares the model for k-bit fine-tuning by enabling input gradient checkpointing
print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# Define LoRA config to reduce trainable parameters by focusing on attention projection layers
print("Applying LoRA configuration...")
lora_config = LoraConfig(
    r=16,                      # rank of LoRA (controls bottleneck size)
    lora_alpha=32,             # scaling factor for LoRA
    target_modules=["q_proj", "v_proj"],  # apply LoRA to attention projections
    lora_dropout=0.05,         # dropout for regularization
    bias="none",               # no bias tuning
    task_type="CAUSAL_LM",     # task type: causal language modeling
)

# Wrap the base model with LoRA layers using PEFT
model = get_peft_model(model, lora_config)

# Format each data point into an instruction-response prompt and tokenize it
def generate_and_tokenize_prompt(data_point):
    prompt = f"### Instruction:\n{data_point['input']}\n\n### Response:\n{data_point['output']}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    # Set labels to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    return {k: v.squeeze() for k, v in tokenized.items()}

# Apply prompt formatting and tokenization to the full dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(generate_and_tokenize_prompt)

# Define Hugging Face training configuration including batch size, steps, optimizer, and mixed precision
training_args = TrainingArguments(
    output_dir="../models/llama2-finetuned-cyberagent",  # where to save model checkpoints
    per_device_train_batch_size=2,                      # small batch size per device
    gradient_accumulation_steps=4,                      # to simulate larger effective batch
    warmup_steps=20,                                    # for LR warm-up
    max_steps=500,                                      # total training steps
    learning_rate=2e-4,                                 # learning rate
    fp16=True,                                          # use mixed precision training
    logging_steps=10,                                   # log every 10 steps
    save_steps=250,                                     # save model every 250 steps
    save_total_limit=1,                                 # keep only the latest checkpoint
    optim="paged_adamw_8bit",                           # memory-efficient optimizer
)

# Create a Trainer instance with the model, data, tokenizer, and training settings
print("Setting up Trainer...")
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
)

# Begin the fine-tuning process
print("Starting fine-tuning...")
trainer.train()

# Save the final fine-tuned model and tokenizer for inference
print("Saving final model...")
model.save_pretrained("../models/llama2-finetuned-cyberagent")
tokenizer.save_pretrained("../models/llama2-finetuned-cyberagent")
print("Fine-tuning complete and model saved!")