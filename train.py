import torch.utils._pytree as _pt
# This MUST happen before anything else
if not hasattr(_pt, 'register_pytree_node'):
    _pt.register_pytree_node = _pt._register_pytree_node

import os
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import login

# HF LOGIN
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# SETTINGS
MODEL_NAME = "distilgpt2" 
DATASET_NAME = "Abirate/english_quotes"
REPO_NAME = "my-fun-quote-generator"

print("--- STARTING CLEAN BUILD RUN ---")
print(f"Loading {MODEL_NAME}...")

dataset = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["quote"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(500))

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    push_to_hub=True,
    hub_model_id=REPO_NAME,
    hub_token=hf_token,
    fp16=True, 
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train,
)

print("Training is starting now...")
trainer.train()

print("Pushing model to your Hugging Face profile...")
trainer.push_to_hub()
tokenizer.push_to_hub(REPO_NAME)

print("✅ SUCCESS: Training finished and model is saved!")
while True:
    time.sleep(60)
