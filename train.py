import os
import torch

# FIX: Manual patch for the 'register_pytree_node' error
import torch.utils._pytree as _pt
if not hasattr(_pt, 'register_pytree_node'):
    _pt.register_pytree_node = _pt._register_pytree_node

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import login

# LOGIN
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# CONFIG
MODEL_NAME = "distilgpt2" 
DATASET_NAME = "Abirate/english_quotes"
REPO_NAME = "my-fun-quote-generator"

print(f"Loading {MODEL_NAME} and dataset...")
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
    per_device_train_batch_size=4, # Lowered for extra safety
    num_train_epochs=1,            # Set to 1 for a quick "fun" test run
    learning_rate=2e-5,
    push_to_hub=True,
    hub_model_id=REPO_NAME,
    hub_token=hf_token,
    fp16=True, 
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train,
)

print("Starting training...")
trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()
tokenizer.push_to_hub(REPO_NAME)
print("Done! You can sleep now.")
import time
while True: time.sleep(60)
