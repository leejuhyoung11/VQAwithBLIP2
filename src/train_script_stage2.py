import os
import torch
import pandas as pd
import yaml, sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM, Blip2Model, BlipImageProcessor, 
    AutoTokenizer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

# From src dir
from model import BLIP2ForPhi, setup_model, select_train_params
from dataset import ImageCaptioningDataset, get_captioning_datasets
from trainer import CustomTrainer

os.environ["HF_TOKEN"] = "HF TOKEN"

BASE_DIR = Path("..").resolve()
CONFIG_DIR = BASE_DIR / "configs"
OUTPUT_DIR = BASE_DIR / "outputs"

config_path = CONFIG_DIR / "config.yaml"

with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

model, image_processor, tokenizer = setup_model(config)
num_trainable_params = select_train_params(model, language_model=False)

print(f"training {num_trainable_params} params...")
print(f"Preparing dataset...")

train_dataset, valid_dataset, train_debug, valid_debug = get_captioning_datasets(config['dataset']['llava'], image_processor, tokenizer, tokenizer_max_length=config['training']['tokenizer_max_length'])
print(f"Train Dataset length: {len(train_dataset)}, Valid Dataset length: {len(valid_dataset)}")

trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(
    trainable_params, 
    lr=float(config['training']['learning_rate']), 
    weight_decay=config['training']['weight_decay']
)

trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        dataset_name='ImageCaptioning-stage2',
        batch_size=config['training']['batch_size'],
        save_dir=config['path']['save_dir'],
        repo_id=config['hf']['repo_id']
    )

print("Starting training...")
trainer.train(
    num_epochs=config['training']['num_epochs'],
    # Add checkpoint path
    resume_from_checkpoint=config['path'].get('stage1_checkpoint'),
    new_stage=True
)
print("Training finished!")