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

import wandb

# From src dir
from model import BLIP2ForPhi, setup_model, select_train_params
from dataset import ImageCaptioningDataset, get_captioning_datasets
from trainer import CustomTrainer, set_seed

os.environ["HF_TOKEN"] = "HF TOKEN"


BASE_DIR = Path("..").resolve()
CONFIG_DIR = BASE_DIR / "configs"
OUTPUT_DIR = BASE_DIR / "outputs"


config_path = CONFIG_DIR / "config.yaml"

with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

wandb.login()

wandb.init(
    project="blip2-project",
    name="ImageCaptioning-stage1",
    config={
        "learning_rate": config['training']['learning_rate'],
        "batch_size": config['training']['batch_size'],
        "epochs": config['training']['num_epochs'],
    }
)


model, image_processor, tokenizer = setup_model(config)
num_trainable_params = select_train_params(model, language_model=False)

print(f"training {num_trainable_params} params...")
print(f"Preparing dataset...")

train_dataset1, valid_dataset1, train_debug1, valid_debug1 = get_captioning_datasets(config['dataset']['cocoCaption'], image_processor, tokenizer, tokenizer_max_length=config['training']['tokenizer_max_length'])
train_dataset2, valid_dataset2, train_debug2, valid_debug2 = get_captioning_datasets(config['dataset']['flickr'], image_processor, tokenizer, tokenizer_max_length=config['training']['tokenizer_max_length'])
train_dataset3, valid_dataset3, train_debug3, valid_debug3 = get_captioning_datasets(config['dataset']['llava'], image_processor, tokenizer, tokenizer_max_length=config['training']['tokenizer_max_length'])
train_dataset4, valid_dataset4, train_debug4, valid_debug4 = get_captioning_datasets(config['dataset']['textCaps'], image_processor, tokenizer, tokenizer_max_length=config['training']['tokenizer_max_length'])

concat_train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3, train_dataset4])
concat_valid_dataset = ConcatDataset([valid_dataset1, valid_dataset2, valid_dataset3, valid_dataset4])


print(f"Train Dataset length: {len(concat_train_dataset)}, Valid Dataset length: {len(concat_valid_dataset)}")


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
        train_dataset=train_debug2,
        val_dataset=train_debug2,
        dataset_name='ImageCaptioning-stage1',
        batch_size=config['training']['batch_size'],
        save_dir_root=config['path']['save_dir'],
        repo_id=config['hf']['repo_id']
    )

print("Starting training...")
trainer.train(
    num_epochs=config['training']['num_epochs'],
)
print("Training finished!")