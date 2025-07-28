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
from dataset import VQADataset, LlavaInstructDataset, get_vqa_datasets, get_llava_datasets, export_qna_from_conversation
from trainer import CustomTrainer, set_seed

# os.environ["HF_TOKEN"] = ""


BASE_DIR = Path("..").resolve()
CONFIG_DIR = BASE_DIR / "configs"
OUTPUT_DIR = BASE_DIR / "outputs"


config_path = CONFIG_DIR / "config.yaml"

with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

# wandb.login()

# wandb.init(
#     project="blip2-project",
#     name="ImageCaptioning-stage3",
#     config={
#         "learning_rate": config['training']['learning_rate'],
#         "batch_size": config['training']['batch_size'],
#         "epochs": config['training']['num_epochs'],
#     }
# )


model, image_processor, tokenizer = setup_model(config)
num_trainable_params = select_train_params(model, language_model=False)

print(f"training {num_trainable_params} params...")
print(f"Preparing dataset...")


# tokenizer_max_length: 180
llava = load_dataset(config['dataset']['llavaNext'])
subset = llava['train'].select(range(133_300))

train_dataset3, valid_dataset3, train_debug3, valid_debug3 = get_vqa_datasets(subset, image_processor, tokenizer, tokenizer_max_length=config['training']['stage3']['tokenizer_max_length'])



print(f"Train Dataset length: {len(train_dataset3)}, Valid Dataset length: {len(valid_dataset3)}")


trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(
    trainable_params, 
    lr=float(config['training']['stage3']['learning_rate']), 
    weight_decay=config['training']['stage3']['weight_decay']
)

trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_dataset=train_dataset3,
        val_dataset=valid_dataset3,
        dataset_name='ImageCaptioning-stage3',
        batch_size=config['training']['stage3']['batch_size'],
        save_dir_root=config['path']['save_dir']['stage3'],
        # repo_id=config['hf']['repo_id']
    )




print("Starting training...")
trainer.train(
    num_epochs=config['training']['stage3']['num_epochs'],
    # Add checkpoint path
    resume_from_checkpoint=config['path']['stage2_checkpoint'],
    new_stage=True
)
print("Training finished!")