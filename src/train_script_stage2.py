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
#     name="ImageCaptioning-stage2",
#     config={
#         "learning_rate": config['training']['stage1']['learning_rate'],
#         "batch_size": config['training']['stage1']['batch_size'],
#         "epochs": config['training']['stage1']['num_epochs'],
#     }
# )


model, image_processor, tokenizer = setup_model(config)
num_trainable_params = select_train_params(model, language_model=False)

print(f"training {num_trainable_params} params...")
print(f"Preparing dataset...")



train_dataset1, valid_dataset1, train_debug1, valid_debug1 = get_vqa_datasets(config['dataset']['vqav2'], image_processor, tokenizer, tokenizer_max_length=config['training']['stage2']['tokenizer_max_length'])
train_dataset2, valid_dataset2, train_debug2, valid_debug2 = get_vqa_datasets(config['dataset']['gqa'], image_processor, tokenizer, tokenizer_max_length=config['training']['stage2']['tokenizer_max_length'])
train_dataset3, valid_dataset3, train_debug3, valid_debug3 = get_vqa_datasets(config['dataset']['llavaNext'], image_processor, tokenizer, tokenizer_max_length=config['training']['stage2']['tokenizer_max_length'])
train_dataset4, valid_dataset4, train_debug4, valid_debug4 = get_llava_datasets(config['dataset']['instruct150'], image_processor, tokenizer, tokenizer_max_length=config['training']['stage2']['tokenizer_max_length'], df_dir=config['path']['instruct_df'], img_dir=config['path']['coco2014'])

subset_of_train_dataset2 = Subset(train_dataset2, indices=range(len(train_dataset2) // 2))
subset_of_valid_dataset2 = Subset(valid_dataset2, indices=range(len(valid_dataset2) // 2))

concat_train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3, train_dataset4])
concat_valid_dataset = ConcatDataset([valid_dataset1, valid_dataset2, valid_dataset3, valid_dataset4])

concat_train_debug = ConcatDataset([train_debug1, train_debug2, train_debug3, train_debug4])
concat_valid_debug = ConcatDataset([valid_debug1, valid_debug2, valid_debug3, valid_debug4])

print(f"Train Dataset length: {len(concat_train_dataset)}, Valid Dataset length: {len(concat_valid_dataset)}")



trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(
    trainable_params, 
    lr=float(config['training']['stage2']['learning_rate']), 
    weight_decay=config['training']['stage2']['weight_decay']
)

trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_dataset=concat_train_dataset,
        val_dataset=concat_valid_dataset,
        dataset_name='ImageCaptioning-stage2',
        batch_size=config['training']['stage2']['batch_size'],
        save_dir_root=config['path']['save_dir']['stage2'],
        # repo_id=config['hf']['repo_id']
    )




print("Starting training...")
trainer.train(
    num_epochs=config['training']['stage2']['num_epochs'],
    # Add checkpoint path
    resume_from_checkpoint=config['path']['stage1_checkpoint'],
    new_stage=True
)
print("Training finished!")