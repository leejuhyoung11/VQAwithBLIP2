import torch
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM, Blip2Model, BlipImageProcessor, 
    AutoTokenizer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset

# From src dir
from model import BLIP2ForPhi, setup_model
from dataset import ImageCaptioningDataset, get_datasets
from trainer import CustomTrainer

def select_train_params(model, qformer=True, projection=True, language_model=True):
    for param in model.vision_model.parameters():
        param.requires_grad = False

    if qformer:
        for param in model.q_former.parameters():
            param.requires_grad = True
    else:
        for param in model.q_former.parameters():
            param.requires_grad = False
    if projection:
        for param in model.projection.parameters():
            param.requires_grad = True
    else:
        for param in model.projection.parameters():
            param.requires_grad = False

    if not language_model:
        for param in model.phi_model.parameters():
            param.requires_grad = False

    trainable_param_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return len(trainable_param_names)


def main(config_path: str):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model, image_processor, tokenizer = setup_model(config)
    trainable_params = select_train_params(model, language_model=False)

    print(f"training {trainable_params} params... Preparing dataset...")

    train_dataset, valid_dataset, train_debug, valid_debug = get_datasets(config['dataset']['image_captioning'], config, image_processor, tokenizer)


    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=float(config['training']['learning_rate']), 
        weight_decay=config['training']['weight_decay']
    )

    # 5. Trainer 
    trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_dataset=train_debug,
        val_dataset=valid_debug,
        batch_size=config['training']['batch_size'],
        save_dir=config['path']['save_dir'],
        repo_id=config['hf']['repo_id']
    )

    # 6. Start Training
    print("Starting training...")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        resume_from_checkpoint=config['path'].get('resume_from_checkpoint') 
    )
    print("Training finished!")

if __name__ == '__main__':
    # 이 스크립트는 src/ 안에 있으므로, 상위 폴더의 configs를 바라보도록 경로 설정
    config_file = '../configs/config.yaml' 
    main(config_file)