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
from model import BLIP2ForPhi
from dataset import ImageCaptioningDataset
from trainer import CustomTrainer

def setup_model(config: dict, with_lora=True):
    print("Loading model components...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    vision_model_name = config['model']['vision_model_name']
    llm_name = config['model']['llm_name']

    image_processor = BlipImageProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blip2_model = Blip2Model.from_pretrained(vision_model_name)
    vision_model = blip2_model.vision_model
    q_former = blip2_model.qformer
    query_tokens = blip2_model.query_tokens

    phi_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if 'lora' in config['model'] and with_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(**config['model']['lora'])
        phi_model = get_peft_model(phi_model, lora_config)
        phi_model.print_trainable_parameters()
    
    model = BLIP2ForPhi(vision_model, q_former, phi_model, query_tokens)
    
    return model, image_processor, tokenizer

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

    print(f"training {len(trainable_params)} params... Preparing dataset...")

    raw_dataset = load_dataset(config['dataset']['image_captioning'])['val']

    split_dataset = raw_dataset.train_test_split(test_size=0.2)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = ImageCaptioningDataset(
        train_raw_dataset, 
        image_dir=config['data']['image_dir'],
        image_processor=image_processor, 
        tokenizer=tokenizer,
        max_length=config['training']['tokenizer_max_length'],
        is_train=True
    )
    valid_dataset = ImageCaptioningDataset(
        eval_raw_dataset,
        image_dir=config['data']['image_dir'],
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=config['training']['tokenizer_max_length'],
        is_train=False
    )

    train_dataset = ImageCaptioningDataset(train_raw_dataset, image_processor, tokenizer)
    valid_dataset = ImageCaptioningDataset(eval_raw_dataset, image_processor, tokenizer)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )

    # 5. Trainer 
    trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=config['training']['batch_size'],
        save_dir=config['path']['save_dir']
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