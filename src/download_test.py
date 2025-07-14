from datasets import load_dataset
import os
import torch
import pandas as pd
import yaml, sys
from pathlib import Path

BASE_DIR = Path("..").resolve()
CONFIG_DIR = BASE_DIR / "configs"
OUTPUT_DIR = BASE_DIR / "outputs"


config_path = CONFIG_DIR / "config.yaml"

with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

vqav2 = load_dataset(config['dataset']['vqav2'])
okvqa = load_dataset(config['dataset']['okvqa'])
gqa_q = load_dataset("lmms-lab/GQA", "train_balanced_images")
gqa_i = load_dataset("lmms-lab/GQA", "train_balanced_instructions")
ll = load_dataset(config['dataset']['llavaNext'])