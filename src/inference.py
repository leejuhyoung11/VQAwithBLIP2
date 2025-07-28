import torch
import yaml
from PIL import Image
from transformers import BlipImageProcessor, AutoTokenizer
import sys, re, os
import pandas as pd
from tqdm import tqdm

from model import BLIP2ForPhi
from train import setup_model

def extract_answer_letter(text):
    match = re.search(r"Answer:\s*([A-Da-d])\b", text)
    return match.group(1).upper() if match else "?"

def main(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model, image_processor, tokenizer = setup_model(config)
    
    print(f"Loading trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device)
    model.eval()

    test = pd.read_csv('../datasets/test.csv')
    results = []


    for _, row in tqdm(test.iterrows(), total=len(test)):
        image = Image.open(os.path.join('../datasets', row['img_path'])).convert("RGB")
        choices = [row[c] for c in ['A', 'B', 'C', 'D']]

        prompt = (
            "You are a helpful AI that answers multiple-choice questions based on the given image.\n" +
            f"Select only the single best answer from A, B, C, or D.\n\n"
            f"Respond only with one letter (e.g., 'A').\n\n"
            f"Question: {row['Question']}\n\n" +
            "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]) +
            "\n\nAnswer:"
            )

        image_inputs = image_processor(image, return_tensors="pt")
        text_inputs = tokenizer(prompt, return_tensors="pt")

        inputs = {
                "pixel_values": image_inputs["pixel_values"].to(device),
                "input_ids": text_inputs["input_ids"].to(device),
                "attention_mask": text_inputs["attention_mask"].to(device)
            }

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        results.append(decoded[0] if decoded else 'C')

    print('Done.')

    submission = pd.read_csv('../datasets/sample_submission.csv')
    submission['answer'] = results
    submission.to_csv('../outputs/predicitons/blip2_phi1.5_submission.csv', index=False)
    print("✅ Done.")


if __name__ == '__main__':

    config_file = '../configs/config.yaml' 
    checkpoint_path = "../outputs/checkpoints/ImageCaptioning-stage4/epoch_4/checkpoint_epoch4.pt"
    main(config_file, checkpoint_path)