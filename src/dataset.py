from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageCaptioningDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB")
        captions = item['answer']
        caption = captions[torch.randint(0, len(captions), (1,)).item()].replace('\n', ' ').strip()

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values

        inputs = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        text_labels = inputs.input_ids.clone()
        text_labels[text_labels == self.tokenizer.pad_token_id] = -100
        
        query_labels = torch.full((1, self.num_query_tokens), -100)

        combined_labels = torch.cat([query_labels, text_labels], dim=1)

        return {
            "pixel_values": pixel_values.squeeze(),
            "input_ids": inputs.input_ids.squeeze(),   
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": combined_labels.squeeze()
        }