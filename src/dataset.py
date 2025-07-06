from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset


class ImageCaptioningDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128, is_train=True):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        self.is_train = is_train

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomRotation(degrees=15), 
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomVerticalFlip(p=0.4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # 정규화
            ])

        else: 
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB")
        captions = item['answer']
        caption = captions[torch.randint(0, len(captions), (1,)).item()].replace('\n', ' ').strip()

        pixel_values = self.transforms(image)
        # pixel_values = self.image_processor(image, return_tensors="pt").pixel_values

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
    


def get_datasets(dataset_name, config, image_processor, tokenizer):
    raw_datasets = load_dataset(dataset_name)
    # Use only first dict
    raw_dataset = list(raw_datasets.values())[0]

    split_dataset = raw_dataset.train_test_split(test_size=0.2)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = ImageCaptioningDataset(
        train_raw_dataset, 
        image_processor=image_processor, 
        tokenizer=tokenizer,
        max_length=config['training']['tokenizer_max_length'],
        is_train=True
    )
    valid_dataset = ImageCaptioningDataset(
        eval_raw_dataset,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=config['training']['tokenizer_max_length'],
        is_train=False
    )

    train_dataset = ImageCaptioningDataset(train_raw_dataset, image_processor, tokenizer)
    valid_dataset = ImageCaptioningDataset(eval_raw_dataset, image_processor, tokenizer)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    return train_dataset, valid_dataset, train_debug, valid_debug
