from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
from tqdm import tqdm

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
        try:
            item = self.dataset[idx]
            image = item['image'].convert("RGB")
            
            #1 cocoCaption
            if 'answer' in item and item['answer']:
                captions = item['answer']
                caption = captions[torch.randint(0, len(captions), (1,)).item()].replace('\n', ' ').strip()
            #2 flickr30k
            elif 'caption' in item and item['caption']:
                captions_list = item.get('caption')
                if captions_list:
                    valid_captions = [c.strip().replace('\n', ' ') for c in captions_list if c and c.strip()]
                if valid_captions:
                    caption = valid_captions[torch.randint(0, len(valid_captions), (1,)).item()]
            #LLaVa-Recap
            elif 'conversations' in item and item['conversations']:
                for turn in item['conversations']:
                    if turn.get('from') == 'gpt':
                        caption = turn.get('value')
                        break
            
            if caption is None or image is None:
                print(f"No Element in {idx}")
                return None
            
            
            pixel_values = self.transforms(image)
            
            prompt = (f"Describe the image:")
            len_of_prompt = len(self.tokenizer(prompt)['input_ids'])

            inputs = self.tokenizer(
                prompt + caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            text_labels = inputs.input_ids.clone()
            text_labels[0, :len_of_prompt] = -100
            text_labels[text_labels == self.tokenizer.pad_token_id] = -100
            
            query_labels = torch.full((1, self.num_query_tokens), -100)

            combined_labels = torch.cat([query_labels, text_labels], dim=1)

            return {
                "pixel_values": pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),   
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": combined_labels.squeeze()
            }
        except Exception as e:
            print(f"Whil processing index {idx} , error occured ({e}), Skip Element.")
            return None


class VQADataset(Dataset):
    def __init__(self, hf_dataset, image_processor, tokenizer, num_query_tokens=32, max_length=128, is_train=True, image_dataset=None):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_query_tokens = num_query_tokens
        self.is_train = is_train
        self.image_dataset = image_dataset

        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
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
        try:
            item = self.dataset[idx]

            #GQA Dataset
            if self.image_dataset:
                image_id = item['imageId']
                image = self.image_dataset[image_id].convert("RGB")
                question = item['question']
                answer = item['fullAnswer']

            else:
                image = item['image'].convert("RGB")
                question = item['question']
                #VQAV2 Dataset
                if 'multiple_choice_answer' in item and item['multiple_choice_answer']:
                    answer = item['multiple_choice_answer']
                # OKVQA Dataset
                else:
                    if not item['answers']:
                        return None
                    answers = item['answers']
                    answer_counts = Counter(answers)
                    # Choose most common answer from the list
                    most_common_answer = answer_counts.most_common(1)
                    if most_common_answer:
                        answer = most_common_answer[0][0]
                    else:
                        return None
            
            pixel_values = self.transforms(image)

            prompt = (f"Question: {question}\n" +"Answer:")
            len_of_prompt = len(self.tokenizer(prompt)['input_ids'])

            inputs = self.tokenizer(
                prompt + answer,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )
            
            answer_tokens = inputs.input_ids.clone()
            answer_tokens[0, :len_of_prompt] = -100
            answer_tokens[answer_tokens == self.tokenizer.pad_token_id] = -100

            query_labels = torch.full((1, self.num_query_tokens), -100)
            combined_labels = torch.cat([query_labels, answer_tokens], dim=1)


            return {
                "pixel_values": pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),   
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": combined_labels.squeeze()
            }
        except Exception as e:
            print(f"Whil processing index {idx} , error occured ({e}), Skip Element.")
            return None





def get_captioning_datasets(dataset_name, image_processor, tokenizer, tokenizer_max_length=128):
    raw_datasets = load_dataset(dataset_name)
    # Use only first dict
    raw_dataset = list(raw_datasets.values())[0]

    split_dataset = raw_dataset.train_test_split(test_size=0.2)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = ImageCaptioningDataset(train_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length)
    valid_dataset = ImageCaptioningDataset(eval_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    return train_dataset, valid_dataset, train_debug, valid_debug


def get_vqa_datasets(dataset_name, image_processor, tokenizer, tokenizer_max_length=128):
    
    id_to_image = None
    # images and quesitons are seperated
    if isinstance(dataset_name, list):
        image_dataset = load_dataset(dataset_name[0], dataset_name[1][0])
        image_dataset = list(image_dataset.values())[0]
        id_to_image = {item['id']: item['image'] for item in tqdm(image_dataset, desc="Building image dictionary")}
        raw_datasets = load_dataset(dataset_name[0], dataset_name[1][1])
        raw_dataset = list(raw_datasets.values())[0]
        
    else:
        raw_datasets = load_dataset(dataset_name)
        # Use only first dict
        raw_dataset = list(raw_datasets.values())[0]
        image_dataset = None
    
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_raw_dataset = split_dataset['train']
    eval_raw_dataset = split_dataset['test']

    train_dataset = VQADataset(train_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    valid_dataset = VQADataset(eval_raw_dataset, image_processor, tokenizer, max_length=tokenizer_max_length, image_dataset=id_to_image)
    train_debug = Subset(train_dataset, indices=range(50))
    valid_debug = Subset(valid_dataset, indices=range(50))

    return train_dataset, valid_dataset, train_debug, valid_debug
